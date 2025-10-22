"""Helper functions for handling OOM situations in PPO trainer workflows."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.workgroup_oom_guard import WorkgroupOOMGuard

from pprint import pprint

# add current directory to sys.path for local imports
sys.path.append(str(Path(__file__).resolve().parents[4]))
from llm import OpenAIGPT5Client # type: ignore  # optional dependency

if TYPE_CHECKING:
    from .ray_trainer import RayPPOTrainer


_MINI_BATCH_PATH = "actor_rollout_ref.actor.ppo_mini_batch_size"
_MICRO_BATCH_PATH = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
_ALLOWED_PARAMETER_PATHS = {_MINI_BATCH_PATH, _MICRO_BATCH_PATH}


def _default_prompt_path() -> Path:
    return Path(__file__).resolve().parents[4] / "prompt.txt"


@lru_cache(maxsize=1)
def _load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found at {path}")
    return path.read_text(encoding="utf-8")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop opening fence
        lines = lines[1:]
        for idx, line in enumerate(lines):
            if line.strip().startswith("```"):
                stripped = "\n".join(lines[:idx])
                break
        else:
            stripped = "\n".join(lines)
    return stripped.strip()


def _parse_llm_response(response: str) -> dict:
    candidate = _strip_code_fence(response)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as json_error:
        raise ValueError(f"LLM response is not valid JSON: {json_error}") from json_error


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _build_context_prompt(
    *,
    template: str,
    error_message: str,
    mini_batch_size: int,
    micro_batch_size: int,
    attempt: int | None,
    max_retries: int | None,
) -> str:
    attempt_text = str(attempt) if attempt is not None else "unknown"
    max_text = str(max_retries) if max_retries is not None else "unknown"
    context = (
        f"GPU OOM detected while updating PPO actor. Attempt {attempt_text} / {max_text}.\n"
        f"Current mini batch size: {mini_batch_size}.\n"
        f"Current micro batch size per GPU: {micro_batch_size}.\n"
        f"Error message:\n{error_message}"
    )
    try:
        return template.format(context=context)
    except KeyError as key_error:
        raise ValueError(
            "Prompt template must contain a '{context}' placeholder to inject runtime details."
        ) from key_error


def build_actor_update_oom_handler(
    trainer: "RayPPOTrainer", guard: WorkgroupOOMGuard
) -> Callable[[DataProto, BaseException], Optional[DataProto]]:
    """Create an OOM handler that reduces actor batch sizes and logs diagnostics.

    Args:
        trainer: The PPO trainer orchestrating the workload.
        guard: The OOM guard invoking the handler. Used to build a placeholder batch.

    Returns:
        Callable invoked by the OOM guard when recovery fails. It updates the trainer
        configuration before returning a placeholder ``DataProto``.
    """

    def _handler(
        batch: DataProto,
        exc: BaseException,
        *,
        attempt: int | None = None,
        max_retries: int | None = None,
        guard: WorkgroupOOMGuard | None = None,
        final: bool = False,
    ) -> Optional[DataProto]:
        path = _MINI_BATCH_PATH
        current = OmegaConf.select(trainer.config, path)
        new_value = current

        if isinstance(current, int) and current > 1:
            new_value = max(1, current // 2)
            if new_value == current and current > 1:
                new_value = current - 1

        if isinstance(new_value, int) and new_value < current:
            print(
                f"[OOMHandler] actor.update_actor OOM detected. "
                f"Reducing PPO mini batch size from {current} to {new_value}."
            )
            trainer.update_config(path, new_value)

            micro_path = _MICRO_BATCH_PATH
            micro_val = OmegaConf.select(trainer.config, micro_path)
            if isinstance(micro_val, int) and micro_val > new_value:
                print(
                    f"[OOMHandler] Adjusting actor micro_batch_size_per_gpu from {micro_val} to {new_value} "
                    "to stay within the reduced mini-batch size."
                )
                trainer.update_config(micro_path, new_value)

            # Allow retries with adjusted configuration
            return None

        print(
            "[OOMHandler] actor.update_actor OOM detected but PPO mini batch size cannot be reduced further."
        )

        target_guard = guard or trainer.actor_update_oom_guard
        return target_guard._build_aborted_dataprotos(batch, exc)

    return _handler


def build_llm_actor_update_oom_handler(
    trainer: "RayPPOTrainer",
    guard: WorkgroupOOMGuard,
    *,
    prompt_path: Path | None = None,
    resource_group: str = "msrvV2",
) -> Callable[[DataProto, BaseException], Optional[DataProto]]:
    """Create an OOM handler that delegates batch-size tuning to an LLM.

    The handler reads a prompt template, injects runtime OOM details, and
    requests structured commands from an Azure OpenAI deployment. Any proposed
    configuration updates are validated and applied to the trainer before
    allowing the OOM guard to retry the workload.
    """
    prompt_location = prompt_path or _default_prompt_path()
    fallback_handler = build_actor_update_oom_handler(trainer, guard)
    client: OpenAIGPT5Client | None = None

    def _ensure_client() -> OpenAIGPT5Client:
        nonlocal client
        if client is None:
            client = OpenAIGPT5Client()
        return client

    def _handler(
        batch: DataProto,
        exc: BaseException,
        *,
        attempt: int | None = None,
        max_retries: int | None = None,
        guard: WorkgroupOOMGuard | None = None,
        final: bool = False,
    ) -> Optional[DataProto]:
        target_guard = guard or trainer.actor_update_oom_guard

        try:
            template = _load_prompt_template(prompt_location)
        except FileNotFoundError as missing_prompt:
            print(
                f"[OOMHandler] Prompt template missing ({missing_prompt}). Falling back to deterministic reduction."
            )
            return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

        try:
            mini_batch = OmegaConf.select(trainer.config, _MINI_BATCH_PATH)
            micro_batch = OmegaConf.select(trainer.config, _MICRO_BATCH_PATH)
        except Exception as cfg_error:
            print(f"[OOMHandler] Failed to read batch configuration: {cfg_error}. Aborting current batch.")
            return target_guard._build_aborted_dataprotos(batch, exc)

        if not isinstance(mini_batch, int) or not isinstance(micro_batch, int):
            print(
                f"[OOMHandler] Batch configuration values must be integers (mini={mini_batch}, micro={micro_batch})."
            )
            return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

        prompt = _build_context_prompt(
            template=template,
            error_message=_format_error(exc),
            mini_batch_size=int(mini_batch),
            micro_batch_size=int(micro_batch),
            attempt=attempt,
            max_retries=max_retries,
        )

        # store prompt for debugging
        debug_path = Path(f"./logs/prompt_{attempt}.txt")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(prompt, encoding="utf-8")
        print(f"[OOMHandler] Stored prompt to {debug_path.resolve()}")

        try:
            response = _ensure_client().query_full_prompt(prompt)
            pprint(f"[OOMHandler] LLM response: {response!r}")
        except Exception as query_error:
            print(
                f"[OOMHandler] Azure OpenAI query failed ({query_error}). Falling back to deterministic reduction."
            )
            return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

        try:
            response_dict = _parse_llm_response(response)
        except ValueError as parse_error:
            print(
                f"[OOMHandler] Could not parse LLM response: {parse_error}. Response was: {response!r}."
            )
            return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

        commands = response_dict.get("commands", [])
        if not isinstance(commands, list) or not commands:
            print(
                f"[OOMHandler] LLM response missing 'commands' list. Response: {response_dict}."
            )
            return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

        applied_updates = False
        for command in commands:
            if not isinstance(command, dict):
                continue
            parameter_path = command.get("parameter_path")
            value = command.get("value")
            if parameter_path not in _ALLOWED_PARAMETER_PATHS:
                print(
                    f"[OOMHandler] Ignoring unsupported parameter '{parameter_path}'. Allowed: {_ALLOWED_PARAMETER_PATHS}."
                )
                continue
            if not isinstance(value, int) or value < 1:
                print(f"[OOMHandler] Ignoring invalid value for {parameter_path}: {value}")
                continue
            current_value = OmegaConf.select(trainer.config, parameter_path)
            if current_value == value:
                continue
            print(
                f"[OOMHandler] Applying LLM suggestion: {parameter_path} {current_value} -> {value}."
            )
            trainer.update_config(parameter_path, value)
            applied_updates = True

        reason = response_dict.get("reason")
        if isinstance(reason, str) and reason:
            print(f"[OOMHandler] LLM rationale: {reason}")

        if applied_updates:
            return None

        print("[OOMHandler] No valid updates from LLM. Falling back to deterministic reduction.")
        return fallback_handler(batch, exc, attempt=attempt, max_retries=max_retries, guard=target_guard, final=final)

    return _handler
