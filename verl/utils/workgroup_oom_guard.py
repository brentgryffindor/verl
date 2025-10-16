"""Utilities for wrapping WorkGroup methods with GPU OOM recovery.

This module provides a reusable wrapper that can be installed on any worker group
method (e.g. ``generate_sequences``) to catch out-of-memory errors raised by the
underlying workers. When an OOM happens, the wrapper retries the invocation,
optionally splits the incoming batch into smaller chunks, and concatenates the
successful results. If the OOM persists for a single sample, it produces an
aborted placeholder so the caller can continue without crashing.

The wrapper is intentionally decoupled from any specific trainer so it can be
reused across different worker groups.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from ray.exceptions import RayActorError, RayTaskError

from verl import DataProto

__all__ = ["WorkgroupOOMGuard", "wrap_generate_sequences_with_oom_guard"]

logger = logging.getLogger(__name__)

_OOM_KEYWORDS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas error: out of memory",
    "cudnn: cudnn_status_not_supported",
    "hip out of memory",
    "out of global memory",
    "failed to allocate",
    "resource exhausted: out of memory",
    "unhandled error in worker: out of memory",
)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Collect the chain of exceptions (cause/context) for inspection."""
    chain = []
    visited: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in visited:
        chain.append(current)
        visited.add(id(current))
        ray_cause = getattr(current, "cause", None)
        if isinstance(ray_cause, BaseException):
            current = ray_cause
            continue
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None:
            current = current.__context__
        else:
            current = None
    return chain


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if the exception chain indicates a GPU OOM."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, (RayTaskError, RayActorError)):
        # Ray wraps the original exception; inspect recursively.
        cause_accessor = getattr(exc, "as_instanceof_cause", None)
        if callable(cause_accessor):
            try:
                cause = cause_accessor()
            except Exception:  # pragma: no cover - defensive
                cause = None
            if isinstance(cause, BaseException):
                return _is_oom_error(cause)
    for chained in _exception_chain(exc):
        if isinstance(chained, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(chained, RuntimeError):
            message = " ".join(str(arg).lower() for arg in chained.args)
            if any(keyword in message for keyword in _OOM_KEYWORDS):
                return True
        message = str(chained).lower()
        if any(keyword in message for keyword in _OOM_KEYWORDS):
            return True
    return False


def _format_exception(exc: BaseException) -> str:
    """Render a concise error string for telemetry."""
    root = _exception_chain(exc)[0]
    return f"{type(root).__name__}: {root}"


@dataclass
class OOMRecoveryStats:
    """Track wrapper level statistics for observability."""

    total_retries: int = 0
    total_splits: int = 0
    recovered_batches: int = 0
    aborted_samples: int = 0


class WorkgroupOOMGuard:
    """Wrap a worker group method to recover from GPU OOM errors.

    The guard retries the method on OOM, attempts to split the incoming batch into
    smaller pieces, and concatenates the successful outputs. It exposes statistics
    on how often recovery happens to help with monitoring.
    """

    def __init__(
        self,
        workgroup: Any,
        method_name: str = "generate_sequences",
        *,
        max_retries: int = 1,
        min_chunk_size: int = 1,
        on_aborted: Optional[Callable[[DataProto, BaseException], DataProto]] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.workgroup = workgroup
        self.method_name = method_name
        self._original_method = getattr(workgroup, method_name)
        self.max_retries = max(0, max_retries)
        self.min_chunk_size = max(1, min_chunk_size)
        self._on_aborted = on_aborted or self._build_aborted_dataprotos
        self.stats = OOMRecoveryStats()

        self._logger = log or logger
        self._install_wrapper()

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def unwrap(self) -> None:
        """Restore the original unwrapped method."""
        setattr(self.workgroup, self.method_name, self._original_method)

    # --------------------------------------------------------------------- #
    # Internal implementation
    # --------------------------------------------------------------------- #
    def _install_wrapper(self) -> None:
        original = self._original_method

        guard = self

        def wrapped(batch: DataProto, *args, **kwargs):
            return guard._invoke_with_recovery(original, batch, *args, **kwargs)

        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
        setattr(self.workgroup, self.method_name, wrapped)

    def _invoke_with_recovery(self, fn, batch: DataProto, *args, **kwargs) -> DataProto:
        try:
            return self._call_with_retries(fn, batch, *args, **kwargs)
        except BaseException as exc:
            if not _is_oom_error(exc):
                raise

            self._logger.warning(
                "[OOMGuard] %s.%s encountered OOM after retries; attempting to split batch "
                "(size=%d). Error: %s",
                type(self.workgroup).__name__,
                self.method_name,
                len(batch),
                _format_exception(exc),
            )
            self.stats.total_splits += 1

            if len(batch) <= self.min_chunk_size:
                aborted = self._on_aborted(batch, exc)
                self.stats.aborted_samples += len(batch)
                return aborted

            split_size = self._compute_split_size(len(batch))
            if split_size is None:
                aborted = self._on_aborted(batch, exc)
                self.stats.aborted_samples += len(batch)
                return aborted

            sub_batches = batch.split(split_size)
            outputs: list[DataProto] = []
            for sub_batch in sub_batches:
                outputs.append(self._invoke_with_recovery(fn, sub_batch, *args, **kwargs))

            recovered = DataProto.concat(outputs)
            recovered.meta_info = self._merge_meta_info(outputs, recovered.meta_info)
            recovered.meta_info.setdefault("oom_guard", {})["num_splits"] = (
                recovered.meta_info.get("oom_guard", {}).get("num_splits", 0) + 1
            )
            self.stats.recovered_batches += 1
            return recovered

    def _call_with_retries(self, fn, batch: DataProto, *args, **kwargs) -> DataProto:
        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                if attempt > 0:
                    self._logger.info(
                        "[OOMGuard] Retry %d/%d for %s.%s (batch_size=%d)",
                        attempt,
                        self.max_retries,
                        type(self.workgroup).__name__,
                        self.method_name,
                        len(batch),
                    )
                    self.stats.total_retries += 1
                return fn(batch, *args, **kwargs)
            except BaseException as exc:
                last_exc = exc
                if not _is_oom_error(exc):
                    raise
        assert last_exc is not None
        raise last_exc

    def _compute_split_size(self, length: int) -> Optional[int]:
        """Choose a split size that reduces batch size while staying above the minimum."""
        if length <= 1:
            return None
        split_size = max(self.min_chunk_size, length // 2)
        if split_size >= length:
            split_size = length - 1
        return split_size if split_size >= 1 else None

    def _merge_meta_info(self, outputs: list[DataProto], base_meta: dict[str, Any]) -> dict[str, Any]:
        """Merge meta info dictionaries, preserving timing data when possible."""
        meta = deepcopy(base_meta) if base_meta is not None else {}
        total_timing: dict[str, float] = {}
        for out in outputs:
            if "timing" not in out.meta_info:
                continue
            for key, value in out.meta_info["timing"].items():
                if isinstance(value, (int, float)):
                    total_timing[key] = total_timing.get(key, 0.0) + float(value)
        if total_timing:
            meta.setdefault("timing", {})
            for key, value in total_timing.items():
                meta["timing"][f"{key}_sum"] = meta["timing"].get(f"{key}_sum", 0.0) + value
        meta.setdefault("oom_guard", {})
        meta["oom_guard"]["enabled"] = True
        meta["oom_guard"]["total_retries"] = self.stats.total_retries
        meta["oom_guard"]["total_splits"] = self.stats.total_splits
        meta["oom_guard"]["aborted_samples"] = self.stats.aborted_samples
        return meta

    # ------------------------------------------------------------------ #
    # Default aborted result helper
    # ------------------------------------------------------------------ #
    def _build_aborted_dataprotos(self, batch: DataProto, exc: BaseException) -> DataProto:
        """Create a placeholder DataProto for samples that cannot be recovered."""
        batch_size = len(batch)
        tensors: dict[str, torch.Tensor] = {}

        prompt_ids = (
            batch.batch["input_ids"] if batch.batch is not None and "input_ids" in batch.batch.keys() else None
        )
        attention_mask = (
            batch.batch["attention_mask"]
            if batch.batch is not None and "attention_mask" in batch.batch.keys()
            else None
        )
        position_ids = (
            batch.batch["position_ids"] if batch.batch is not None and "position_ids" in batch.batch.keys() else None
        )

        if prompt_ids is None:
            prompt_ids = torch.zeros((batch_size, 0), dtype=torch.long)
        tensors["input_ids"] = prompt_ids.clone().to("cpu")

        vocab_dtype = tensors["input_ids"].dtype if tensors["input_ids"].numel() > 0 else torch.long
        responses = torch.empty((batch_size, 0), dtype=vocab_dtype)
        tensors["responses"] = responses
        tensors["sequences"] = tensors["input_ids"].clone()
        tensors["old_log_probs"] = torch.empty((batch_size, 0), dtype=torch.float32)

        if attention_mask is not None:
            tensors["attention_mask"] = attention_mask.clone().to("cpu")
        else:
            tensors["attention_mask"] = torch.ones_like(tensors["input_ids"], dtype=torch.long)

        if position_ids is not None:
            tensors["position_ids"] = position_ids.clone().to("cpu")
        else:
            if tensors["input_ids"].numel() == 0:
                tensors["position_ids"] = torch.empty_like(tensors["input_ids"])
            else:
                seq_len = tensors["input_ids"].shape[1]
                tensors["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        meta_info = deepcopy(batch.meta_info) if batch.meta_info is not None else {}
        meta_info.setdefault("oom_guard", {})
        meta_info["oom_guard"]["aborted"] = True
        meta_info["oom_guard"]["error"] = _format_exception(exc)

        non_tensor_batch = deepcopy(batch.non_tensor_batch) if batch.non_tensor_batch is not None else {}
        oom_reason = np.array([_format_exception(exc)] * batch_size, dtype=object)
        non_tensor_batch["oom_guard/error"] = oom_reason
        non_tensor_batch["oom_guard/aborted"] = np.ones(batch_size, dtype=bool)

        placeholder = DataProto.from_dict(tensors=tensors, non_tensors=non_tensor_batch, meta_info=meta_info)
        placeholder.meta_info.setdefault("timing", {})["generate_sequences"] = 0.0
        return placeholder


def wrap_generate_sequences_with_oom_guard(
    workgroup: Any,
    *,
    max_retries: int = 1,
    min_chunk_size: int = 1,
    on_aborted: Optional[Callable[[DataProto, BaseException], DataProto]] = None,
    log: Optional[logging.Logger] = None,
) -> WorkgroupOOMGuard:
    """Convenience helper to install an OOM guard on ``generate_sequences``."""
    attr_name = "_oom_guard_installed"
    if getattr(workgroup, attr_name, None):
        return getattr(workgroup, attr_name)

    guard = WorkgroupOOMGuard(
        workgroup,
        method_name="generate_sequences",
        max_retries=max_retries,
        min_chunk_size=min_chunk_size,
        on_aborted=on_aborted,
        log=log,
    )
    setattr(workgroup, attr_name, guard)
    return guard
