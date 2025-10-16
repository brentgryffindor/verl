"""Utilities for wrapping WorkGroup methods with GPU OOM recovery.

This module provides a reusable wrapper that can be installed on any worker group
method (e.g. ``generate_sequences``) to catch out-of-memory errors raised by the
underlying workers. When an OOM happens, the wrapper retries the invocation and,
if it still fails, delegates recovery to a user-provided handler that can apply
custom mitigation or return a placeholder batch so the caller can continue
without crashing.

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
from pprint import pprint

from verl import DataProto

__all__ = ["WorkgroupOOMGuard", "wrap_generate_sequences_with_oom_guard", "wrap_method_with_oom_guard"]

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
    aborted_samples: int = 0


class WorkgroupOOMGuard:
    """Wrap a worker group method to recover from GPU OOM errors.

    The guard retries the method on OOM and, if it still fails, invokes a recovery
    handler so callers can decide how to proceed. It exposes lightweight statistics
    to aid monitoring.
    """

    def __init__(
        self,
        workgroup: Any,
        method_name: str = "generate_sequences",
        *,
        max_retries: int = 5,
        on_aborted: Optional[Callable[[DataProto, BaseException], DataProto]] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.workgroup = workgroup
        self.method_name = method_name
        self._original_method = getattr(workgroup, method_name)
        self.max_retries = max(0, max_retries)
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

        if hasattr(original, "__name__"):
            wrapped.__name__ = original.__name__
        else:
            wrapped.__name__ = f"{self.method_name}_wrapped"
        wrapped.__doc__ = getattr(original, "__doc__", wrapped.__doc__)
        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(self.workgroup, self.method_name, wrapped)

    def _invoke_with_recovery(self, fn, batch: DataProto, *args, **kwargs) -> DataProto:
        try:
            return self._call_with_retries(fn, batch, *args, **kwargs)
        except BaseException as exc:
            if not _is_oom_error(exc):
                raise

            self._logger.warning(
                "[OOMGuard] %s.%s encountered OOM after retries (batch_size=%d). Error: %s",
                type(self.workgroup).__name__,
                self.method_name,
                len(batch),
                _format_exception(exc),
            )
            result = self._call_abort_handler(batch, exc, attempt=self.max_retries, final=True)
            if result is None:
                result = self._build_aborted_dataprotos(batch, exc)

            self.stats.aborted_samples += len(batch)
            return result

    def _call_with_retries(self, fn, batch: DataProto, *args, **kwargs) -> DataProto:
        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                if attempt > 0:
                    pprint(f"[OOMGuard] Retry {attempt}/{self.max_retries} for {type(self.workgroup).__name__}.{self.method_name})")
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
                result = self._call_abort_handler(batch, exc, attempt=attempt, final=False)
                if result is not None:
                    self.stats.aborted_samples += len(batch)
                    return result
        assert last_exc is not None
        raise last_exc

    def _call_abort_handler(
        self,
        batch: DataProto,
        exc: BaseException,
        *,
        attempt: Optional[int],
        final: bool,
    ) -> Optional[DataProto]:
        """Invoke the abort handler with best-effort compatibility."""
        kwargs = {
            "attempt": attempt,
            "max_retries": self.max_retries,
            "guard": self,
            "final": final,
        }
        try:
            return self._on_aborted(batch, exc, **kwargs)
        except TypeError:
            return self._on_aborted(batch, exc)

    # ------------------------------------------------------------------ #
    # Default aborted result helper
    # ------------------------------------------------------------------ #
    def _build_aborted_dataprotos(self, batch: DataProto, exc: BaseException, **_) -> DataProto:
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
        placeholder.meta_info.setdefault("timing", {})[self.method_name] = 0.0
        return placeholder


def wrap_method_with_oom_guard(
    workgroup: Any,
    method_name: str,
    *,
    max_retries: int = 5,
    on_aborted: Optional[Callable[[DataProto, BaseException], DataProto]] = None,
    log: Optional[logging.Logger] = None,
) -> WorkgroupOOMGuard:
    """Install an OOM guard on an arbitrary workgroup method."""
    if not hasattr(workgroup, method_name):
        raise AttributeError(f"{type(workgroup).__name__} has no attribute '{method_name}'")

    attr_name = f"_oom_guard_installed_{method_name}"
    existing = getattr(workgroup, attr_name, None)
    if isinstance(existing, WorkgroupOOMGuard):
        return existing

    guard = WorkgroupOOMGuard(
        workgroup,
        method_name=method_name,
        max_retries=max_retries,
        on_aborted=on_aborted,
        log=log,
    )
    setattr(workgroup, attr_name, guard)
    return guard


def wrap_generate_sequences_with_oom_guard(
    workgroup: Any,
    *,
    max_retries: int = 5,
    on_aborted: Optional[Callable[[DataProto, BaseException], DataProto]] = None,
    log: Optional[logging.Logger] = None,
) -> WorkgroupOOMGuard:
    """Convenience helper to install an OOM guard on ``generate_sequences``."""
    return wrap_method_with_oom_guard(
        workgroup,
        "generate_sequences",
        max_retries=max_retries,
        on_aborted=on_aborted,
        log=log,
    )
