"""Helper functions for handling OOM situations in PPO trainer workflows."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.workgroup_oom_guard import WorkgroupOOMGuard

if TYPE_CHECKING:
    from .ray_trainer import RayPPOTrainer


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
        path = "actor_rollout_ref.actor.ppo_mini_batch_size"
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

            micro_path = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
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
