import torch

from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.oom_handlers import build_actor_update_oom_handler
from verl.utils.workgroup_oom_guard import WorkgroupOOMGuard


class DummyTrainer:
    def __init__(self, config):
        self.config = config
        self._updates = []

    def update_config(self, parameter_path: str, value):
        from verl.utils.config import update_config_value

        self._updates.append((parameter_path, value))
        update_config_value(self.config, parameter_path, value)


class DummyWorkgroup:
    def update_actor(self, batch):  # pragma: no cover - not executed directly
        raise torch.cuda.OutOfMemoryError("OOM")


def _build_trainer():
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 64,
                    "ppo_micro_batch_size_per_gpu": 8,
                }
            }
        }
    )
    return DummyTrainer(cfg)


def test_build_actor_update_oom_handler_reduces_batch_size():
    trainer = _build_trainer()
    workgroup = DummyWorkgroup()
    guard = WorkgroupOOMGuard(workgroup, method_name="update_actor")

    handler = build_actor_update_oom_handler(trainer, guard)

    batch = DataProto.from_dict({"input_ids": torch.zeros((2, 1), dtype=torch.long)})
    result = handler(batch, torch.cuda.OutOfMemoryError("OOM"), attempt=0, max_retries=3, guard=guard)

    assert result is None
    assert OmegaConf.select(trainer.config, "actor_rollout_ref.actor.ppo_mini_batch_size") == 32
    assert ("actor_rollout_ref.actor.ppo_mini_batch_size", 32) in trainer._updates


def test_build_actor_update_oom_handler_handles_minimum_batch():
    trainer = _build_trainer()
    trainer.update_config("actor_rollout_ref.actor.ppo_mini_batch_size", 1)

    workgroup = DummyWorkgroup()
    guard = WorkgroupOOMGuard(workgroup, method_name="update_actor")

    handler = build_actor_update_oom_handler(trainer, guard)

    batch = DataProto.from_dict({"input_ids": torch.zeros((1, 1), dtype=torch.long)})
    result = handler(batch, torch.cuda.OutOfMemoryError("OOM"), attempt=0, max_retries=3, guard=guard)

    assert OmegaConf.select(trainer.config, "actor_rollout_ref.actor.ppo_mini_batch_size") == 1
    assert result.meta_info["oom_guard"]["aborted"] is True
