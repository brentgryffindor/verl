import types

import torch

from verl import DataProto
from verl.utils.workgroup_oom_guard import WorkgroupOOMGuard, wrap_method_with_oom_guard


class CallableWithoutName:
    def __call__(self, batch, *_, **__):
        return batch


def test_wrap_method_without_name():
    workgroup = types.SimpleNamespace(generate_sequences=CallableWithoutName())
    guard = WorkgroupOOMGuard(workgroup, max_retries=0)

    batch = DataProto.from_dict({"responses": torch.zeros(1, 1)}, meta_info={})
    result = workgroup.generate_sequences(batch)

    assert result is batch
    guard.unwrap()


class UpdateActorWorkgroup:
    def __init__(self):
        self.calls = 0

    def update_actor(self, _batch):
        self.calls += 1
        raise RuntimeError("CUDA out of memory")


def test_wrap_update_actor_without_splitting():
    workgroup = UpdateActorWorkgroup()
    wrap_method_with_oom_guard(workgroup, "update_actor", max_retries=0)

    batch = DataProto.from_dict({"input_ids": torch.zeros((2, 1), dtype=torch.long)}, meta_info={})
    result = workgroup.update_actor(batch)

    assert workgroup.calls == 1
    assert result.meta_info["oom_guard"]["aborted"] is True
    assert "update_actor" in result.meta_info["timing"]
