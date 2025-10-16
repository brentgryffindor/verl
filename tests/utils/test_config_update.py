import pytest
from omegaconf import OmegaConf

from verl.utils.config import list_config_paths, update_config_value
from verl.workers.config.actor import ActorConfig


def test_update_dictconfig_value_with_struct():
    cfg = OmegaConf.create({"actor": {"ppo_mini_batch_size": 256}})
    OmegaConf.set_struct(cfg, True)

    update_config_value(cfg, "actor.ppo_mini_batch_size", 128)

    assert cfg.actor.ppo_mini_batch_size == 128


def test_update_dataclass_value():
    actor_cfg = ActorConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=1)

    update_config_value(actor_cfg, "ppo_mini_batch_size", 64)

    assert actor_cfg.ppo_mini_batch_size == 64


def test_update_missing_path_raises():
    cfg = OmegaConf.create({"actor": {"ppo_mini_batch_size": 256}})

    with pytest.raises(KeyError):
        update_config_value(cfg, "actor.invalid_param", 1)


def test_list_config_paths_dictconfig():
    cfg = OmegaConf.create({"actor": {"ppo_mini_batch_size": 256, "optim": {"lr": 1e-6}}})
    paths = list_config_paths(cfg)
    assert set(paths) == {"actor.ppo_mini_batch_size", "actor.optim.lr"}


def test_list_config_paths_dataclass():
    actor_cfg = ActorConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=1)
    paths = list_config_paths(actor_cfg)
    assert "ppo_mini_batch_size" in paths
    assert "ppo_micro_batch_size_per_gpu" in paths
