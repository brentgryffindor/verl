# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import fields, is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config", "update_config_value", "list_config_paths"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def list_config_paths(config: Any, prefix: str = "") -> list[str]:
    """Return a list of dotted paths for all configurable leaves in ``config``."""

    if not _is_container(config):
        return [prefix] if prefix else []

    paths: list[str] = []
    for key, value in _iter_config_items(config):
        if key is None:
            continue
        key_str = str(key)
        path = f"{prefix}.{key_str}" if prefix else key_str

        if _is_container(value):
            child_paths = list_config_paths(value, path)
            if child_paths:
                paths.extend(child_paths)
            else:
                paths.append(path)
        else:
            paths.append(path)

    return paths


def update_config_value(config: Any, parameter_path: str, value: Any) -> None:
    """Update a configuration value using a dotted parameter path.

    Supports both OmegaConf ``DictConfig`` objects and dataclass based configs
    (including subclasses of ``BaseConfig``). The function walks the dotted
    path (e.g. ``"actor_rollout_ref.actor.ppo_mini_batch_size"``) and updates
    the referenced leaf value in-place.

    Args:
        config: The configuration object to modify.
        parameter_path: Dotted path locating the parameter to update.
        value: The value to assign to the parameter.

    Raises:
        KeyError: If the parameter path does not exist.
        TypeError: If an intermediate object in the path is not indexable.
        omegaconf.errors.ReadonlyConfigError: If OmegaConf struct mode forbids the update.
        dataclasses.FrozenInstanceError: If the dataclass field is immutable.
    """

    if isinstance(config, DictConfig | ListConfig):
        _update_dictconfig_value(config, parameter_path, value)
        return

    if is_dataclass(config):
        _update_dataclass_value(config, parameter_path, value)
        return

    raise TypeError(f"Unsupported config type: {type(config)}")


def _update_dictconfig_value(config: DictConfig | ListConfig, parameter_path: str, value: Any) -> None:
    """Internal helper to update DictConfig/ListConfig values."""
    struct_flag = OmegaConf.is_struct(config)
    try:
        if struct_flag:
            OmegaConf.set_struct(config, False)

        # Validate that the path exists before attempting to update.
        OmegaConf.select(config, parameter_path, throw_on_missing=True)
        OmegaConf.update(config, parameter_path, value, merge=False)
    finally:
        if struct_flag:
            OmegaConf.set_struct(config, True)


def _update_dataclass_value(config: Any, parameter_path: str, value: Any) -> None:
    """Internal helper to update dataclass-based configs."""
    parent, attr = _resolve_attr_path(config, parameter_path)

    if isinstance(parent, dict):
        if attr not in parent:
            raise KeyError(f"Key '{attr}' not found while updating '{parameter_path}'")
        parent[attr] = value
        return

    if hasattr(parent, attr):
        setattr(parent, attr, value)
        return

    raise KeyError(f"Attribute '{attr}' not found while updating '{parameter_path}'")


def _resolve_attr_path(config: Any, parameter_path: str) -> tuple[Any, str]:
    """Resolve the parent object and leaf attribute for a dotted parameter path."""
    if not parameter_path:
        raise ValueError("parameter_path must be a non-empty string")

    segments = parameter_path.split(".")
    parent = config

    for segment in segments[:-1]:
        parent = _navigate_to_child(parent, segment, parameter_path)

    return parent, segments[-1]


def _navigate_to_child(current: Any, segment: str, full_path: str) -> Any:
    """Navigate to the next node specified by 'segment' from 'current'."""
    if isinstance(current, DictConfig):
        try:
            return current[segment]
        except Exception as exc:  # pragma: no cover - OmegaConf raises specialised errors
            raise KeyError(f"Failed to access '{segment}' while updating '{full_path}'") from exc

    if isinstance(current, ListConfig):
        try:
            idx = int(segment)
        except ValueError as exc:
            raise TypeError(f"List segment '{segment}' must be an integer while updating '{full_path}'") from exc
        try:
            return current[idx]
        except Exception as exc:  # pragma: no cover
            raise KeyError(f"Index '{segment}' out of range while updating '{full_path}'") from exc

    if isinstance(current, dict):
        if segment not in current:
            raise KeyError(f"Key '{segment}' not found while updating '{full_path}'")
        return current[segment]

    if isinstance(current, list):
        try:
            idx = int(segment)
        except ValueError as exc:
            raise TypeError(f"List segment '{segment}' must be an integer while updating '{full_path}'") from exc
        try:
            return current[idx]
        except Exception as exc:  # pragma: no cover
            raise KeyError(f"Index '{segment}' out of range while updating '{full_path}'") from exc

    if hasattr(current, segment):
        return getattr(current, segment)

    raise KeyError(f"Attribute '{segment}' not found while updating '{full_path}'")


def _is_container(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (DictConfig, dict, ListConfig, list)):
        return not isinstance(value, (str, bytes))
    if is_dataclass(value):
        return True
    return False


def _iter_config_items(config: Any):
    if isinstance(config, DictConfig):
        return config.items()
    if isinstance(config, dict):
        return config.items()
    if isinstance(config, ListConfig):
        return enumerate(config)
    if isinstance(config, list):
        return enumerate(config)
    if is_dataclass(config):
        return ((field.name, getattr(config, field.name)) for field in fields(config) if not field.name.startswith("_"))
    return ()


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    print("[validate_config] All configuration checks passed successfully!")
