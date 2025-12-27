import logging
import os
import time
import typing as tp

import numpy as np
import pandas as pd

_DATASET_ENV = "RECTOOLS_LOG_DATASET_NAME"
_LOGGER = logging.getLogger(__name__)


def get_dataset_name(default: str = "unknown") -> str:
    return (os.environ.get(_DATASET_ENV) or default).strip()


RewardFn = tp.Callable[[pd.DataFrame, tp.Mapping[str, tp.Any]], pd.Series]


_DEFAULT_DATASET_REWARD_FN: list[tuple[str, str]] = [
    ("ml_", "rating_threshold"),
    # ("s3_", "rating_threshold"), # TODO
    # ("beeradvocate_r", "rating_threshold"), # TODO
]


def _default_reward_fn_for_dataset(dataset_name: str) -> str:
    name = (dataset_name or "").strip().lower()
    for prefix, fn in _DEFAULT_DATASET_REWARD_FN:
        if name.startswith(prefix):
            return fn
    raise KeyError(
        f"Default reward_fn is not defined for dataset '{dataset_name}'. "
        "Pass data_preparator_kwargs.reward_fn explicitly."
    )


def _reward_from_rating_threshold(
    interactions_df: pd.DataFrame, kwargs: tp.Mapping[str, tp.Any]
) -> pd.Series:
    threshold = float(kwargs.get("reward_threshold", 3.5))
    rating_col = str(kwargs.get("reward_rating_col", "rating"))
    if rating_col not in interactions_df.columns:
        raise KeyError(f"Rating column '{rating_col}' not found in interactions_df")
    return (interactions_df[rating_col].astype(float) > threshold).astype(np.float32)


_REWARD_FNS: dict[str, RewardFn] = {
    "rating_threshold": _reward_from_rating_threshold,
}


def compute_reward(
    interactions_df: pd.DataFrame,
    dataset_name: tp.Optional[str] = None,
    reward_fn: tp.Optional[str] = None,
    **kwargs: tp.Any,
) -> pd.Series:
    ds = dataset_name or get_dataset_name()
    fn_name = (reward_fn or _default_reward_fn_for_dataset(ds)).strip()
    fn = _REWARD_FNS.get(fn_name)
    if fn is None:
        raise KeyError(f"Unknown reward_fn '{fn_name}'. Known: {sorted(_REWARD_FNS)}")
    t0 = time.perf_counter()
    reward = fn(interactions_df, kwargs)
    dt_s = time.perf_counter() - t0
    _LOGGER.info(
        "reward.compute dataset=%s reward_fn=%s n_rows=%d time_ms=%.2f",
        ds,
        fn_name,
        len(interactions_df),
        dt_s * 1000.0,
    )
    if not isinstance(reward, pd.Series):
        reward = pd.Series(reward, index=interactions_df.index)
    reward = reward.astype(np.float32)
    reward.index = interactions_df.index
    return reward


