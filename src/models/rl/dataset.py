import typing as tp

import pandas as pd
from rectools import Columns
from rectools.models.nn.transformers.data_preparator import SequenceDataset


class RewardSequenceDataset(SequenceDataset):
    def __init__(
        self,
        sessions: tp.List[tp.List[int]],
        weights: tp.List[tp.List[float]],
        rewards: tp.List[tp.List[float]],
    ):
        super().__init__(sessions=sessions, weights=weights)
        self.rewards = rewards

    def __getitem__(self, index: int) -> tp.Tuple[tp.List[int], tp.List[float], tp.List[float]]:
        session, weights = super().__getitem__(index)
        rewards = self.rewards[index]
        return session, weights, rewards

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        reward: pd.Series,
        sort_users: bool = False,
    ) -> "RewardSequenceDataset":
        tmp = interactions[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]].copy()
        tmp["_reward"] = reward.reindex(interactions.index).astype(float).values
        sessions = (
            tmp.sort_values(Columns.Datetime, kind="stable")
            .groupby(Columns.User, sort=sort_users)[[Columns.Item, Columns.Weight, "_reward"]]
            .agg(list)
        )
        return cls(
            sessions=sessions[Columns.Item].to_list(),
            weights=sessions[Columns.Weight].to_list(),
            rewards=sessions["_reward"].to_list(),
        )


