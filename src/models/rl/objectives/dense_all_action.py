import typing as tp
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from rectools import Columns
from torch.utils.data import DataLoader

from src.models.rl.reward_registry import compute_reward, get_dataset_name
from src.models.transformers.objective.dense_all_action import DenseAllActionDataPreparator


class RewardedDenseAllActionSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sessions: tp.List[tp.List[int]],
        weights: tp.List[tp.List[float]],
        rewards: tp.List[tp.List[float]],
        possible_targets_idx: tp.List[tp.List[tp.List[int]]],
    ):
        self.sessions = sessions
        self.weights = weights
        self.rewards = rewards
        self.possible_targets_idx = possible_targets_idx

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(
        self, index: int
    ) -> tp.Tuple[tp.List[int], tp.List[float], tp.List[float], tp.List[tp.List[int]]]:
        return (
            self.sessions[index],
            self.weights[index],
            self.rewards[index],
            self.possible_targets_idx[index],
        )

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        reward: pd.Series,
        targets_window_days: int,
        sort_users: bool = False,
    ) -> "RewardedDenseAllActionSequenceDataset":
        tmp = interactions[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]].copy()
        tmp["_reward"] = reward.reindex(interactions.index).astype(float).values
        sessions = (
            tmp.sort_values(Columns.Datetime, kind="stable")
            .groupby(Columns.User, sort=sort_users)[[Columns.Item, Columns.Weight, Columns.Datetime, "_reward"]]
            .agg(list)
        )
        items = sessions[Columns.Item].to_list()
        weights = sessions[Columns.Weight].to_list()
        datetimes = sessions[Columns.Datetime].to_list()
        rewards = sessions["_reward"].to_list()

        possible_targets_idx: tp.List[tp.List[tp.List[int]]] = []
        for ses_idx, ses in enumerate(items):
            ses_datetimes = datetimes[ses_idx]
            idx_ses_datetimes_mapping = list(enumerate(ses_datetimes))
            ses_targets_idx: tp.List[tp.List[int]] = []
            for idx, min_y_dt in enumerate(ses_datetimes):
                max_y_dt = min_y_dt + timedelta(days=targets_window_days)
                targets_idx = list(
                    map(
                        lambda x: x[0],
                        filter(
                            lambda x: x[1] > min_y_dt and x[1] <= max_y_dt,
                            idx_ses_datetimes_mapping,
                        ),
                    )
                )
                if len(targets_idx) < 1 and idx + 1 < len(ses):
                    targets_idx = [idx + 1]
                ses_targets_idx.append(targets_idx)
            possible_targets_idx.append(ses_targets_idx)

        return cls(
            sessions=items,
            weights=weights,
            rewards=rewards,
            possible_targets_idx=possible_targets_idx,
        )


class RewardedDenseAllActionDataPreparator(DenseAllActionDataPreparator):
    def __init__(
        self,
        *args: tp.Any,
        dataset_name: tp.Optional[str] = None,
        reward_fn: tp.Optional[str] = None,
        reward_threshold: float = 3.5,
        reward_rating_col: str = "rating",
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reward_dataset_name = dataset_name
        self._reward_fn = reward_fn
        self._reward_threshold = reward_threshold
        self._reward_rating_col = reward_rating_col

    def get_dataloader_train(self) -> DataLoader:
        interactions = self.train_dataset.interactions.df
        dataset_name = self._reward_dataset_name or get_dataset_name()
        reward = compute_reward(
            interactions,
            dataset_name=dataset_name,
            reward_fn=self._reward_fn,
            reward_threshold=self._reward_threshold,
            reward_rating_col=self._reward_rating_col,
        )
        sequence_dataset = RewardedDenseAllActionSequenceDataset.from_interactions(
            interactions, reward=reward, targets_window_days=self.targets_window_days
        )
        return DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
        )

    def _collate_fn_train(  # type: ignore[override]
        self,
        batch: tp.List[tp.Tuple[tp.List[int], tp.List[float], tp.List[float], tp.List[tp.List[int]]]],
    ) -> tp.Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        reward = np.zeros((batch_size, self.session_max_len), dtype=np.float32)
        for i, (ses, ses_weights, ses_rewards, ses_possible_target_indices) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]
            reward[i, -len(ses) + 1 :] = np.asarray(ses_rewards[:-1], dtype=np.float32)

            start_y_index = -len(ses) + 1
            for position, possible_target_indices in enumerate(ses_possible_target_indices):
                if len(possible_target_indices) == 0:
                    continue
                target_index = np.random.choice(possible_target_indices, size=1)[0]
                y[i, start_y_index + position] = ses[target_index]
                yw[i, start_y_index + position] = ses_weights[target_index]

        batch_dict: tp.Dict[str, torch.Tensor] = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
            "reward": torch.FloatTensor(reward),
        }
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict,
                lowest_id=self.n_item_extra_tokens,
                highest_id=self.item_id_map.size,
            )
        return batch_dict


