import typing as tp

import numpy as np
import torch
from rectools.models.nn.transformers.constants import MASKING_VALUE
from torch.utils.data import DataLoader

from src.models.rl.dataset import RewardSequenceDataset
from src.models.rl.reward_registry import compute_reward, get_dataset_name
from src.models.transformers.objective.all_action import AllActionDataPreparator


class RewardedAllActionDataPreparator(AllActionDataPreparator):
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
        sequence_dataset = RewardSequenceDataset.from_interactions(interactions, reward=reward)
        return DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
        )

    @staticmethod
    def _max_target_length_from_weights(batch: tp.Iterable[tp.List[float]]) -> int:
        max_target_length = 0
        for ses_weights in batch:
            target_length = len([weight for weight in ses_weights if weight != 0])
            max_target_length = max(max_target_length, target_length)
        return max_target_length

    def _collate_fn_train(  # type: ignore[override]
        self,
        batch: tp.List[tp.Tuple[tp.List[int], tp.List[float], tp.List[float]]],
    ) -> tp.Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_target_length = self._max_target_length_from_weights([w for _, w, _ in batch])

        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, max_target_length))
        yw = np.zeros((batch_size, max_target_length))
        reward = np.zeros((batch_size, self.session_max_len), dtype=np.float32)

        mask_id = int(self.extra_token_ids[MASKING_VALUE])

        for i, (ses, ses_weights, ses_rewards) in enumerate(batch):
            train_indices = [idx for idx, weight in enumerate(ses_weights) if weight == 0]
            train_session = [ses[idx] for idx in train_indices] + [mask_id]
            train_rewards = [float(ses_rewards[idx]) for idx in train_indices] + [0.0]

            target_indices = [idx for idx, weight in enumerate(ses_weights) if weight != 0]
            if len(target_indices) > 0:
                y_slice_start = target_indices[0]
                y[i, -len(target_indices) :] = ses[y_slice_start:]
                yw[i, -len(target_indices) :] = ses_weights[y_slice_start:]

            train_session = train_session[-self.session_max_len :]
            train_rewards = train_rewards[-self.session_max_len :]
            x[i, -len(train_session) :] = train_session
            reward[i, -len(train_rewards) :] = np.asarray(train_rewards, dtype=np.float32)

        batch_dict: tp.Dict[str, torch.Tensor] = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
            "max_target_length": torch.tensor(max_target_length),
        }
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict,
                lowest_id=self.n_item_extra_tokens,
                highest_id=self.item_id_map.size,
                session_len_limit=max_target_length,
            )

        reward_t = torch.FloatTensor(reward)
        reward_t = reward_t.masked_fill((batch_dict["x"] == 0) | (batch_dict["x"] == mask_id), 0.0)
        batch_dict["reward"] = reward_t
        return batch_dict


