import typing as tp

import numpy as np
import torch
from rectools.models.nn.transformers.constants import MASKING_VALUE
from torch.utils.data import DataLoader

from src.models.rl.dataset import RewardSequenceDataset
from src.models.rl.reward_registry import compute_reward, get_dataset_name
from src.models.transformers.objective.next_action import NextActionDataPreparator


class RewardedNextActionDataPreparator(NextActionDataPreparator):
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

    def _collate_fn_train(  # type: ignore[override]
        self,
        batch: tp.List[tp.Tuple[tp.List[int], tp.List[float], tp.List[float]]],
    ) -> tp.Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, 1))
        yw = np.zeros((batch_size, 1))
        reward = np.zeros((batch_size, self.session_max_len), dtype=np.float32)
        for i, (ses, ses_weights, ses_rewards) in enumerate(batch):
            session = ses.copy()
            session[-1] = self.extra_token_ids[MASKING_VALUE]
            x[i, -len(ses) :] = session
            y[i] = ses[-1]
            yw[i] = ses_weights[-1]
            reward[i, -len(ses) :] = np.asarray(ses_rewards, dtype=np.float32)

        batch_dict: tp.Dict[str, torch.Tensor] = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
        }
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, 1, self.n_negatives),
            )
            batch_dict["negatives"] = negatives

        mask_id = int(self.extra_token_ids[MASKING_VALUE])
        reward_t = torch.FloatTensor(reward)
        reward_t = reward_t.masked_fill((batch_dict["x"] == 0) | (batch_dict["x"] == mask_id), 0.0)
        batch_dict["reward"] = reward_t
        return batch_dict


