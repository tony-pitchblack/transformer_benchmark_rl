import typing as tp

import pandas as pd
import torch
import typing_extensions as tpe
from pydantic import BeforeValidator, PlainSerializer
from pytorch_lightning import Trainer
from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ErrorBehaviour, ExternalIds, ModelBase, ModelConfig
from rectools.models.nn.transformers.base import _get_class_obj
from rectools.models.nn.transformers.lightning import TransformerLightningModule
from rectools.utils.misc import get_class_or_function_full_path

from src.models.rl.crr_lightning import CRRLightningModule
from src.models.rl.data_preparators import RewardedSASRecDataPreparator
from src.utils import read_config


TrainerFuncType = tpe.Annotated[
    tp.Callable[[], Trainer],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]


class SASRecCRRConfig(ModelConfig):
    crr_config_file: tp.Optional[str] = None
    crr_config: tp.Optional[tp.Dict[str, tp.Any]] = None
    encoder_ckpt_path: str
    encoder_model_params: tp.Optional[tp.Dict[str, tp.Any]] = None
    get_trainer_func: tp.Optional[TrainerFuncType] = None


def _ensure_trainer(get_trainer_func: tp.Optional[tp.Callable[[], Trainer]]) -> Trainer:
    if get_trainer_func is None:
        return Trainer(max_epochs=1, enable_checkpointing=False, logger=False, enable_progress_bar=False)
    return get_trainer_func()


class SASRecCRR(ModelBase[SASRecCRRConfig]):
    config_class = SASRecCRRConfig

    def __init__(
        self,
        crr_config_file: tp.Optional[str],
        crr_config: tp.Optional[tp.Dict[str, tp.Any]],
        encoder_ckpt_path: str,
        encoder_model_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
        get_trainer_func: tp.Optional[tp.Callable[[], Trainer]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.crr_config_file = crr_config_file
        self.crr_config = crr_config
        self.encoder_ckpt_path = encoder_ckpt_path
        self.encoder_model_params = encoder_model_params or None
        self.get_trainer_func = get_trainer_func

        self.encoder_pl: TransformerLightningModule
        self.crr_pl: CRRLightningModule
        self.fit_trainer: Trainer

        self.item_id_map: tp.Optional[IdMap] = None
        self.user_id_map: tp.Optional[IdMap] = None
        self.session_max_len: tp.Optional[int] = None
        self.n_item_extra_tokens: tp.Optional[int] = None

    def _get_config(self) -> SASRecCRRConfig:
        return SASRecCRRConfig(
            cls=self.__class__,
            crr_config_file=self.crr_config_file,
            crr_config=self.crr_config,
            encoder_ckpt_path=self.encoder_ckpt_path,
            encoder_model_params=self.encoder_model_params,
            get_trainer_func=self.get_trainer_func,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: SASRecCRRConfig) -> tpe.Self:
        return cls(
            crr_config_file=config.crr_config_file,
            crr_config=config.crr_config,
            encoder_ckpt_path=config.encoder_ckpt_path,
            encoder_model_params=config.encoder_model_params,
            get_trainer_func=config.get_trainer_func,
            verbose=config.verbose,
        )

    @staticmethod
    def _build_preparator(
        dataset: Dataset,
        encoder_model_params: tp.Optional[tp.Dict[str, tp.Any]],
        crr_cfg: tp.Dict[str, tp.Any],
    ) -> RewardedSASRecDataPreparator:
        dp_params: tp.Dict[str, tp.Any] = {}
        if encoder_model_params:
            for key in (
                "session_max_len",
                "batch_size",
                "dataloader_num_workers",
                "train_min_user_interactions",
                "shuffle_train",
                "get_val_mask_func",
            ):
                if key in encoder_model_params:
                    dp_params[key] = encoder_model_params[key]
        dp_params.update(crr_cfg.get("data_preparator", {}))

        if "session_max_len" not in dp_params:
            dp_params["session_max_len"] = int(crr_cfg.get("session_max_len", 50))
        if "batch_size" not in dp_params:
            dp_params["batch_size"] = int(crr_cfg.get("batch_size", 256))
        if "dataloader_num_workers" not in dp_params:
            dp_params["dataloader_num_workers"] = int(crr_cfg.get("dataloader_num_workers", 0))
        if "train_min_user_interactions" not in dp_params:
            dp_params["train_min_user_interactions"] = int(crr_cfg.get("train_min_user_interactions", 2))
        dp_params.setdefault("shuffle_train", True)

        dp = RewardedSASRecDataPreparator(
            n_negatives=None,
            negative_sampler=None,
            dataset_name=crr_cfg.get("dataset_name"),
            reward_fn=crr_cfg.get("reward_fn"),
            reward_threshold=float(crr_cfg.get("reward_threshold", 3.5)),
            reward_rating_col=str(crr_cfg.get("reward_rating_col", "rating")),
            **dp_params,
        )
        dp.process_dataset_train(dataset)
        return dp

    def _fit(self, dataset: Dataset) -> None:
        if self.crr_config is not None:
            crr_cfg = dict(self.crr_config)
        elif self.crr_config_file is not None:
            crr_cfg = read_config(self.crr_config_file)
        else:
            raise ValueError("Expected crr_config or crr_config_file")

        self.encoder_pl = TransformerLightningModule.load_from_checkpoint(self.encoder_ckpt_path)
        self.encoder_pl.eval()
        for p in self.encoder_pl.parameters():
            p.requires_grad = False

        dp = self._build_preparator(dataset, self.encoder_model_params, crr_cfg)
        train_loader = dp.get_dataloader_train()
        val_loader = None
        if getattr(dp, "val_interactions", None) is not None:
            try:
                val_loader = dp.get_dataloader_val()
            except Exception:
                val_loader = None

        self.item_id_map = dp.item_id_map
        self.user_id_map = dp.train_dataset.user_id_map
        self.session_max_len = int(dp.session_max_len)
        self.n_item_extra_tokens = int(dp.n_item_extra_tokens)

        self.crr_pl = CRRLightningModule(
            encoder_torch_model=self.encoder_pl.torch_model,
            actor_hidden_dims=crr_cfg.get("actor_hidden_dims", [256, 256]),
            critic_hidden_dims=crr_cfg.get("critic_hidden_dims", [256, 256]),
            dropout=float(crr_cfg.get("dropout", 0.0)),
            n_negatives=int(crr_cfg.get("n_negatives", 256)),
            gamma=float(crr_cfg.get("gamma", 0.99)),
            beta=float(crr_cfg.get("beta", 1.0)),
            tau=float(crr_cfg.get("tau", 0.005)),
            m_adv=int(crr_cfg.get("m_adv", 4)),
            m_td=int(crr_cfg.get("m_td", 4)),
            lr_actor=float(crr_cfg.get("lr_actor", 1e-4)),
            lr_critic=float(crr_cfg.get("lr_critic", 1e-4)),
            weight_decay=float(crr_cfg.get("weight_decay", 0.0)),
        )

        self.fit_trainer = _ensure_trainer(self.get_trainer_func)
        self.fit_trainer.fit(self.crr_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def recommend(
        self,
        users: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        if self.item_id_map is None or self.user_id_map is None:
            raise RuntimeError("Model is not fitted.")
        if self.session_max_len is None or self.n_item_extra_tokens is None:
            raise RuntimeError("Model is not fitted.")

        users_arr = pd.Index(users)
        internal_users = self.user_id_map.convert_to_internal(users_arr.to_numpy(), on_unsupported_targets)
        internal_users = pd.Index(internal_users)
        mask_supported = internal_users >= 0
        users_supported = users_arr[mask_supported]
        internal_users = internal_users[mask_supported]

        interactions = dataset.interactions.df
        if Columns.Datetime in interactions.columns:
            interactions = interactions.sort_values(Columns.Datetime, kind="stable")
        interactions = interactions[interactions[Columns.User].isin(users_supported)]
        interactions[Columns.Item] = self.item_id_map.convert_to_internal(
            interactions[Columns.Item].to_numpy(),
            on_unsupported_targets="warn",
        )
        interactions = interactions[interactions[Columns.Item] >= self.n_item_extra_tokens]

        grouped = interactions.groupby(Columns.User, sort=False)[Columns.Item].apply(list)
        x = torch.zeros((len(users_supported), self.session_max_len), dtype=torch.long)
        for i, u in enumerate(users_supported):
            seq = grouped.get(u, [])
            if not seq:
                continue
            seq = seq[-self.session_max_len :]
            x[i, -len(seq) :] = torch.tensor(seq, dtype=torch.long)

        item_embs = self.encoder_pl.torch_model.item_model.get_all_embeddings().detach()
        batch = {"x": x.to(self.crr_pl.device)}
        with torch.no_grad():
            state_embs = self.encoder_pl.torch_model.encode_sessions(batch, item_embs.to(self.crr_pl.device))
            lengths = (batch["x"] != 0).sum(dim=1).clamp_min(1) - 1
            last_state = state_embs[torch.arange(state_embs.shape[0]), lengths]
            policy_emb = self.crr_pl.actor(last_state)
            scores = policy_emb @ item_embs.to(policy_emb.device).T
            scores[:, : self.n_item_extra_tokens] = float("-inf")

            if items_to_recommend is not None:
                allowed = self.item_id_map.convert_to_internal(
                    pd.Index(items_to_recommend).to_numpy(), on_unsupported_targets="warn"
                )
                allowed = torch.tensor(allowed, device=scores.device)
                allowed = allowed[allowed >= self.n_item_extra_tokens]
                allowed_mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                allowed_mask[allowed] = False
                scores = scores.masked_fill(allowed_mask.unsqueeze(0), float("-inf"))

            if filter_viewed:
                viewed = batch["x"]
                b = viewed.shape[0]
                row = torch.arange(b, device=scores.device).unsqueeze(1).expand_as(viewed)
                col = viewed.to(scores.device)
                m = col != 0
                scores[row[m], col[m]] = float("-inf")

            _, topk = scores.topk(k=k, dim=1)

        ext_items = self.item_id_map.convert_to_external(topk.detach().cpu().numpy())
        res = pd.DataFrame(
            {
                Columns.User: users_supported.to_numpy().repeat(k),
                Columns.Item: ext_items.reshape(-1),
                Columns.Score: scores.gather(1, topk).detach().cpu().numpy().reshape(-1),
            }
        )
        if add_rank_col:
            res[Columns.Rank] = res.groupby(Columns.User).cumcount() + 1
        return res


