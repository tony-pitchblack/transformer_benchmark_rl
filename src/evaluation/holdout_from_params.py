import argparse
import logging
import os
import typing as tp
from pathlib import Path

import pandas as pd

from src.evaluation.common import (
    get_report_path,
    get_val_scheme_arguments,
    iterate_model_params,
    validate_model_on_holdout,
)
from src.utils import console_logging, read_config, setup_deterministic


def _resolve_ckpt_from_report_csv(
    report_csv: str,
    dataset_name: str,
    comment: str,
    cls: str,
) -> tp.Optional[str]:
    report = pd.read_csv(report_csv)
    if "ckpt" not in report.columns:
        return None
    m = (report["dataset_name"] == dataset_name) & (report["comment"] == comment) & (report["cls"] == cls)
    rows = report[m]
    if len(rows) == 0:
        return None
    if len(rows) > 1:
        logging.warning(
            "Multiple checkpoint matches for dataset_name=%r comment=%r cls=%r in %r; using the last row (%d matches)",
            dataset_name,
            comment,
            cls,
            report_csv,
            len(rows),
        )
    ckpt = rows.iloc[-1]["ckpt"]
    if pd.isna(ckpt):
        return None
    return str(ckpt)


def _default_pretrained_report_csv(
    pretrained_spec: tp.Mapping[str, tp.Any],
    val_scheme: str,
    dataset_name: str,
) -> str:
    report_file_path = get_report_path(val_scheme, dataset_name, "holdout")
    return os.path.join(report_file_path, str(pretrained_spec["report_file_name"]) + ".csv")


def validate_models_on_holdout(
    config_file: str = "configs/holdout/current_params.yaml",
):
    """Validate models on holdout set for all parameter combinations.

    Parameters
    ----------
    config_file : str
        Path to config file with model parameters
    """
    config = read_config(config_file)
    from_pretrained = bool(config.get("from_pretrained", False))

    for val_scheme in config["val_schemes"]:
        # Get metrics and k from validation scheme
        scheme_args = get_val_scheme_arguments(val_scheme, ["METRICS", "K"])
        metrics, k = scheme_args["METRICS"], scheme_args["K"]

        for dataset_name in config["datasets"]:
            for model_search_spec in config["models"]:
                model_cls = model_search_spec["cls"]
                logging.info(
                    f"Validate {model_cls} on {dataset_name} using {val_scheme}"
                )
                os.environ["RECTOOLS_LOG_MODEL_CLS"] = str(model_cls)

                # Get report path
                report_file_path = get_report_path(val_scheme, dataset_name, "holdout")
                report_file = os.path.join(
                    report_file_path, model_search_spec["report_file_name"] + ".csv"
                )

                if not from_pretrained:
                    for model_params, search_name in iterate_model_params(model_search_spec):
                        if len(search_name) > 0:
                            current_name = model_search_spec["comment"] + "_" + search_name
                        else:
                            current_name = model_search_spec["comment"]
                        os.environ["RECTOOLS_LOG_COMMENT"] = str(model_search_spec["comment"]).strip()
                        validate_model_on_holdout(
                            dataset_name=dataset_name,
                            model_params=model_params,
                            val_scheme=val_scheme,
                            metrics=metrics,
                            k=k,
                            report_file=report_file,
                            current_name=current_name,
                        )
                    continue

                pretrained_config_file = config.get("pretrained_config_file")
                pretrained_report_csv = config.get("pretrained_report_csv")
                if not pretrained_config_file:
                    raise ValueError("from_pretrained=true requires pretrained_config_file")

                pretrained_cfg = read_config(pretrained_config_file)
                for pretrained_spec in pretrained_cfg["models"]:
                    report_csv = str(pretrained_report_csv) if pretrained_report_csv else _default_pretrained_report_csv(
                        pretrained_spec, val_scheme=val_scheme, dataset_name=dataset_name
                    )
                    for pretrained_params, pretrained_search_name in iterate_model_params(pretrained_spec):
                        if len(pretrained_search_name) > 0:
                            pretrained_comment = pretrained_spec["comment"] + "_" + pretrained_search_name
                        else:
                            pretrained_comment = pretrained_spec["comment"]
                        ckpt = _resolve_ckpt_from_report_csv(
                            report_csv,
                            dataset_name=dataset_name,
                            comment=pretrained_comment,
                            cls=str(pretrained_params["cls"]),
                        )
                        if ckpt is None:
                            continue

                        for crr_params, crr_search_name in iterate_model_params(model_search_spec):
                            crr_params["encoder_ckpt_path"] = ckpt
                            crr_params["encoder_model_params"] = pretrained_params

                            if len(crr_search_name) > 0:
                                run_name = f"{pretrained_comment}__{model_search_spec['comment']}_{crr_search_name}"
                            else:
                                run_name = f"{pretrained_comment}__{model_search_spec['comment']}"
                            os.environ["RECTOOLS_LOG_COMMENT"] = str(model_search_spec["comment"]).strip()

                            validate_model_on_holdout(
                                dataset_name=dataset_name,
                                model_params=crr_params,
                                val_scheme=val_scheme,
                                metrics=metrics,
                                k=k,
                                report_file=report_file,
                                current_name=run_name,
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file",
        default="configs/holdout/current_params.yaml",
    )
    parser.add_argument(
        "--log_backend",
        type=str,
        choices=("mlflow", "csv", "both"),
        default="mlflow",
    )
    args = parser.parse_args()

    os.environ["RECTOOLS_LOG_BACKEND"] = args.log_backend
    console_logging(level=logging.INFO)
    setup_deterministic()
    validate_models_on_holdout(config_file=args.config_file)
