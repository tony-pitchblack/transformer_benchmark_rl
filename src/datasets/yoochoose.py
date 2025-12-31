"""Yoochoose dataset preparation.

Run `python src/datasets/yoochoose.py`.

By default it will try to download the Kaggle zip via `curl` (no credentials). If that
fails, download manually and place `recsys-challenge-2015.zip` into `data/raw/yoochoose/`.
"""

import logging
import subprocess  # nosec
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import (
    apply_filtering,
    extract_archive,
    process_interactions_ids,
    process_validation_schemes,
    RAW_DATA_DIR,
)
from src.utils import console_logging

DATASET_NAME = "yoochoose"

ZIP_FILENAME = "recsys-challenge-2015.zip"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015"
KAGGLE_API_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/chadgostopp/recsys-challenge-2015"
)

CLICKS_FILENAME = "yoochoose-clicks.dat"
BUYS_FILENAME = "yoochoose-buys.dat"

KAGGLE_DOWNLOAD_CMD = """#!/bin/bash
curl -L -o data/raw/yoochoose/recsys-challenge-2015.zip\\
  https://www.kaggle.com/api/v1/datasets/download/chadgostopp/recsys-challenge-2015
"""

SESSION_CORE = 2
ITEM_CORE = 5


def _try_download_zip(zip_path: Path) -> None:
    if zip_path.exists():
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.check_call(  # nosec
            ["curl", "-L", "-o", str(zip_path), KAGGLE_API_URL]
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Yoochoose via curl. "
            f"Download it manually from {KAGGLE_DATASET_URL} and place it at {zip_path}"
        ) from e


def download_and_extract(raw_data_path: Path) -> None:
    zip_path = raw_data_path / ZIP_FILENAME
    _try_download_zip(zip_path)

    clicks_present = bool(list(raw_data_path.rglob(CLICKS_FILENAME)))
    buys_present = bool(list(raw_data_path.rglob(BUYS_FILENAME)))
    if clicks_present and buys_present:
        return

    extract_archive(zip_path, raw_data_path, archive_type="zip")


def _read_clicks(raw_data_path: Path) -> pd.DataFrame:
    clicks_path = _resolve_raw_file(raw_data_path, CLICKS_FILENAME)
    clicks = pd.read_csv(
        clicks_path,
        header=None,
        names=["session_id", "timestamp", "item_id", "category"],
    )
    clicks["timestamp"] = pd.to_datetime(clicks["timestamp"], utc=True, errors="coerce")
    clicks = clicks.dropna(subset=["timestamp"])
    clicks["timestamp"] = clicks["timestamp"].dt.tz_convert(None)
    clicks.rename(columns={"timestamp": Columns.Datetime}, inplace=True)
    return clicks


def _read_buys(raw_data_path: Path) -> pd.DataFrame:
    buys_path = _resolve_raw_file(raw_data_path, BUYS_FILENAME)
    buys = pd.read_csv(
        buys_path,
        header=None,
        names=["session_id", "timestamp", "item_id", "price", "quantity"],
    )
    buys["timestamp"] = pd.to_datetime(buys["timestamp"], utc=True, errors="coerce")
    buys = buys.dropna(subset=["timestamp"])
    buys["timestamp"] = buys["timestamp"].dt.tz_convert(None)
    return buys


def process_raw_file(raw_data_path: Path) -> pd.DataFrame:
    clicks = _read_clicks(raw_data_path)
    buys = _read_buys(raw_data_path)

    buy_pairs = buys[["session_id", "item_id"]].drop_duplicates()
    buy_pairs["is_buy"] = 1
    clicks = clicks.merge(buy_pairs, on=["session_id", "item_id"], how="left")
    clicks["is_buy"] = clicks["is_buy"].fillna(0).astype(int)

    clicks.rename(
        columns={
            "session_id": Columns.User,
            "item_id": Columns.Item,
        },
        inplace=True,
    )

    clicks[Columns.Weight] = 1.0

    clicks["_row"] = range(len(clicks))
    clicks.sort_values([Columns.Datetime, "_row"], inplace=True, kind="mergesort")
    clicks.drop(columns=["_row"], inplace=True)

    clicks = apply_filtering(clicks, user_core=SESSION_CORE, item_core=ITEM_CORE)

    dataset_path = Path("data") / DATASET_NAME
    clicks = process_interactions_ids(clicks, dataset_path)

    return clicks


def _resolve_raw_file(raw_data_path: Path, filename: str) -> Path:
    direct = raw_data_path / filename
    if direct.is_file():
        return direct
    matches = list(raw_data_path.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Expected {filename} under {raw_data_path}")
    raise RuntimeError(f"Multiple {filename} found under {raw_data_path}: {matches}")


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    raw_data_path = RAW_DATA_DIR / DATASET_NAME
    download_and_extract(raw_data_path)

    process_validation_schemes(DATASET_NAME, process_raw_file)

