"""RetailRocket dataset preparation.

Downloads and prepares interactions for evaluation schemes.

If download fails, run:
#!/bin/bash
curl -L -o ~/Downloads/ecommerce-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
"""

import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import (
    RAW_DATA_DIR,
    apply_filtering,
    extract_archive,
    process_interactions_ids,
    process_validation_schemes,
    shell,
)
from src.utils import console_logging

DATASET_NAME = "retailrocket"

ZIP_FILENAME = "ecommerce-dataset.zip"
URL = "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset"
KAGGLE_DOWNLOAD_CMD = """#!/bin/bash
curl -L -o ~/Downloads/ecommerce-dataset.zip\\
  https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
"""

INTERACTIONS_FILENAME = "events.csv"

USER_CORE = 2
ITEM_CORE = 5

TIME_SPLIT_TEST_SIZE = "7D"

KEEP_EVENTS = {"view", "addtocart"}  # SA2C-aligned: drop transactions


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


def _bash(cmd: str) -> None:
    logging.info(f"running bash command:\n{cmd}")
    subprocess.check_call(["bash", "-lc", cmd])


def ensure_raw_dataset() -> None:
    raw_data_path = RAW_DATA_DIR / DATASET_NAME
    raw_data_path.mkdir(parents=True, exist_ok=True)

    try:
        _resolve_raw_file(raw_data_path, INTERACTIONS_FILENAME)
        logging.info("dataset is already extracted")
        return
    except FileNotFoundError:
        pass

    archive_path = raw_data_path / ZIP_FILENAME

    if not archive_path.is_file():
        downloads_zip = Path.home() / "Downloads" / ZIP_FILENAME
        _bash(
            "curl -L -o ~/Downloads/ecommerce-dataset.zip\\\n"
            "  https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset"
        )
        if not downloads_zip.is_file():
            raise FileNotFoundError(
                f"Download did not produce {downloads_zip}. "
                f"Please download manually:\n{KAGGLE_DOWNLOAD_CMD}"
            )
        shutil.copy2(downloads_zip, archive_path)

    extract_archive(archive_path, raw_data_path, archive_type="zip")

    _resolve_raw_file(raw_data_path, INTERACTIONS_FILENAME)


def process_raw_file(raw_data_path: Path) -> pd.DataFrame:
    events_path = _resolve_raw_file(raw_data_path, INTERACTIONS_FILENAME)
    events = pd.read_csv(events_path)

    events.columns = [str(c).strip().lower() for c in events.columns]

    col_map = {
        "visitorid": Columns.User,
        "itemid": Columns.Item,
        "timestamp": Columns.Datetime,
    }
    events.rename(columns=col_map, inplace=True)

    missing = [c for c in [Columns.User, Columns.Item, Columns.Datetime] if c not in events.columns]
    if missing:
        raise KeyError(f"Missing required columns in {events_path}: {missing}")

    if "event" not in events.columns:
        raise KeyError(f"Missing required column 'event' in {events_path}")

    events[Columns.Datetime] = pd.to_datetime(
        events[Columns.Datetime], unit="ms", utc=True, errors="coerce"
    )
    events = events.dropna(subset=[Columns.Datetime])
    events[Columns.Datetime] = events[Columns.Datetime].dt.tz_convert(None)

    events["event"] = events["event"].astype(str).str.lower()
    events = events[events["event"].isin(KEEP_EVENTS)]
    events["is_buy"] = (events["event"] == "addtocart").astype(int)

    events[Columns.Weight] = 1.0

    events["_row"] = range(len(events))
    events.sort_values([Columns.Datetime, "_row"], inplace=True, kind="mergesort")
    events.drop(columns=["_row"], inplace=True)

    events = apply_filtering(events, user_core=USER_CORE, item_core=ITEM_CORE)

    dataset_path = Path("data") / DATASET_NAME
    events = process_interactions_ids(events, dataset_path)
    return events


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    ensure_raw_dataset()

    process_validation_schemes(
        DATASET_NAME,
        process_raw_file,
        select_val_schemes=["leave_one_out.py", "time_split.py"],
        time_split_test_size=TIME_SPLIT_TEST_SIZE,
    )


