from src.datasets.bert_repro import DATASETS as BERT_REPRO_DATASETS, URL_TEMPLATE as BERT_REPRO_URL
from src.datasets.beeradvocate_r import DATASET_NAME as BEER_NAME, INTERACTIONS_FILENAME as BEER_FILE, URL as BEER_URL
from src.datasets.common import RAW_DATA_DIR, make_manual_download_bash
from src.datasets.kion_r import DATASET_NAME as KION_NAME, ZIP_FILENAME as KION_ZIP, URL as KION_URL, EXTRACTED_DIRNAME as KION_EXTRACTED
from src.datasets.ml_1m import (
    DATASET_NAME as ML1M_NAME,
    ZIP_FILENAME as ML1M_ZIP,
    URL as ML1M_URL,
    EXTRACTED_DIRNAME as ML1M_EXTRACTED,
    INTERACTIONS_FILENAME as ML1M_FILE,
)
from src.datasets.ml_20m import (
    DATASET_NAME as ML20M_NAME,
    ZIP_FILENAME as ML20M_ZIP,
    URL as ML20M_URL,
    EXTRACTED_DIRNAME as ML20M_EXTRACTED,
    INTERACTIONS_FILENAME as ML20M_FILE,
)
from src.datasets.retailrocket import DATASET_NAME as RR_NAME, ZIP_FILENAME as RR_ZIP, URL as RR_URL
from src.datasets.s3_repro import DATASET_VARIANTS as S3_VARIANTS, URL_TEMPLATE as S3_URL
from src.datasets.s3_repro_w_time import (
    DATASET_VARIANTS as S3_TIME_VARIANTS,
    URL as S3_TIME_URL,
    ZIP_FILENAME as S3_TIME_ZIP,
    EXTRACTED_DIRNAME as S3_TIME_EXTRACTED,
)
from src.datasets.yoochoose import DATASET_NAME as YC_NAME, ZIP_FILENAME as YC_ZIP, KAGGLE_API_URL as YC_URL


def _print(dataset_name: str, cmd: str) -> None:
    print(f"### {dataset_name}")
    print(cmd.rstrip())
    print()


def main() -> None:
    _print(
        ML1M_NAME,
        make_manual_download_bash(
            dataset_name=ML1M_NAME,
            url=ML1M_URL,
            raw_data_path=RAW_DATA_DIR / ML1M_NAME,
            filename=ML1M_ZIP,
            archive_type="zip",
            extracted_dirname=ML1M_EXTRACTED,
        ),
    )

    _print(
        ML20M_NAME,
        make_manual_download_bash(
            dataset_name=ML20M_NAME,
            url=ML20M_URL,
            raw_data_path=RAW_DATA_DIR / ML20M_NAME,
            filename=ML20M_ZIP,
            archive_type="zip",
            extracted_dirname=ML20M_EXTRACTED,
        ),
    )

    _print(
        KION_NAME,
        make_manual_download_bash(
            dataset_name=KION_NAME,
            url=KION_URL,
            raw_data_path=RAW_DATA_DIR / KION_NAME,
            filename=KION_ZIP,
            archive_type="zip",
            extracted_dirname=KION_EXTRACTED,
        ),
    )

    _print(
        BEER_NAME,
        make_manual_download_bash(
            dataset_name=BEER_NAME,
            url=BEER_URL,
            raw_data_path=RAW_DATA_DIR / BEER_NAME,
            filename=BEER_FILE,
        ),
    )

    _print(
        RR_NAME,
        make_manual_download_bash(
            dataset_name=RR_NAME,
            url=RR_URL,
            raw_data_path=RAW_DATA_DIR / RR_NAME,
            filename=RR_ZIP,
            archive_type="zip",
            extracted_dirname=None,
        ),
    )

    _print(
        YC_NAME,
        make_manual_download_bash(
            dataset_name=YC_NAME,
            url=YC_URL,
            raw_data_path=RAW_DATA_DIR / YC_NAME,
            filename=YC_ZIP,
            archive_type="zip",
            extracted_dirname=None,
        ),
    )

    for variant, remote_name in S3_VARIANTS.items():
        dataset_name = f"s3_{variant}"
        _print(
            dataset_name,
            make_manual_download_bash(
                dataset_name=dataset_name,
                url=S3_URL.format(remote_name),
                raw_data_path=RAW_DATA_DIR / dataset_name,
                filename=f"{variant}.txt",
            ),
        )

    for dataset_name in S3_TIME_VARIANTS.keys():
        _print(
            dataset_name,
            make_manual_download_bash(
                dataset_name=dataset_name,
                url=S3_TIME_URL,
                raw_data_path=RAW_DATA_DIR / dataset_name,
                filename=S3_TIME_ZIP,
                archive_type="zip",
                extracted_dirname=S3_TIME_EXTRACTED,
            ),
        )

    for base_name, filename in BERT_REPRO_DATASETS.items():
        dataset_name = f"{base_name}_repro"
        _print(
            dataset_name,
            make_manual_download_bash(
                dataset_name=dataset_name,
                url=BERT_REPRO_URL.format(filename=filename),
                raw_data_path=RAW_DATA_DIR / dataset_name,
                filename=filename,
            ),
        )

    _print(
        f"{ML1M_NAME} (raw file expected after extraction)",
        f"ls -lh {RAW_DATA_DIR / ML1M_NAME / ML1M_FILE}",
    )
    _print(
        f"{ML20M_NAME} (raw file expected after extraction)",
        f"ls -lh {RAW_DATA_DIR / ML20M_NAME / ML20M_FILE}",
    )


if __name__ == "__main__":
    main()


