import pandas
import yaml
import logging
import pyarrow.parquet


def get_data(path: str, limit: int = 0, batch_size: int = 10000) -> pandas.DataFrame:
    """
    Downloads data from a parquet file in batches.

    Args:
        path (str): The path to the parquet file.
        limit (int): The maximum number of rows to download.
        batch_size (int): The size of each batch.

    Returns:
        pandas.DataFrame: The downloaded data.
    """

    logging.info(f"Getting data from {path} in batches of {batch_size} rows")
    table = pyarrow.parquet.read_table(path)
    batches = table.to_batches(batch_size)

    data_frames = []
    total_rows = 0

    for batch in batches:
        df = batch.to_pandas()
        data_frames.append(df)
        total_rows += len(df)

        if limit > 0 and total_rows >= limit:
            break

    data = pandas.concat(data_frames, ignore_index=True)

    if limit > 0:
        data = data.head(limit)

    return data


def get_features_from_yml(path: str) -> list[str]:
    """
    Gets the features from a yml file.

    Args:
        path (str): The path to the yml file.

    Returns:
        list[str]: The features.
    """

    with open(path, "r") as file:
        features = yaml.safe_load(file)
    return features
