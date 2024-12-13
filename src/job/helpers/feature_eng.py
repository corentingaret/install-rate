import pandas
import numpy
import json


def get_app_already_displayed(row: pandas.Series, nb_days: int) -> bool:
    if nb_days not in [1, 7, 28]:
        raise ValueError("nb_days must be one of 1, 7, or 28")

    try:
        displays = json.loads(
            row[f"Impression Count in last {nb_days} days, by Promoted Entity"]
        )
        if row["promoted_entity"].lower() in displays.keys():
            return True
    except (TypeError, json.JSONDecodeError):
        return False
    return False


def get_previous_session_fs_cpm(row: pandas.Series) -> float:
    try:
        return row["previous_session_cpm"]["fs"]

    except (TypeError, KeyError):
        return 0


def feature_engineering(
    data: pandas.DataFrame, features: list[str]
) -> pandas.DataFrame:

    if "Log App installs in last 7 days" in features:

        data["Log App installs in last 7 days"] = data[
            "App installs in last 7 days"
        ].fillna(0)

        # Add 1 to avoid log(0)
        data["Log App installs in last 7 days"] = numpy.log1p(
            data["Log App installs in last 7 days"]
        )

    for nb_days in [1, 7, 28]:
        feature_name = f"Already displayed in last {nb_days} days"
        if feature_name in features:
            data[feature_name] = data.apply(
                get_app_already_displayed, axis=1, nb_days=nb_days
            )

    if "Previous session fs cpm" in features:
        data["Previous session fs cpm"] = data.apply(
            get_previous_session_fs_cpm, axis=1
        )

    return data
