# somop/utils.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

today = datetime.today().date()


def _df_from_models(models: list, model_cls) -> pd.DataFrame:
    """
    Convert a list of Pydantic v2 models to a DataFrame with columns
    in the model's declared field order.
    """
    rows = [m.model_dump() for m in models]
    columns = list(model_cls.model_fields.keys())
    for r in rows:
        for c in columns:
            r.setdefault(c, None)
    return pd.DataFrame(rows, columns=columns)


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists (no error if it already does).
    """
    os.makedirs(path, exist_ok=True)


def write_df(df: "pd.DataFrame", path: str, header: bool) -> None:
    """
    Append a DataFrame to a TSV file, writing the header only when requested.
    """
    df.to_csv(path, sep="\t", index=False, header=header, mode="a")


def random_birthdate(age, today=today, rng=np.random.default_rng()):
    start_date = today.replace(year=today.year - age - 1) + timedelta(days=1)
    end_date = today.replace(year=today.year - age)
    delta_days = int((end_date - start_date).days)

    return start_date + timedelta(days=int(rng.integers(0, delta_days + 1)))


def random_past_date(max_years: int = 10, rng=np.random.default_rng()):
    max_days = max_years * 365
    offset_days = int(rng.integers(0, max_days + 1))
    return today - timedelta(days=offset_days)
