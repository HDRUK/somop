from __future__ import annotations
import os
import random
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pydantic_core import PydanticUndefined
from .config import Config
from .utils import ensure_dir, write_df
from .dist import _sample_ages

from .omop.v5_4_3 import (
    Person,
    DrugExposure,
    Measurement,
    ConditionOccurrence,
    Observation,
    ProcedureOccurrence,
    Specimen,
    Death,
    Location,
)


class ColorFormatter(logging.Formatter):
    # ANSI escape codes
    GREEN = "\033[92m"
    RESET = "\033[0m"

    def formatTime(self, record, datefmt=None):
        s = super().formatTime(record, datefmt)
        return f"{self.GREEN}{s}{self.RESET}"


# configure root logger
handler = logging.StreamHandler()
formatter = ColorFormatter(
    "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

today = datetime.today().date()
_TODAY64 = np.datetime64(today, "D")


def _load_config(config: Optional[dict | str | os.PathLike]) -> Config:
    if config is None:
        return Config()
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    elif isinstance(config, dict):
        return Config(**config)
    else:
        raise TypeError("config must be None, dict, or path to YAML")


def _field_defaults(model_cls) -> dict:
    """Default value for each model field (None where the field is required)."""
    out = {}
    for name, f in model_cls.model_fields.items():
        out[name] = None if f.default is PydanticUndefined else f.default
    return out


def _build_df(model_cls, n: int, columns: dict) -> pd.DataFrame:
    """Build a DataFrame with every model field as a column, in declared order.

    ``columns`` supplies the values we computed; any field not supplied is
    filled with its model default (constant defaults are broadcast, ``None``
    defaults become empty cells on write). This reproduces the shape/order that
    ``model_dump()`` used to produce, without per-row model construction.
    """
    fields = list(model_cls.model_fields.keys())
    defaults = _field_defaults(model_cls)
    data = {}
    for name in fields:
        if name in columns:
            data[name] = columns[name]
        else:
            default = defaults[name]
            data[name] = np.full(n, np.nan) if default is None else [default] * n
    return pd.DataFrame(data, columns=fields)


def _past_dates(n: int, rng, max_years: int = 10) -> np.ndarray:
    """Vectorised equivalent of ``random_past_date`` — returns ``datetime.date`` objects."""
    max_days = max_years * 365
    offsets = rng.integers(0, max_days + 1, size=n).astype("timedelta64[D]")
    return (_TODAY64 - offsets).astype(object)


def _sample_offsets(rng, size: int, p_arr: np.ndarray):
    """Vectorised Bernoulli sampling across items.

    For each item, the number of persons who "fire" follows Binomial(size, p),
    which is statistically identical to the old ``np.random.rand(size) < p`` mask
    but costs one draw per item instead of ``size`` draws per item. Returns the
    per-item hit counts, the flattened item index for each fired row, and a
    uniform person offset (0..size-1) for each fired row.
    """
    counts = rng.binomial(size, p_arr)
    total = int(counts.sum())
    if total == 0:
        return counts, 0, None, None
    item_idx = np.repeat(np.arange(len(p_arr)), counts)
    offsets = rng.integers(0, size, total)
    return counts, total, item_idx, offsets


def generate(
    config: Optional[dict | str | os.PathLike] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate data using a YAML (or dict) configuration.

    Parameters
    ----------
    config : path to YAML, dict, or None
        The configuration describing which tables to create and with what probabilities/parameters.
    overrides : dict
        Optional shallow overrides applied after loading the config (e.g., {'person': {'n_people': 1000}})

    Returns
    -------
    Dict[str, str]
        Mapping of logical table names to written file paths.
    """
    cfg = _load_config(config)
    if overrides:
        # simple shallow merge
        import copy

        merged = copy.deepcopy(cfg.model_dump())
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        cfg = Config(**merged)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dir = cfg.out_dir
    ensure_dir(out_dir)

    paths = {
        "person": os.path.join(out_dir, "PERSON.csv"),
        "drug": os.path.join(out_dir, "DRUG_EXPOSURE.csv"),
        "meas": os.path.join(out_dir, "MEASUREMENT.csv"),
        "cond": os.path.join(out_dir, "CONDITION_OCCURRENCE.csv"),
        "obs": os.path.join(out_dir, "OBSERVATION.csv"),
        "proc": os.path.join(out_dir, "PROCEDURE_OCCURRENCE.csv"),
        "spec": os.path.join(out_dir, "SPECIMEN.csv"),
        "death": os.path.join(out_dir, "DEATH.csv"),
        "loc": os.path.join(out_dir, "LOCATION.csv"),
    }
    # clear existing outputs if any
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

    n = cfg.person.n_people
    chunk = max(1, int(cfg.chunk_size))
    n_chunks = (n + chunk - 1) // chunk

    ids = {"drug": 1, "meas": 1, "cond": 1, "obs": 1, "proc": 1, "spec": 1}
    wrote = {k: False for k in paths}

    # Pre-compute per-item concept id and probability arrays (constant across chunks).
    def _prep(items):
        return (
            np.array([it.concept_id for it in items], dtype=np.int64),
            np.array([it.p for it in items], dtype=float),
        )

    drug_ids_arr, drug_p = _prep(cfg.drug_exposure.items) if cfg.drug_exposure.items else (None, None)
    cond_ids_arr, cond_p = _prep(cfg.condition.items) if cfg.condition.items else (None, None)
    obs_ids_arr, obs_p = _prep(cfg.observation.items) if cfg.observation.items else (None, None)
    proc_ids_arr, proc_p = _prep(cfg.procedure.items) if cfg.procedure.items else (None, None)
    spec_ids_arr, spec_p = _prep(cfg.specimen.items) if cfg.specimen.items else (None, None)
    spec_units = (
        np.array([it.unit_concept_id for it in cfg.specimen.items], dtype=object)
        if cfg.specimen.items
        else None
    )

    # LOCATION — static reference table, written once before the chunk loop
    location_ids: list = []
    if cfg.location.enabled and cfg.location.items:
        loc_models = [
            Location(
                location_id=item.location_id,
                address_1=item.address_1,
                address_2=item.address_2,
                city=item.city,
                state=item.state,
                zip=item.zip,
                county=item.county,
                location_source_value=item.location_source_value,
                country_concept_id=item.country_concept_id,
                country_source_value=item.country_source_value,
                latitude=item.latitude,
                longitude=item.longitude,
            )
            for item in cfg.location.items
        ]
        loc_df = pd.DataFrame(
            (m.model_dump() for m in loc_models),
            columns=list(Location.model_fields.keys()),
        )
        write_df(loc_df, paths["loc"], header=True)
        wrote["loc"] = True
        location_ids = [item.location_id for item in cfg.location.items]
        logger.info("LOCATION: wrote %s rows", len(loc_models))
    elif cfg.location.enabled and cfg.location.prebuilt_file:
        loc_df = pd.read_csv(cfg.location.prebuilt_file, sep="\t", dtype={"location_id": int})
        write_df(loc_df, paths["loc"], header=True)
        wrote["loc"] = True
        location_ids = loc_df["location_id"].tolist()
        logger.info("LOCATION: loaded %s rows from %s", len(loc_df), cfg.location.prebuilt_file)
    location_ids_arr = np.array(location_ids) if location_ids else None

    def _write_events(key, model_cls, total, item_idx, offsets, ids_arr, start, extra=None):
        """Assemble and append a generic event table (id, person, concept, date)."""
        columns = {
            model_cls_pk[key]: np.arange(ids[key], ids[key] + total),
            "person_id": start + offsets,
            model_cls_concept[key]: ids_arr[item_idx],
            model_cls_date[key]: _past_dates(total, rng),
        }
        if extra:
            columns.update(extra)
        df = _build_df(model_cls, total, columns)
        write_df(df, paths[key], header=not wrote[key])
        wrote[key] = True
        ids[key] += total

    model_cls_pk = {
        "drug": "drug_exposure_id",
        "cond": "condition_occurrence_id",
        "obs": "observation_id",
        "proc": "procedure_occurrence_id",
        "spec": "specimen_id",
    }
    model_cls_concept = {
        "drug": "drug_concept_id",
        "cond": "condition_concept_id",
        "obs": "observation_concept_id",
        "proc": "procedure_concept_id",
        "spec": "specimen_concept_id",
    }
    model_cls_date = {
        "drug": "drug_exposure_start_date",
        "cond": "condition_start_date",
        "obs": "observation_date",
        "proc": "procedure_date",
        "spec": "specimen_date",
    }

    for c in range(n_chunks):
        start = c * chunk + 1
        size = min(chunk, n - (start - 1))

        logger.info(
            "Chunk %s/%s: generating %s persons (ids %s–%s)",
            c + 1,
            n_chunks,
            size,
            start,
            start + size - 1,
        )

        # PERSON
        if cfg.person.enabled:
            genders = [g.concept_id for g in cfg.person.genders]
            gprobs = [g.p for g in cfg.person.genders]
            gvals = rng.choice(genders, size=size, p=np.array(gprobs))

            eths = [e.concept_id for e in cfg.person.ethnicities]
            eth_probs = np.array([e.p for e in cfg.person.ethnicities], dtype=float)
            evals = rng.choice(eths, size=size, p=eth_probs / eth_probs.sum())

            races = [e.concept_id for e in cfg.person.races]
            race_probs = np.array([e.p for e in cfg.person.races], dtype=float)
            rvals = rng.choice(races, size=size, p=race_probs / race_probs.sum())

            lvals = (
                rng.choice(location_ids_arr, size=size)
                if location_ids_arr is not None
                else np.full(size, np.nan)
            )

            ages_years = _sample_ages(
                size=size,
                dist=cfg.person.age_dist,
                p1=cfg.person.age_param1,
                p2=cfg.person.age_param2,
                min_age=cfg.person.min_age,
                max_age=cfg.person.max_age,
                rng=rng,
            ).astype("int64")

            # Uniform day within the year-long window ending `age` years ago.
            days_back = (ages_years * 365 + rng.integers(0, 365, size=size)).astype(
                "timedelta64[D]"
            )
            bd = _TODAY64 - days_back  # datetime64[D]
            years = bd.astype("datetime64[Y]").astype(int) + 1970
            months = bd.astype("datetime64[M]").astype(int) % 12 + 1
            days = (bd - bd.astype("datetime64[M]")).astype("timedelta64[D]").astype(int) + 1

            person_df = _build_df(
                Person,
                size,
                {
                    "person_id": np.arange(start, start + size, dtype=int),
                    "gender_concept_id": gvals,
                    "year_of_birth": years,
                    "month_of_birth": months,
                    "day_of_birth": days,
                    "birth_datetime": bd.astype(object),
                    "race_concept_id": rvals,
                    "ethnicity_concept_id": evals,
                    "location_id": lvals,
                },
            )
            write_df(person_df, paths["person"], header=not wrote["person"])
            wrote["person"] = True

        had_drug = np.zeros(size, dtype=bool)
        had_cond = np.zeros(size, dtype=bool)

        # DRUG_EXPOSURE
        if cfg.drug_exposure.enabled and cfg.drug_exposure.items:
            counts, total, item_idx, offsets = _sample_offsets(rng, size, np.clip(drug_p, 0, 1))
            if total:
                _write_events(
                    "drug", DrugExposure, total, item_idx, offsets, drug_ids_arr, start,
                    extra={"drug_exposure_end_date": _past_dates(total, rng)},
                )
                had_drug[offsets] = True

        # CONDITION_OCCURRENCE
        if cfg.condition.enabled and cfg.condition.items:
            mult = 1.0
            if had_drug.any():
                mult *= float(cfg.interactions.after_drug_exposure.get("condition", 1.0))
            counts, total, item_idx, offsets = _sample_offsets(
                rng, size, np.clip(cond_p * mult, 0, 1)
            )
            if total:
                _write_events("cond", ConditionOccurrence, total, item_idx, offsets, cond_ids_arr, start)
                had_cond[offsets] = True

        # OBSERVATION
        if cfg.observation.enabled and cfg.observation.items:
            mult = 1.0
            if had_drug.any():
                mult *= float(cfg.interactions.after_drug_exposure.get("observation", 1.0))
            counts, total, item_idx, offsets = _sample_offsets(
                rng, size, np.clip(obs_p * mult, 0, 1)
            )
            if total:
                _write_events("obs", Observation, total, item_idx, offsets, obs_ids_arr, start)

        # MEASUREMENT (per-item values / thresholds handled over firing items only)
        if cfg.measurement.enabled and cfg.measurement.items:
            mult = 1.0
            if had_drug.any():
                mult *= float(cfg.interactions.after_drug_exposure.get("measurement", 1.0))
            if had_cond.any():
                mult *= float(cfg.interactions.after_condition.get("measurement", 1.0))

            p_eff = np.clip(
                np.array([it.p for it in cfg.measurement.items], dtype=float) * mult, 0, 1
            )
            counts = rng.binomial(size, p_eff)
            total = int(counts.sum())
            if total:
                person_ids = np.empty(total, dtype=np.int64)
                concept_ids = np.empty(total, dtype=np.int64)
                unit_ids = np.empty(total, dtype=object)
                values = np.full(total, np.nan)
                value_concepts = np.empty(total, dtype=object)
                value_concepts[:] = None

                pos = 0
                for j in np.nonzero(counts)[0]:
                    item = cfg.measurement.items[j]
                    k = int(counts[j])
                    sl = slice(pos, pos + k)
                    person_ids[sl] = start + rng.integers(0, size, k)
                    concept_ids[sl] = item.concept_id
                    unit_ids[sl] = getattr(item, "unit_concept_id", 0)

                    if item.dist:
                        a = item.param1 if item.param1 is not None else 0.0
                        b = item.param2 if item.param2 is not None else 1.0
                        if item.dist == "normal":
                            v = rng.normal(a, b, size=k)
                        elif item.dist == "lognormal":
                            v = rng.lognormal(a, b, size=k)
                        else:
                            v = rng.uniform(a, b, size=k)
                        values[sl] = v
                        if item.threshold is not None:
                            t = item.threshold
                            value_concepts[sl] = np.where(
                                v > t.value, t.above_concept_id, t.below_concept_id
                            )
                    pos += k

                meas_df = _build_df(
                    Measurement,
                    total,
                    {
                        "measurement_id": np.arange(ids["meas"], ids["meas"] + total),
                        "person_id": person_ids,
                        "measurement_concept_id": concept_ids,
                        "measurement_date": _past_dates(total, rng),
                        "value_as_number": values,
                        "value_as_concept_id": value_concepts,
                        "unit_concept_id": unit_ids,
                    },
                )
                write_df(meas_df, paths["meas"], header=not wrote["meas"])
                wrote["meas"] = True
                ids["meas"] += total

        # PROCEDURE_OCCURRENCE
        if cfg.procedure.enabled and cfg.procedure.items:
            mult = 1.0
            if had_drug.any():
                mult *= float(cfg.interactions.after_drug_exposure.get("procedure", 1.0))
            if had_cond.any():
                mult *= float(cfg.interactions.after_condition.get("procedure", 1.0))
            counts, total, item_idx, offsets = _sample_offsets(
                rng, size, np.clip(proc_p * mult, 0, 1)
            )
            if total:
                proc_dates = _past_dates(total, rng)
                _write_events(
                    "proc", ProcedureOccurrence, total, item_idx, offsets, proc_ids_arr, start,
                    extra={
                        "procedure_date": proc_dates,
                        "procedure_end_date": proc_dates,
                        "quantity": [1] * total,
                    },
                )

        # SPECIMEN
        if cfg.specimen.enabled and cfg.specimen.items:
            counts, total, item_idx, offsets = _sample_offsets(rng, size, np.clip(spec_p, 0, 1))
            if total:
                _write_events(
                    "spec", Specimen, total, item_idx, offsets, spec_ids_arr, start,
                    extra={"unit_concept_id": spec_units[item_idx]},
                )

        # DEATH — at most one row per person
        if cfg.death.enabled and cfg.death.p > 0:
            n_dead = int(rng.binomial(size, cfg.death.p))
            if n_dead:
                dead_offsets = rng.integers(0, size, n_dead)
                cause_concepts = None
                if cfg.death.causes:
                    cause_ids = [cc.concept_id for cc in cfg.death.causes]
                    raw = np.array([cc.p for cc in cfg.death.causes], dtype=float)
                    cause_concepts = rng.choice(cause_ids, size=n_dead, p=raw / raw.sum())

                death_df = _build_df(
                    Death,
                    n_dead,
                    {
                        "person_id": start + dead_offsets,
                        "death_date": _past_dates(n_dead, rng),
                        "death_type_concept_id": [cfg.death.death_type_concept_id] * n_dead,
                        **(
                            {"cause_concept_id": cause_concepts}
                            if cause_concepts is not None
                            else {}
                        ),
                    },
                )
                write_df(death_df, paths["death"], header=not wrote["death"])
                wrote["death"] = True

    return paths
