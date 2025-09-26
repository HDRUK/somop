from __future__ import annotations
import os
import random
import yaml
import numpy as np
import pandas as pd
import shutil
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from .config import Config
from .utils import ensure_dir, write_df, random_birthdate, random_past_date
from .dist import _sample_ages

from .omop.v5_4_3 import (
    Person,
    DrugExposure,
    Measurement,
    ConditionOccurrence,
    Observation,
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


def _choice_with_probs(items: List, probs: List[float], size: int):
    return np.random.choice(items, size=size, p=np.array(probs))


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

    out_dir = cfg.out_dir
    ensure_dir(out_dir)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    file_path = os.path.join(script_dir, "..", "data", "CONCEPT.csv")
    dest_path = os.path.join(out_dir, "CONCEPT.csv")
    shutil.copy(file_path, dest_path)

    paths = {
        "person": os.path.join(out_dir, "PERSON.csv"),
        "drug": os.path.join(out_dir, "DRUG_EXPOSURE.csv"),
        "meas": os.path.join(out_dir, "MEASUREMENT.csv"),
        "cond": os.path.join(out_dir, "CONDITION_OCCURRENCE.csv"),
        "obs": os.path.join(out_dir, "OBSERVATION.csv"),
    }
    # clear existing outputs if any
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

    n = cfg.person.n_people
    chunk = max(1, int(cfg.chunk_size))
    n_chunks = (n + chunk - 1) // chunk

    ids = {"drug": 1, "meas": 1, "cond": 1, "obs": 1}
    wrote = {"person": False, "drug": False, "meas": False, "cond": False, "obs": False}

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
            rng = np.random.default_rng()
            genders = [g.concept_id for g in cfg.person.genders]
            probs = [g.p for g in cfg.person.genders]
            gvals = _choice_with_probs(genders, probs, size)
            person_ids = np.arange(start, start + size, dtype=int)

            ages_years = _sample_ages(
                size=size,
                dist=cfg.person.age_dist,
                p1=cfg.person.age_param1,
                p2=cfg.person.age_param2,
                min_age=cfg.person.min_age,
                max_age=cfg.person.max_age,
                rng=rng,
            )

            birthdates = np.array(
                [random_birthdate(int(age), today, rng) for age in ages_years]
            )

            people = [
                Person(
                    person_id=int(pid),
                    gender_concept_id=int(g),
                    birth_datetime=d,
                    year_of_birth=d.year,
                    month_of_birth=d.month,
                    day_of_birth=d.day,
                    race_concept_id=0,
                    ethnicity_concept_id=0,
                )
                for pid, g, d in zip(person_ids, gvals, birthdates)
            ]

            person_df = pd.DataFrame(
                (p.model_dump() for p in people),
                columns=list(Person.model_fields.keys()),
            )
            write_df(person_df, paths["person"], header=not wrote["person"])
            wrote["person"] = True
        else:
            person_ids = np.arange(start, start + size, dtype=int)

        had_drug = np.zeros(size, dtype=bool)
        had_cond = np.zeros(size, dtype=bool)

        if cfg.drug_exposure.enabled and cfg.drug_exposure.items:
            drug_models = []
            for item in cfg.drug_exposure.items:
                mask = np.random.rand(size) < item.p
                if mask.any():
                    idxs = np.where(mask)[0]
                    for i in idxs:
                        drug_models.append(
                            DrugExposure(
                                drug_exposure_id=ids["drug"],
                                person_id=int(start + i),
                                drug_concept_id=int(item.concept_id),
                                drug_type_concept_id=0,
                                drug_exposure_start_date=random_past_date(),
                                drug_exposure_end_date=random_past_date(),
                            )
                        )
                        ids["drug"] += 1
                    had_drug[mask] = True

            if drug_models:
                drug_df = pd.DataFrame(
                    (m.model_dump() for m in drug_models),
                    columns=list(DrugExposure.model_fields.keys()),
                )
                write_df(drug_df, paths["drug"], header=not wrote["drug"])
                wrote["drug"] = True

        # CONDITION_OCCURRENCE
        if cfg.condition.enabled and cfg.condition.items:
            cond_models = []
            cond_multiplier = 1.0
            if had_drug.any():
                cond_multiplier *= float(
                    cfg.interactions.after_drug_exposure.get("condition", 1.0)
                )

            for item in cfg.condition.items:
                p_eff = min(1.0, float(item.p) * cond_multiplier)
                mask = np.random.rand(size) < p_eff
                if mask.any():
                    idxs = np.where(mask)[0]
                    for i in idxs:
                        cond_models.append(
                            ConditionOccurrence(
                                condition_occurrence_id=ids["cond"],
                                person_id=int(start + i),
                                condition_concept_id=int(item.concept_id),
                                condition_type_concept_id=0,
                                condition_start_date=random_past_date(),
                            )
                        )
                        ids["cond"] += 1
                    had_cond[mask] = True

            if cond_models:
                cond_df = pd.DataFrame(
                    (m.model_dump() for m in cond_models),
                    columns=list(ConditionOccurrence.model_fields.keys()),
                )
                write_df(cond_df, paths["cond"], header=not wrote["cond"])
                wrote["cond"] = True

        # OBSERVATION
        if cfg.observation.enabled and cfg.observation.items:
            obs_models = []
            obs_multiplier = 1.0
            if had_drug.any():
                obs_multiplier *= float(
                    cfg.interactions.after_drug_exposure.get("observation", 1.0)
                )

            for item in cfg.observation.items:
                p_eff = min(1.0, float(item.p) * obs_multiplier)
                mask = np.random.rand(size) < p_eff
                if mask.any():
                    idxs = np.where(mask)[0]
                    for i in idxs:
                        obs_models.append(
                            Observation(
                                observation_id=ids["obs"],
                                person_id=int(start + i),
                                observation_concept_id=int(item.concept_id),
                                observation_date=random_past_date(),
                                observation_type_concept_id=0,
                            )
                        )
                        ids["obs"] += 1

            if obs_models:
                obs_df = pd.DataFrame(
                    (m.model_dump() for m in obs_models),
                    columns=list(Observation.model_fields.keys()),
                )
                write_df(obs_df, paths["obs"], header=not wrote["obs"])
                wrote["obs"] = True

        # MEASUREMENT
        if cfg.measurement.enabled and cfg.measurement.items:
            meas_models = []
            meas_multiplier = 1.0
            if had_drug.any():
                meas_multiplier *= float(
                    cfg.interactions.after_drug_exposure.get("measurement", 1.0)
                )
            if had_cond.any():
                meas_multiplier *= float(
                    cfg.interactions.after_condition.get("measurement", 1.0)
                )

            for item in cfg.measurement.items:
                p_eff = min(1.0, float(item.p) * meas_multiplier)
                mask = np.random.rand(size) < p_eff
                if not mask.any():
                    continue

                idxs = np.where(mask)[0]

                vals = None
                if getattr(item, "dist", None):
                    a = item.param1 if item.param1 is not None else 0.0
                    b = item.param2 if item.param2 is not None else 1.0
                    if item.dist == "normal":
                        vals = np.random.normal(a, b, size=len(idxs))
                    elif item.dist == "lognormal":
                        vals = np.random.lognormal(a, b, size=len(idxs))
                    else:
                        vals = np.random.uniform(a, b, size=len(idxs))

                for k, i in enumerate(idxs):
                    meas_models.append(
                        Measurement(
                            measurement_id=ids["meas"],
                            person_id=int(start + i),
                            measurement_concept_id=int(item.concept_id),
                            measurement_date=random_past_date(),
                            value_as_number=(
                                float(vals[k]) if vals is not None else None
                            ),
                            unit_concept_id=getattr(item, "unit_concept_id", 0),
                            measurement_type_concept_id=0,
                        )
                    )
                    ids["meas"] += 1

            if meas_models:
                meas_df = pd.DataFrame(
                    (m.model_dump() for m in meas_models),
                    columns=list(Measurement.model_fields.keys()),
                )
                write_df(meas_df, paths["meas"], header=not wrote["meas"])
                wrote["meas"] = True

    return paths
