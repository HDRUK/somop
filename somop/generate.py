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
    wrote = {
        "person": False,
        "drug": False,
        "meas": False,
        "cond": False,
        "obs": False,
        "proc": False,
        "spec": False,
        "death": False,
        "loc": False,
    }

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
            gprobs = [g.p for g in cfg.person.genders]
            gvals = _choice_with_probs(genders, gprobs, size)

            eths = [e.concept_id for e in cfg.person.ethnicities]
            eth_probs = [e.p for e in cfg.person.ethnicities]
            evals = _choice_with_probs(eths, eth_probs, size)

            races = [e.concept_id for e in cfg.person.races]
            race_probs = [e.p for e in cfg.person.races]
            rvals = _choice_with_probs(races, race_probs, size)

            person_ids = np.arange(start, start + size, dtype=int)

            lvals = (
                rng.choice(location_ids, size=size) if location_ids else [None] * size
            )

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
                    race_concept_id=int(r),
                    ethnicity_concept_id=int(e),
                    location_id=int(loc) if loc is not None else None,
                )
                for pid, g, e, r, d, loc in zip(person_ids, gvals, evals, rvals, birthdates, lvals)
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

                concept_ids_for_vals = None
                if item.threshold is not None and vals is not None:
                    t = item.threshold
                    concept_ids_for_vals = [
                        t.above_concept_id if v > t.value else t.below_concept_id
                        for v in vals
                    ]

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
                            value_as_concept_id=(
                                concept_ids_for_vals[k] if concept_ids_for_vals is not None else None
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

                # PROCEDURE_OCCURRENCE

        # PROCEDURE_OCCURRENCE
        if cfg.procedure.enabled and cfg.procedure.items:
            proc_models = []
            proc_multiplier = 1.0
            if had_drug.any():
                proc_multiplier *= float(
                    cfg.interactions.after_drug_exposure.get("procedure", 1.0)
                )
            if had_cond.any():
                proc_multiplier *= float(
                    cfg.interactions.after_condition.get("procedure", 1.0)
                )

            for item in cfg.procedure.items:
                p_eff = min(1.0, float(item.p) * proc_multiplier)
                mask = np.random.rand(size) < p_eff
                if mask.any():
                    idxs = np.where(mask)[0]
                    for i in idxs:
                        proc_date = random_past_date()
                        quantity = getattr(item, "quantity", 1)
                        if quantity in (None, 0):
                            quantity = 1

                        po = ProcedureOccurrence(
                            procedure_occurrence_id=ids["proc"],
                            person_id=int(start + i),
                            procedure_concept_id=int(item.concept_id),
                            procedure_date=proc_date,
                            procedure_datetime=None,
                            procedure_end_date=proc_date,
                            procedure_end_datetime=None,
                            procedure_type_concept_id=int(
                                getattr(item, "procedure_type_concept_id", 0)
                            ),
                            modifier_concept_id=getattr(
                                item, "modifier_concept_id", None
                            ),
                            quantity=quantity,
                            provider_id=getattr(item, "provider_id", None),
                            visit_occurrence_id=getattr(
                                item, "visit_occurrence_id", None
                            ),
                            visit_detail_id=getattr(item, "visit_detail_id", None),
                            procedure_source_value=getattr(
                                item, "procedure_source_value", None
                            ),
                            procedure_source_concept_id=getattr(
                                item, "procedure_source_concept_id", None
                            ),
                            modifier_source_value=getattr(
                                item, "modifier_source_value", None
                            ),
                        )

                        proc_models.append(po)
                        ids["proc"] += 1

            if proc_models:
                proc_df = pd.DataFrame(
                    (m.model_dump() for m in proc_models),
                    columns=list(ProcedureOccurrence.model_fields.keys()),
                )
                write_df(proc_df, paths["proc"], header=not wrote["proc"])
                wrote["proc"] = True

        # SPECIMEN
        if cfg.specimen.enabled and cfg.specimen.items:
            spec_models = []
            for item in cfg.specimen.items:
                mask = np.random.rand(size) < item.p
                if not mask.any():
                    continue
                idxs = np.where(mask)[0]
                for i in idxs:
                    spec_models.append(
                        Specimen(
                            specimen_id=ids["spec"],
                            person_id=int(start + i),
                            specimen_concept_id=int(item.concept_id),
                            specimen_date=random_past_date(),
                            unit_concept_id=getattr(item, "unit_concept_id", None),
                        )
                    )
                    ids["spec"] += 1

            if spec_models:
                spec_df = pd.DataFrame(
                    (m.model_dump() for m in spec_models),
                    columns=list(Specimen.model_fields.keys()),
                )
                write_df(spec_df, paths["spec"], header=not wrote["spec"])
                wrote["spec"] = True

        # DEATH — at most one row per person
        if cfg.death.enabled and cfg.death.p > 0:
            death_mask = np.random.rand(size) < cfg.death.p
            if death_mask.any():
                death_models = []
                dead_idxs = np.where(death_mask)[0]

                cause_ids = None
                cause_weights = None
                if cfg.death.causes:
                    cause_ids = [c.concept_id for c in cfg.death.causes]
                    raw_weights = np.array(
                        [c.p for c in cfg.death.causes], dtype=float
                    )
                    cause_weights = raw_weights / raw_weights.sum()

                for i in dead_idxs:
                    cause_concept_id = None
                    if cause_ids is not None:
                        cause_concept_id = int(
                            np.random.choice(cause_ids, p=cause_weights)
                        )
                    death_models.append(
                        Death(
                            person_id=int(start + i),
                            death_date=random_past_date(),
                            death_type_concept_id=cfg.death.death_type_concept_id,
                            cause_concept_id=cause_concept_id,
                        )
                    )

                death_df = pd.DataFrame(
                    (m.model_dump() for m in death_models),
                    columns=list(Death.model_fields.keys()),
                )
                write_df(death_df, paths["death"], header=not wrote["death"])
                wrote["death"] = True

    return paths
