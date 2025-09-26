from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat, validator


class Item(BaseModel):
    concept_id: conint(ge=0)
    p: confloat(ge=0, le=1) = 0.0

    unit_concept_id: Optional[int] = None
    dist: Optional[Literal["normal", "lognormal", "uniform"]] = None
    param1: Optional[float] = None  # mean/mu/low
    param2: Optional[float] = None  # sd/sigma/high


class TableConfig(BaseModel):
    """Generic collection of concept items. Empty items = disabled."""

    enabled: bool = True
    items: List[Item] = Field(default_factory=list)


class GenderProb(BaseModel):
    concept_id: conint(ge=0)
    p: confloat(ge=0, le=1)


class PersonConfig(BaseModel):
    enabled: bool = True
    n_people: conint(ge=1) = 10000
    genders: List[GenderProb] = Field(
        default_factory=lambda: [
            GenderProb(concept_id=8507, p=0.5),  # male
            GenderProb(concept_id=8532, p=0.5),  # female
        ]
    )

    age_dist: Optional[Literal["normal", "lognormal", "uniform"]] = "normal"
    age_param1: Optional[float] = 40.0
    age_param2: Optional[float] = 18.0
    min_age: float = 0.0
    max_age: float = 110.0

    @validator("genders")
    def probs_sum_to_one(cls, v):
        total = sum(g.p for g in v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Sum of gender probabilities must be 1.0")
        return v


class InteractionEffects(BaseModel):
    after_drug_exposure: Dict[str, float] = Field(default_factory=dict)
    after_condition: Dict[str, float] = Field(default_factory=dict)


class Config(BaseModel):
    person: PersonConfig = PersonConfig()
    drug_exposure: TableConfig = TableConfig()
    measurement: TableConfig = TableConfig()
    condition: TableConfig = TableConfig()
    observation: TableConfig = TableConfig()

    interactions: InteractionEffects = InteractionEffects()
    # global
    seed: Optional[int] = None
    chunk_size: int = 100_000
    out_dir: str = "."
