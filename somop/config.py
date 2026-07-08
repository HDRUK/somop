from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat, validator, PositiveInt


class ThresholdSwitch(BaseModel):
    """Map a numeric value to a concept_id based on a threshold.

    If value_as_number > value → above_concept_id, otherwise → below_concept_id.
    """
    value: float
    above_concept_id: int
    below_concept_id: Optional[int] = None


class Item(BaseModel):
    concept_id: conint(ge=0)
    p: confloat(ge=0, le=1) = 0.0

    unit_concept_id: Optional[int] = None
    dist: Optional[Literal["normal", "lognormal", "uniform"]] = None
    param1: Optional[float] = None  # mean/mu/low
    param2: Optional[float] = None  # sd/sigma/high
    threshold: Optional[ThresholdSwitch] = None


class TableConfig(BaseModel):
    """Generic collection of concept items. Empty items = disabled."""

    enabled: bool = True
    items: List[Item] = Field(default_factory=list)


class GenderProb(BaseModel):
    concept_id: conint(ge=0)
    p: confloat(ge=0, le=1)


class ConceptProb(BaseModel):
    concept_id: int
    p: float


class PersonConfig(BaseModel):
    enabled: bool = True
    n_people: conint(ge=1) = 10000
    genders: List[GenderProb] = Field(
        default_factory=lambda: [
            GenderProb(concept_id=8507, p=0.5),  # male
            GenderProb(concept_id=8532, p=0.5),  # female
        ]
    )

    ethnicities: List[ConceptProb] = Field(
        default_factory=lambda: [ConceptProb(concept_id=0, p=1.0)]
    )

    races: List[ConceptProb] = Field(
        default_factory=lambda: [ConceptProb(concept_id=0, p=1.0)]
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


class DeathConfig(BaseModel):
    enabled: bool = True
    p: confloat(ge=0, le=1) = 0.0
    causes: List[ConceptProb] = Field(default_factory=list)
    death_type_concept_id: int = 32519  # EHR


class LocationItem(BaseModel):
    location_id: PositiveInt
    address_1: Optional[str] = None
    address_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    county: Optional[str] = None
    location_source_value: Optional[str] = None
    country_concept_id: Optional[int] = None
    country_source_value: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LocationConfig(BaseModel):
    enabled: bool = True
    items: List[LocationItem] = Field(default_factory=list)
    prebuilt_file: Optional[str] = None  # path to a pre-generated LOCATION CSV


class DatasetEntry(BaseModel):
    name: str
    config: str
    collection_id: str
    db_name: Optional[str] = None
    db_password: Optional[str] = None
    db_url: Optional[str] = None
    db_port: Optional[int] = None
    drop_db: bool = True


class MultiConfig(BaseModel):
    api_url: str
    api_username: str
    api_password: str
    db_password: str = "postgres"
    bunny_build: Optional[str] = None
    concepts: Optional[str] = None
    datasets: List[DatasetEntry]


class Config(BaseModel):
    person: PersonConfig = PersonConfig()
    drug_exposure: TableConfig = TableConfig()
    measurement: TableConfig = TableConfig()
    condition: TableConfig = TableConfig()
    observation: TableConfig = TableConfig()
    procedure: TableConfig = TableConfig()
    specimen: TableConfig = TableConfig()
    death: DeathConfig = DeathConfig()
    location: LocationConfig = LocationConfig()

    interactions: InteractionEffects = InteractionEffects()
    # global
    seed: Optional[int] = None
    chunk_size: int = 100_000
    out_dir: str = "."
