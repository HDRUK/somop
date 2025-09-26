from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Person(BaseModel):
    person_id: PositiveInt
    gender_concept_id: NonNegativeInt
    year_of_birth: Optional[int] = None
    month_of_birth: Optional[int] = None
    day_of_birth: Optional[int] = None
    birth_datetime: Optional[datetime] = None
    race_concept_id: Optional[NonNegativeInt] = None
    ethnicity_concept_id: Optional[NonNegativeInt] = None
    location_id: Optional[NonNegativeInt] = None
    provider_id: Optional[NonNegativeInt] = None
    care_site_id: Optional[NonNegativeInt] = None
    person_source_value: Optional[str] = None
    gender_source_value: Optional[str] = None
    gender_source_concept_id: Optional[NonNegativeInt] = None
    race_source_value: Optional[str] = None
    race_source_concept_id: Optional[NonNegativeInt] = None
    ethnicity_source_value: Optional[str] = None
    ethnicity_source_concept_id: Optional[NonNegativeInt] = None
