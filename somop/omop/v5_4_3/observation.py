from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Observation(BaseModel):
    observation_id: PositiveInt
    person_id: PositiveInt
    observation_concept_id: NonNegativeInt
    observation_date: date
    observation_datetime: Optional[datetime] = None
    observation_type_concept_id: Optional[NonNegativeInt] = 0
    value_as_number: Optional[float] = None
    value_as_string: Optional[str] = None
    value_as_concept_id: Optional[NonNegativeInt] = None
    qualifier_concept_id: Optional[NonNegativeInt] = None
    unit_concept_id: Optional[NonNegativeInt] = None
    provider_id: Optional[NonNegativeInt] = None
    visit_occurrence_id: Optional[NonNegativeInt] = None
    visit_detail_id: Optional[NonNegativeInt] = None
    observation_source_value: Optional[str] = None
    observation_source_concept_id: Optional[NonNegativeInt] = None
    unit_source_value: Optional[str] = None
    qualifier_source_value: Optional[str] = None
    value_source_value: Optional[str] = None
    observation_event_id: Optional[int] = None
    obs_event_field_concept_id: Optional[NonNegativeInt] = None
