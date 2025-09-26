from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Measurement(BaseModel):
    measurement_id: PositiveInt
    person_id: PositiveInt
    measurement_concept_id: NonNegativeInt
    measurement_date: date
    measurement_datetime: Optional[datetime] = None
    measurement_time: Optional[str] = None
    measurement_type_concept_id: Optional[NonNegativeInt] = 0
    operator_concept_id: Optional[NonNegativeInt] = None
    value_as_number: Optional[float] = None
    value_as_concept_id: Optional[NonNegativeInt] = None
    unit_concept_id: Optional[NonNegativeInt] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    provider_id: Optional[NonNegativeInt] = None
    visit_occurrence_id: Optional[NonNegativeInt] = None
    visit_detail_id: Optional[NonNegativeInt] = None
    measurement_source_value: Optional[str] = None
    measurement_source_concept_id: Optional[NonNegativeInt] = None
    unit_source_value: Optional[str] = None
    unit_source_concept_id: Optional[NonNegativeInt] = None
    value_source_value: Optional[str] = None
    measurement_event_id: Optional[int] = None
    meas_event_field_concept_id: Optional[NonNegativeInt] = None
