from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class ConditionOccurrence(BaseModel):
    condition_occurrence_id: PositiveInt
    person_id: PositiveInt
    condition_concept_id: NonNegativeInt
    condition_start_date: date
    condition_start_datetime: Optional[datetime] = None
    condition_end_date: Optional[date] = None
    condition_end_datetime: Optional[datetime] = None
    condition_type_concept_id: Optional[NonNegativeInt] = 0
    condition_status_concept_id: Optional[NonNegativeInt] = 0
    stop_reason: Optional[str] = None
    provider_id: Optional[NonNegativeInt] = None
    visit_occurrence_id: Optional[NonNegativeInt] = None
    visit_detail_id: Optional[NonNegativeInt] = None
    condition_source_value: Optional[str] = None
    condition_source_concept_id: Optional[NonNegativeInt] = None
    condition_status_source_value: Optional[str] = None
