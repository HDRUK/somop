from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class ProcedureOccurrence(BaseModel):
    procedure_occurrence_id: PositiveInt
    person_id: PositiveInt
    procedure_concept_id: NonNegativeInt
    procedure_date: date
    procedure_datetime: Optional[datetime] = None
    procedure_end_date: Optional[date] = None
    procedure_end_datetime: Optional[datetime] = None
    procedure_type_concept_id: NonNegativeInt
    modifier_concept_id: Optional[NonNegativeInt] = None
    quantity: Optional[int] = None
    provider_id: Optional[NonNegativeInt] = None
    visit_occurrence_id: Optional[NonNegativeInt] = None
    visit_detail_id: Optional[NonNegativeInt] = None
    procedure_source_value: Optional[str] = None
    procedure_source_concept_id: Optional[NonNegativeInt] = None
    modifier_source_value: Optional[str] = None
