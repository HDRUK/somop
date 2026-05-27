from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Death(BaseModel):
    person_id: PositiveInt
    death_date: date
    death_datetime: Optional[datetime] = None
    death_type_concept_id: Optional[NonNegativeInt] = None
    cause_concept_id: Optional[NonNegativeInt] = None
    cause_source_value: Optional[str] = None
    cause_source_concept_id: Optional[NonNegativeInt] = None
