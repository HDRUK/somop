from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Specimen(BaseModel):
    specimen_id: PositiveInt
    person_id: PositiveInt
    specimen_concept_id: NonNegativeInt
    specimen_type_concept_id: NonNegativeInt = 0
    specimen_date: date
    specimen_datetime: Optional[datetime] = None
    quantity: Optional[float] = None
    unit_concept_id: Optional[NonNegativeInt] = None
    anatomic_site_concept_id: Optional[NonNegativeInt] = None
    disease_status_concept_id: Optional[NonNegativeInt] = None
    specimen_source_id: Optional[str] = None
    specimen_source_value: Optional[str] = None
    unit_source_value: Optional[str] = None
    anatomic_site_source_value: Optional[str] = None
    disease_status_source_value: Optional[str] = None
