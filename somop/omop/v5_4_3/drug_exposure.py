from __future__ import annotations
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class DrugExposure(BaseModel):
    drug_exposure_id: PositiveInt
    person_id: PositiveInt
    drug_concept_id: NonNegativeInt
    drug_exposure_start_date: date
    drug_exposure_start_datetime: Optional[datetime] = None
    drug_exposure_end_date: Optional[date] = None
    drug_exposure_end_datetime: Optional[datetime] = None
    verbatim_end_date: Optional[date] = None
    drug_type_concept_id: Optional[NonNegativeInt] = 0
    stop_reason: Optional[str] = None
    refills: Optional[int] = None
    quantity: Optional[float] = None
    days_supply: Optional[int] = None
    sig: Optional[str] = None
    route_concept_id: Optional[NonNegativeInt] = None
    lot_number: Optional[str] = None
    provider_id: Optional[NonNegativeInt] = None
    visit_occurrence_id: Optional[NonNegativeInt] = None
    visit_detail_id: Optional[NonNegativeInt] = None
    drug_source_value: Optional[str] = None
    drug_source_concept_id: Optional[NonNegativeInt] = None
    route_source_value: Optional[str] = None
    dose_unit_source_value: Optional[str] = None
