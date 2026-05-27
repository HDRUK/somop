from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, PositiveInt, NonNegativeInt


class Location(BaseModel):
    location_id: PositiveInt
    address_1: Optional[str] = None
    address_2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    county: Optional[str] = None
    location_source_value: Optional[str] = None
    country_concept_id: Optional[NonNegativeInt] = None
    country_source_value: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
