import math
from pydantic import BaseModel, field_validator
from typing import Any, Optional
from datetime import datetime


class PlaceResponse(BaseModel):
    id: str
    name: str
    address: Optional[str] = None
    district: Optional[str] = None
    rating: Optional[float] = None
    user_rating_count: Optional[float] = None
    price_level: Optional[str] = None
    main_category: Optional[str] = None
    sub_category: Optional[str] = None
    opening_hours: Optional[str] = None
    website: Optional[str] = None
    maps_url: Optional[str] = None
    menu_url: Optional[str] = None
    serves_vegetarian: Optional[bool] = None
    outdoor_seating: Optional[bool] = None
    serves_coffee: Optional[bool] = None
    serves_breakfast: Optional[bool] = None
    serves_lunch: Optional[bool] = None
    serves_dinner: Optional[bool] = None
    serves_dessert: Optional[bool] = None
    serves_beer: Optional[bool] = None
    serves_wine: Optional[bool] = None
    serves_cocktails: Optional[bool] = None
    dine_in: Optional[bool] = None
    takeout: Optional[bool] = None
    reservable: Optional[bool] = None
    live_music: Optional[bool] = None
    good_for_groups: Optional[bool] = None
    menu_for_children: Optional[bool] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    price_range_start: Optional[float] = None
    price_range_end: Optional[float] = None
    google_maps_direct_link: Optional[str] = None
    editorial_summary: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def nan_to_none(cls, v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v


class SavedPlaceResponse(PlaceResponse):
    is_favourite: bool
    session_id: Optional[str] = None
    created_at: datetime


class ToggleFavouriteRequest(BaseModel):
    sessionId: Optional[str] = None
