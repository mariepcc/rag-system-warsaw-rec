from fastapi import APIRouter, Depends, HTTPException
import logging

import psycopg2
from schemas.places import SavedPlaceResponse, ToggleFavouriteRequest
from database.places_repository import PlacesRepository
from dependencies import get_current_user
from config.settings import get_settings

router = APIRouter()
settings = get_settings()
places_repo = PlacesRepository(settings.database.service_url)
logger = logging.getLogger(__name__)


@router.get("/all")
async def get_all_places(user: dict = Depends(get_current_user)):
    try:
        places = places_repo.get_all_places()
        return places
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_all_places: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"get_all_places error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")


@router.get("/favourites", response_model=list[SavedPlaceResponse])
async def get_saved_places(
    user: dict = Depends(get_current_user),
):
    try:
        places = places_repo.get_favourite_places(user["user_id"])
        return places
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_saved_places: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"get_saved_places error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")


@router.post("/favourites/{place_id}/toggle")
async def toggle_favourite(
    place_id: str,
    body: ToggleFavouriteRequest,
    user: dict = Depends(get_current_user),
):
    try:
        is_fav = places_repo.toggle_favourite(user["user_id"], body.sessionId, place_id)
        return {"is_favourite": is_fav}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in toggle_favourite: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"toggle_favourite error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")


@router.get("/favourite-names")
async def get_favourite_names(user: dict = Depends(get_current_user)):
    try:
        names = places_repo.get_favourite_names(user["user_id"])
        return {"names": names}
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_favourite_names: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"get_favourite_names error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")
