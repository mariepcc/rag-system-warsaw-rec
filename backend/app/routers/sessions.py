from datetime import datetime, timezone
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
import psycopg2
from dependencies import get_current_user
from schemas.session import MessageResponse, SessionResponse
from database.chat_repository import ChatRepository
from database.places_repository import PlacesRepository
from config.settings import get_settings

router = APIRouter()
settings = get_settings()
chat_repo = ChatRepository(settings.database.service_url)
places_repo = PlacesRepository(settings.database.service_url)
logger = logging.getLogger(__name__)


@router.get("/", response_model=list[SessionResponse])
async def get_sessions(user: dict = Depends(get_current_user)):
    try:
        sessions = chat_repo.get_sessions(user["user_id"])
        return [
            SessionResponse(
                id=str(s["id"]),
                created_at=s["created_at"],
                first_message=s["first_message"],
                message_count=s["message_count"],
            )
            for s in sessions
        ]
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_session_messages: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"get_session_messages error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Wystąpił błąd serwera")


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    try:
        if not chat_repo.session_belongs_to_user(session_id, user["user_id"]):
            raise HTTPException(
                status_code=403, detail="Not authorized to access this session"
            )
        messages = chat_repo.get_history(session_id)
        return [
            MessageResponse(
                id=str(uuid.uuid4()),
                role=m.role,
                content=m.content,
                type=m.message_type,
                places=places_repo.get_places_by_ids(m.recommended_places or []),
                created_at=datetime.now(timezone.utc),
            )
            for m in messages
        ]
    except HTTPException:
        raise
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_session_messages: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"get_session_messages error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Wystąpił błąd serwera")


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    try:
        chat_repo.delete_session(session_id, user["user_id"])
        return {"success": True}
    except PermissionError:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this session"
        )
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in delete_session: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"delete_session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Wystąpił błąd serwera")
