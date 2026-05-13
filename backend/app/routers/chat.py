from fastapi import APIRouter, Depends, HTTPException
import psycopg2
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
from dependencies import get_current_user
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService
from database.chat_repository import ChatRepository
from config.settings import get_settings
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()
chat_repo = ChatRepository(settings.database.service_url)
chat_service = ChatService()


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = chat_service.handle_message(
            user_id=user["user_id"],
            session_id=session_id,
            message=request.message,
        )
        return ChatResponse(
            answer=response.answer,
            type=response.type,
            session_id=session_id,
            places=response.recommended_places,
            enough_context=response.enough_context,
        )
    except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as e:
        raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in send_message: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception:
        logger.error("send_message error", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")


@router.get("/all-names")
async def get_all_names(user: dict = Depends(get_current_user)):
    try:
        names = chat_repo.get_all_recommended_names(user["user_id"])
        return {"names": names}
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in get_all_names: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception:
        logger.error("get_all_names error", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")


@router.get("/search-history")
async def search_history(q: str, user: dict = Depends(get_current_user)):
    try:
        results = chat_repo.search_sessions(user_id=user["user_id"], search_query=q)
        return results
    except psycopg2.OperationalError as e:
        logger.error(f"DB unavailable in search_history: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception:
        logger.error("search_history error", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")
