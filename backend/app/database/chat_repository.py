import json
from typing import List
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from schemas.chat import ChatMessage
import logging

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id          TEXT NOT NULL REFERENCES chat_sessions(id),
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    message_type        TEXT CHECK (message_type IN ('rag', 'followup', 'hybrid')),
    recommended_places  TEXT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON chat_messages(session_id, created_at DESC);
    
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_chat_messages_content_trgm 
ON chat_messages USING GIN (content gin_trgm_ops);
"""


class ChatRepository:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_tables()

    def _get_conn(self):
        return psycopg2.connect(self.connection_string)

    def _init_tables(self) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)

    def create_session(self, user_id: str, session_id: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (session_id, user_id),
                )

    def get_sessions(self, user_id: str) -> List[dict]:
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT 
                            s.id,
                            s.created_at,
                            (
                                SELECT content 
                                FROM chat_messages 
                                WHERE session_id = s.id 
                                AND role = 'user'
                                ORDER BY created_at ASC 
                                LIMIT 1
                            ) as first_message,
                            (
                                SELECT COUNT(*) 
                                FROM chat_messages 
                                WHERE session_id = s.id
                            ) as message_count
                        FROM chat_sessions s
                        WHERE s.user_id = %s::text
                        ORDER BY s.created_at DESC
                        """,
                        (user_id,),
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
        except psycopg2.OperationalError as e:
            logger.error(f"DB connection error in get_sessions: {e}", exc_info=True)
            raise

    def delete_session(self, session_id: str, user_id: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM chat_sessions
                    WHERE id = %s AND user_id = %s
                    """,
                    (session_id, user_id),
                )
                if cur.fetchone() is None:
                    raise PermissionError("Brak dostępu do sesji")

                cur.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                cur.execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE id = %s AND user_id = %s
                    """,
                    (session_id, user_id),
                )
            conn.commit()

    def save_message(
        self, session_id, role, content, message_type=None, recommended_places=None
    ):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, message_type, recommended_places)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        message_type,
                        recommended_places or None,
                    ),
                )

    def get_history(self, session_id: str, limit: int = 20) -> List[ChatMessage]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT role, content, message_type, recommended_places
                    FROM chat_messages
                    WHERE session_id = %s::text
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()
        return [
            ChatMessage(
                role=r["role"],
                content=r["content"],
                message_type=r.get("message_type"),
                recommended_places=r.get("recommended_places") or [],
            )
            for r in rows
        ]

    def get_last_rag_context(self, session_id: str) -> str | None:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT p.*
                    FROM chat_messages m
                    CROSS JOIN LATERAL unnest(m.recommended_places) AS place_id
                    JOIN places p ON p.id = place_id
                    WHERE m.session_id = %s
                    AND m.role = 'assistant'
                    AND m.message_type IN ('rag', 'hybrid')
                    AND m.recommended_places IS NOT NULL
                    ORDER BY m.created_at DESC
                    LIMIT 5
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        return json.dumps([dict(r) for r in rows])

    def get_all_recommended_names(self, user_id: str) -> List[str]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT p.name
                    FROM chat_sessions s
                    JOIN chat_messages m ON m.session_id = s.id
                    CROSS JOIN LATERAL unnest(m.recommended_places) AS place_id
                    JOIN places p ON p.id = place_id
                    WHERE s.user_id = %s
                    AND m.role = 'assistant'
                    AND m.recommended_places IS NOT NULL
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows] if rows else []

    def search_sessions(self, user_id: str, search_query: str) -> List[dict]:
        safe_query = re.sub(r"[^\w\s\-]", "", search_query).strip()[:100]
        like_query = f"%{safe_query}%"

        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT s.id,
                        s.created_at,
                        (
                            SELECT content 
                            FROM chat_messages 
                            WHERE session_id = s.id AND role = 'user'
                            ORDER BY created_at ASC 
                            LIMIT 1
                        ) as first_message
                    FROM chat_sessions s
                    WHERE s.user_id = %s
                    AND EXISTS (
                        SELECT 1
                        FROM chat_messages m
                        LEFT JOIN LATERAL unnest(m.recommended_places) AS place_id ON true
                        LEFT JOIN places p ON p.id = place_id
                        WHERE m.session_id = s.id
                        AND (
                            m.content ILIKE %s
                            OR p.name ILIKE %s
                        )
                    )
                    ORDER BY s.created_at DESC
                    """,
                    (user_id, like_query, like_query),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def session_belongs_to_user(self, session_id: str, user_id: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 1 FROM chat_sessions
                    WHERE id = %s AND user_id = %s
                    """,
                    (session_id, user_id),
                )
                return cur.fetchone() is not None
