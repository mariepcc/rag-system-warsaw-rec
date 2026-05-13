from typing import List
import psycopg2
from psycopg2 import errors as pg_errors
from psycopg2.extras import RealDictCursor
from schemas.places import PlaceResponse, SavedPlaceResponse


CREATE_SAVED_PLACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS saved_places (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      TEXT NOT NULL,
    place_id     TEXT NOT NULL REFERENCES places(id),
    session_id   TEXT,
    is_favourite BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, place_id)
);
"""


class PlacesRepository:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._init_tables()

    def _get_conn(self):
        return psycopg2.connect(self.connection_string)

    def _init_tables(self):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_SAVED_PLACES_TABLE_SQL)

    def _extract_editorial_summary(self, content: str) -> str | None:
        if not content:
            return None
        try:
            before_reviews = content.split(". Opinie:")[0]
            sentences = before_reviews.split(". ", 1)
            if len(sentences) > 1:
                return sentences[1].strip() + "."
            return None
        except Exception:
            return None

    def get_all_places(self) -> List[dict]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        id, name, address, district, main_category, sub_category,
                        lat, lon, rating, user_rating_count, price_level,
                        maps_url, menu_url, google_maps_direct_link,
                        opening_hours, price_range_start, price_range_end,
                        editorial_summary,
                        serves_vegetarian, serves_coffee, serves_beer,
                        serves_wine, serves_cocktails, serves_breakfast,
                        serves_lunch, serves_dinner, serves_dessert,
                        outdoor_seating, live_music, good_for_groups,
                        menu_for_children, reservable, takeout, dine_in
                    FROM places
                    WHERE lat IS NOT NULL AND lon IS NOT NULL
                    ORDER BY name
                    """
                )
                return [dict(row) for row in cur.fetchall()]

    def get_favourite_places(self, user_id: str) -> List[SavedPlaceResponse]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        p.id,
                        sp.user_id,
                        sp.session_id,
                        p.name,
                        p.address,
                        p.district,
                        p.rating,
                        p.user_rating_count,
                        p.price_level,
                        p.website,
                        p.maps_url,
                        p.menu_url,
                        p.main_category,
                        p.sub_category,
                        p.editorial_summary,
                        sp.is_favourite,
                        sp.created_at,
                        p.opening_hours,
                        p.google_maps_direct_link,
                        p.lat,
                        p.lon,
                        p.price_range_start,
                        p.price_range_end,
                        p.serves_vegetarian,
                        p.serves_coffee,
                        p.serves_beer,
                        p.serves_wine,
                        p.serves_cocktails,
                        p.serves_breakfast,
                        p.serves_lunch,
                        p.serves_dinner,
                        p.serves_dessert,
                        p.outdoor_seating,
                        p.live_music,
                        p.good_for_groups,
                        p.menu_for_children,
                        p.reservable,
                        p.dine_in,
                        p.takeout
                    FROM saved_places sp
                    JOIN places p ON p.id = sp.place_id
                    WHERE sp.user_id = %s AND sp.is_favourite = TRUE
                    ORDER BY sp.created_at DESC
                    """,
                    (user_id,),
                )
                return [dict(row) for row in cur.fetchall()]

    def toggle_favourite(self, user_id: str, session_id: str, place_id: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO saved_places (user_id, place_id, session_id, is_favourite)
                        VALUES (%s, %s, %s, TRUE)
                        ON CONFLICT (user_id, place_id) DO UPDATE
                            SET is_favourite = NOT saved_places.is_favourite,
                                session_id   = EXCLUDED.session_id
                        RETURNING is_favourite
                        """,
                        (user_id, place_id, session_id),
                    )
                    row = cur.fetchone()
                    return row[0]
                except (pg_errors.DeadlockDetected, pg_errors.UniqueViolation):
                    conn.rollback()
                    raise ValueError("Concurrent modification — retry")
                except pg_errors.ForeignKeyViolation:
                    conn.rollback()
                    raise ValueError(f"Place {place_id} does not exist")

    def get_favourite_names(self, user_id: str) -> list[str]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.name
                    FROM saved_places sp
                    JOIN places p ON p.id = sp.place_id
                    WHERE sp.user_id = %s AND sp.is_favourite = TRUE
                    """,
                    (user_id,),
                )
                return [row[0] for row in cur.fetchall()]

    def get_places_by_ids(self, ids: list[str]) -> list[PlaceResponse]:
        if not ids:
            return []
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM places WHERE id = ANY(%s)",
                    (ids,),
                )
                return [PlaceResponse(**dict(row)) for row in cur.fetchall()]
