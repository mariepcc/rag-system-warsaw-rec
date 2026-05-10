import json
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor
from schemas.places import SavedPlaceResponse

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS saved_places (
    id            TEXT,
    user_id       TEXT NOT NULL,
    session_id    TEXT,
    name          TEXT NOT NULL,
    address       TEXT,
    district      TEXT,
    rating        FLOAT,
    user_rating_count FLOAT,
    price_level   TEXT,
    website       TEXT,
    maps_url      TEXT,
    menu_url      TEXT,
    main_category TEXT,
    sub_category  TEXT,
    editorial_summary TEXT,
    is_favourite   BOOLEAN DEFAULT FALSE,
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
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
                cur.execute(CREATE_TABLE_SQL)

    def get_all_places(self) -> List[dict]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON ((metadata->>'name'))
                        id::text as id,
                        contents,
                        metadata->>'name'           as name,
                        metadata->>'address'        as address,
                        metadata->>'district'       as district,
                        metadata->>'main_category'  as main_category,
                        metadata->>'sub_category'   as sub_category,
                        (metadata->>'lat')::float   as lat,
                        (metadata->>'lon')::float   as lon,
                        (metadata->>'rating')::float as rating,
                        (metadata->>'user_rating_count')::float as user_rating_count,
                        metadata->>'price_level'    as price_level,
                        metadata->>'maps_url'       as maps_url,
                        metadata->>'menu_url'       as menu_url,
                        metadata->>'google_maps_direct_link' as google_maps_direct_link,
                        metadata->>'opening_hours'  as opening_hours,
                        metadata->>'price_range_start' as price_range_start,
                        metadata->>'price_range_end'   as price_range_end,
                        (metadata->>'serves_vegetarian')::boolean as serves_vegetarian,
                        (metadata->>'serves_coffee')::boolean     as serves_coffee,
                        (metadata->>'serves_beer')::boolean       as serves_beer,
                        (metadata->>'serves_wine')::boolean       as serves_wine,
                        (metadata->>'serves_cocktails')::boolean  as serves_cocktails,
                        (metadata->>'serves_breakfast')::boolean  as serves_breakfast,
                        (metadata->>'serves_lunch')::boolean      as serves_lunch,
                        (metadata->>'serves_dinner')::boolean     as serves_dinner,
                        (metadata->>'serves_dessert')::boolean    as serves_dessert,
                        (metadata->>'outdoor_seating')::boolean   as outdoor_seating,
                        (metadata->>'live_music')::boolean        as live_music,
                        (metadata->>'good_for_groups')::boolean   as good_for_groups,
                        (metadata->>'menu_for_children')::boolean as menu_for_children,
                        (metadata->>'reservable')::boolean        as reservable,
                        (metadata->>'takeout')::boolean           as takeout,
                        (metadata->>'dine_in')::boolean           as dine_in
                    FROM embeddings
                    WHERE metadata->>'lat' IS NOT NULL
                    AND metadata->>'lon' IS NOT NULL
                    ORDER BY metadata->>'name'
                    """,
                )
                rows = cur.fetchall()
                results = []

                for row in rows:
                    place = dict(row)
                    place["editorial_summary"] = self._extract_editorial_summary(
                        place.get("contents", "")
                    )
                    results.append(place)

                return results

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

    def get_favourite_places(
        self,
        user_id: str,
    ) -> List[SavedPlaceResponse]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT 
                        id::text,
                        user_id,
                        session_id,
                        name,
                        address,
                        district,
                        rating,
                        user_rating_count,
                        price_level,
                        website,
                        maps_url,
                        menu_url,
                        main_category,
                        sub_category,
                        is_favourite,
                        editorial_summary,
                        created_at,
                        metadata->>'opening_hours' as opening_hours,
                        metadata->>'google_maps_direct_link' as google_maps_direct_link,
                        metadata->>'menu_url' as menu_url,
                        (metadata->>'lat')::float as lat,
                        (metadata->>'lon')::float as lon,
                        (metadata->>'price_range_start')::float as price_range_start,
                        (metadata->>'price_range_end')::float as price_range_end,
                        (metadata->>'serves_vegetarian')::boolean as serves_vegetarian,
                        (metadata->>'serves_coffee')::boolean as serves_coffee,
                        (metadata->>'serves_beer')::boolean as serves_beer,
                        (metadata->>'serves_wine')::boolean as serves_wine,
                        (metadata->>'serves_cocktails')::boolean as serves_cocktails,
                        (metadata->>'serves_breakfast')::boolean as serves_breakfast,
                        (metadata->>'serves_lunch')::boolean as serves_lunch,
                        (metadata->>'serves_dinner')::boolean as serves_dinner,
                        (metadata->>'serves_dessert')::boolean as serves_dessert,
                        (metadata->>'outdoor_seating')::boolean as outdoor_seating,
                        (metadata->>'live_music')::boolean as live_music,
                        (metadata->>'good_for_groups')::boolean as good_for_groups,
                        (metadata->>'menu_for_children')::boolean as menu_for_children,
                        (metadata->>'reservable')::boolean as reservable,
                        (metadata->>'dine_in')::boolean as dine_in,
                        (metadata->>'takeout')::boolean as takeout
                    FROM saved_places
                    WHERE user_id = %s AND is_favourite = TRUE
                """
                query += " ORDER BY created_at DESC"
                cur.execute(query, (user_id,))
                rows = cur.fetchall()
                return [dict(row) for row in rows]

    def toggle_favourite(self, user_id: str, place: dict) -> bool:
        place_id = place.get("id")
        if not place_id:
            raise ValueError(
                f"Place must have a Google Place ID, got: {place.get('place_id')}"
            )
        metadata = {
            "opening_hours": place.get("opening_hours"),
            "lat": place.get("lat"),
            "lon": place.get("lon"),
            "price_range_start": place.get("price_range_start"),
            "price_range_end": place.get("price_range_end"),
            "google_maps_direct_link": place.get("google_maps_direct_link"),
            "serves_vegetarian": place.get("serves_vegetarian"),
            "serves_coffee": place.get("serves_coffee"),
            "serves_beer": place.get("serves_beer"),
            "serves_wine": place.get("serves_wine"),
            "serves_cocktails": place.get("serves_cocktails"),
            "serves_breakfast": place.get("serves_breakfast"),
            "serves_lunch": place.get("serves_lunch"),
            "serves_dinner": place.get("serves_dinner"),
            "serves_dessert": place.get("serves_dessert"),
            "outdoor_seating": place.get("outdoor_seating"),
            "live_music": place.get("live_music"),
            "good_for_groups": place.get("good_for_groups"),
            "menu_for_children": place.get("menu_for_children"),
            "takeout": place.get("takeout"),
            "dine_in": place.get("dine_in"),
            "reservable": place.get("reservable"),
        }
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO saved_places (
                        id, user_id, name, address, district, rating,
                        user_rating_count, price_level, maps_url, menu_url,
                        main_category, sub_category, editorial_summary,
                        is_favourite, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, %s)
                    ON CONFLICT (id, user_id) DO UPDATE
                        SET is_favourite = NOT saved_places.is_favourite
                    RETURNING is_favourite
                    """,
                    (
                        place_id,
                        user_id,
                        place.get("name"),
                        place.get("address"),
                        place.get("district"),
                        place.get("rating"),
                        place.get("user_rating_count"),
                        place.get("price_level"),
                        place.get("maps_url"),
                        place.get("menu_url"),
                        place.get("main_category"),
                        place.get("sub_category"),
                        place.get("editorial_summary"),
                        json.dumps(metadata),
                    ),
                )
                row = cur.fetchone()
                return row[0]

    def get_favourite_names(self, user_id: str) -> list[str]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name FROM saved_places WHERE user_id = %s AND is_favourite = true",
                    (user_id,),
                )
                return [row[0] for row in cur.fetchall()]
