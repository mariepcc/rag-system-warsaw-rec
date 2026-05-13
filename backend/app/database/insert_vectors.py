from datetime import datetime
import pandas as pd
import psycopg2
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from config.settings import settings

vec = VectorStore()

df = pd.read_csv("data/warsaw_places.csv", sep=",")
df = df.where(pd.notnull(df), None)


def build_content(row) -> str:
    parts = []

    name = row.get("name", "")
    sub_cat = row.get("sub_category") or ""
    main_cat = row.get("main_category") or ""
    district = row.get("district") or ""
    category_str = f"{sub_cat} ({main_cat})" if sub_cat else main_cat
    parts.append(f"{name} to {category_str} w dzielnicy {district}.")

    summary = row.get("editorial_summary") or ""
    if summary:
        parts.append(summary)

    reviews = row.get("all_reviews") or ""
    if reviews:
        parts.append(f"Opinie: {reviews}")

    return " ".join(parts)


def prepare_record(row):
    content = build_content(row)
    content = " ".join(content.split())
    embedding = vec.get_embedding(content)

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "place_id": row.get("place_id"),
                "name": row.get("name"),
                "address": row.get("address"),
                "address_context": row.get("address_context"),
                "website": row.get("website"),
                "maps_url": row.get("maps_url"),
                "google_maps_direct_link": row.get("google_maps_direct_link"),
                "menu_url": row.get("menu_url"),
                "district": row.get("district"),
                "main_category": row.get("main_category"),
                "sub_category": row.get("sub_category"),
                "primary_type": row.get("primary_type"),
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "rating": row.get("rating"),
                "user_rating_count": row.get("user_rating_count"),
                "price_level": row.get("price_level"),
                "price_range_start": row.get("price_range_start"),
                "price_range_end": row.get("price_range_end"),
                "opening_hours": row.get("godziny_json"),
                "takeout": row.get("takeout"),
                "dine_in": row.get("dine_in"),
                "reservable": row.get("reservable"),
                "serves_breakfast": row.get("serves_breakfast"),
                "serves_lunch": row.get("serves_lunch"),
                "serves_dinner": row.get("serves_dinner"),
                "serves_dessert": row.get("serves_dessert"),
                "serves_coffee": row.get("serves_coffee"),
                "serves_vegetarian": row.get("serves_vegetarian"),
                "serves_wine": row.get("serves_wine"),
                "serves_beer": row.get("serves_beer"),
                "serves_cocktails": row.get("serves_cocktails"),
                "outdoor_seating": row.get("outdoor_seating"),
                "good_for_groups": row.get("good_for_groups"),
                "live_music": row.get("live_music"),
                "menu_for_children": row.get("menu_for_children"),
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


CREATE_PLACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS places (
    place_id              TEXT PRIMARY KEY,
    name                  TEXT,
    address               TEXT,
    address_context       TEXT,
    website               TEXT,
    maps_url              TEXT,
    google_maps_direct_link TEXT,
    menu_url              TEXT,
    district              TEXT,
    main_category         TEXT,
    sub_category          TEXT,
    primary_type          TEXT,
    lat                   DOUBLE PRECISION,
    lon                   DOUBLE PRECISION,
    rating                DOUBLE PRECISION,
    user_rating_count     INTEGER,
    price_level           TEXT,
    price_range_start     DOUBLE PRECISION,
    price_range_end       DOUBLE PRECISION,
    opening_hours         JSONB,
    takeout               BOOLEAN,
    dine_in               BOOLEAN,
    reservable            BOOLEAN,
    serves_breakfast      BOOLEAN,
    serves_lunch          BOOLEAN,
    serves_dinner         BOOLEAN,
    serves_dessert        BOOLEAN,
    serves_coffee         BOOLEAN,
    serves_vegetarian     BOOLEAN,
    serves_wine           BOOLEAN,
    serves_beer           BOOLEAN,
    serves_cocktails      BOOLEAN,
    outdoor_seating       BOOLEAN,
    good_for_groups       BOOLEAN,
    live_music            BOOLEAN,
    menu_for_children     BOOLEAN
);
"""

INSERT_PLACE_SQL = """
INSERT INTO places (
    place_id, name, address, address_context, website, maps_url,
    google_maps_direct_link, menu_url, district, main_category, sub_category,
    primary_type, lat, lon, rating, user_rating_count, price_level,
    price_range_start, price_range_end, opening_hours, takeout, dine_in,
    reservable, serves_breakfast, serves_lunch, serves_dinner, serves_dessert,
    serves_coffee, serves_vegetarian, serves_wine, serves_beer, serves_cocktails,
    outdoor_seating, good_for_groups, live_music, menu_for_children
) VALUES (
    %(place_id)s, %(name)s, %(address)s, %(address_context)s, %(website)s,
    %(maps_url)s, %(google_maps_direct_link)s, %(menu_url)s, %(district)s,
    %(main_category)s, %(sub_category)s, %(primary_type)s, %(lat)s, %(lon)s,
    %(rating)s, %(user_rating_count)s, %(price_level)s, %(price_range_start)s,
    %(price_range_end)s, %(opening_hours)s, %(takeout)s, %(dine_in)s,
    %(reservable)s, %(serves_breakfast)s, %(serves_lunch)s, %(serves_dinner)s,
    %(serves_dessert)s, %(serves_coffee)s, %(serves_vegetarian)s, %(serves_wine)s,
    %(serves_beer)s, %(serves_cocktails)s, %(outdoor_seating)s,
    %(good_for_groups)s, %(live_music)s, %(menu_for_children)s
)
ON CONFLICT (place_id) DO NOTHING;
"""


def _to_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        return bool(val)
    return str(val).strip().lower() in ("true", "1", "yes")


def insert_places(dataframe: pd.DataFrame) -> None:
    import json

    conn = psycopg2.connect(settings.postgres_connection_string)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_PLACES_TABLE_SQL)

                for _, row in dataframe.iterrows():
                    opening_hours_raw = row.get("godziny_json")
                    if isinstance(opening_hours_raw, str):
                        try:
                            opening_hours = json.loads(opening_hours_raw)
                        except (json.JSONDecodeError, ValueError):
                            opening_hours = opening_hours_raw
                    else:
                        opening_hours = opening_hours_raw

                    cur.execute(
                        INSERT_PLACE_SQL,
                        {
                            "place_id": row.get("place_id"),
                            "name": row.get("name"),
                            "address": row.get("address"),
                            "address_context": row.get("address_context"),
                            "website": row.get("website"),
                            "maps_url": row.get("maps_url"),
                            "google_maps_direct_link": row.get(
                                "google_maps_direct_link"
                            ),
                            "menu_url": row.get("menu_url"),
                            "district": row.get("district"),
                            "main_category": row.get("main_category"),
                            "sub_category": row.get("sub_category"),
                            "primary_type": row.get("primary_type"),
                            "lat": row.get("lat"),
                            "lon": row.get("lon"),
                            "rating": row.get("rating"),
                            "user_rating_count": row.get("user_rating_count"),
                            "price_level": row.get("price_level"),
                            "price_range_start": row.get("price_range_start"),
                            "price_range_end": row.get("price_range_end"),
                            "opening_hours": psycopg2.extras.Json(opening_hours)
                            if opening_hours is not None
                            else None,
                            "takeout": _to_bool(row.get("takeout")),
                            "dine_in": _to_bool(row.get("dine_in")),
                            "reservable": _to_bool(row.get("reservable")),
                            "serves_breakfast": _to_bool(row.get("serves_breakfast")),
                            "serves_lunch": _to_bool(row.get("serves_lunch")),
                            "serves_dinner": _to_bool(row.get("serves_dinner")),
                            "serves_dessert": _to_bool(row.get("serves_dessert")),
                            "serves_coffee": _to_bool(row.get("serves_coffee")),
                            "serves_vegetarian": _to_bool(row.get("serves_vegetarian")),
                            "serves_wine": _to_bool(row.get("serves_wine")),
                            "serves_beer": _to_bool(row.get("serves_beer")),
                            "serves_cocktails": _to_bool(row.get("serves_cocktails")),
                            "outdoor_seating": _to_bool(row.get("outdoor_seating")),
                            "good_for_groups": _to_bool(row.get("good_for_groups")),
                            "live_music": _to_bool(row.get("live_music")),
                            "menu_for_children": _to_bool(row.get("menu_for_children")),
                        },
                    )
    finally:
        conn.close()


records_df = df.apply(prepare_record, axis=1)

vec.create_tables()
vec.create_index()
vec.upsert(records_df)
# vec.delete(delete_all=True)

insert_places(df)
