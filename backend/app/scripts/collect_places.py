"""
collect_places.py

Pobiera miejsca z Google Places API (New) dla obszaru Warszawy,
przypisuje dzielnicę na podstawie pliku GeoJSON z granicami,
dodaje puste pola main_category, sub_category, menu_url
i zapisuje wynik do data/warsaw_places.csv.

Wymagania:
    pip install requests shapely pandas tqdm

Użycie:
    GOOGLE_API_KEY=... python collect_places.py

Plik GeoJSON z granicami dzielnic Warszawy w data/warsaw_districts.geojson

"""

import os
import json
import time
import logging
import requests
import pandas as pd
from shapely.geometry import shape, Point
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Brak zmiennej środowiskowej GOOGLE_API_KEY")

GEOJSON_PATH = "data/warsaw_districts.geojson"
OUTPUT_PATH = "data/warsaw_places.csv"

# Centrum Warszawy i promień wyszukiwania (w metrach).
# 12 000 m to rozsądny promień — pokrywa większość dzielnic centralnych.
# Dla pełnego pokrycia miasta używamy siatki kilku punktów.
SEARCH_CENTERS = [
    # (lat, lon, label)
    (52.2297, 21.0122, "centrum"),
    (52.2700, 20.9800, "wola-zoliborz"),
    (52.2700, 21.0600, "praga-polnoc"),
    (52.1900, 21.0600, "praga-poludnie"),
    (52.1600, 21.0200, "mokotow-ursynow"),
    (52.2000, 20.9400, "ochota-wlochy"),
    (52.3000, 21.0500, "bialoleka"),
    (52.2100, 21.1200, "rembertow-wawer"),
]
SEARCH_RADIUS = 8000  # metry — zachodzi na siebie, duplikaty są odfiltrowane

# Typy miejsc do pobrania (Google Places API primary types)
INCLUDED_TYPES = [
    "restaurant",
    "cafe",
    "bar",
    "night_club",
    "bakery",
    "meal_takeaway",
    "meal_delivery",
    "museum",
    "art_gallery",
    "tourist_attraction",
    "park",
    "library",
    "zoo",
    "aquarium",
    "amusement_park",
    "bowling_alley",
    "casino",
    "movie_theater",
    "shopping_mall",
    "food",
]

# Pola do pobrania z API (field mask)
FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.shortFormattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceRange",
        "places.priceLevel",
        "places.primaryType",
        "places.types",
        "places.regularOpeningHours",
        "places.reviews",
        "places.editorialSummary",
        "places.googleMapsUri",
        "places.websiteUri",
        "places.servesVegetarianFood",
        "places.servesCoffee",
        "places.servesBeer",
        "places.servesWine",
        "places.servesCocktails",
        "places.servesBreakfast",
        "places.servesLunch",
        "places.servesDinner",
        "places.servesDessert",
        "places.outdoorSeating",
        "places.goodForGroups",
        "places.reservable",
        "places.takeout",
        "places.dineIn",
        "places.menuForChildren",
    ]
)

PLACES_API_URL = "https://places.googleapis.com/v1/places:searchNearby"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ładowanie granic dzielnic
# ---------------------------------------------------------------------------


def load_districts(geojson_path: str) -> list[dict]:
    """
    Zwraca listę słowników: {"name": str, "geometry": shapely shape}
    """
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    districts = []
    for feature in data["features"]:
        name = (
            feature["properties"].get("name")
            or feature["properties"].get("nazwa")
            or feature["properties"].get("NAZWA")
            or feature["properties"].get("district")
            or "Nieznana"
        )
        geom = shape(feature["geometry"])
        districts.append({"name": name, "geometry": geom})

    logger.info("Załadowano %d dzielnic z %s", len(districts), geojson_path)
    return districts


def assign_district(lat: float, lon: float, districts: list[dict]) -> str | None:
    """
    Przypisuje dzielnicę na podstawie współrzędnych geograficznych.
    Zwraca None jeśli punkt nie leży w żadnej dzielnicy (poza Warszawą).
    """
    point = Point(lon, lat)  # shapely: (x=lon, y=lat)
    for d in districts:
        if d["geometry"].contains(point):
            return d["name"]
    return None


# ---------------------------------------------------------------------------
# Pobieranie miejsc z Google Places API (New)
# ---------------------------------------------------------------------------


def fetch_places_for_center(
    lat: float,
    lon: float,
    radius: int,
    included_type: str,
    api_key: str,
) -> list[dict]:
    """
    Pobiera miejsca dla jednego centrum i jednego typu.
    API (New) zwraca max 20 wyników na request — nie ma paginacji w searchNearby.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    body = {
        "includedTypes": [included_type],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius),
            }
        },
        "languageCode": "pl",
    }

    try:
        response = requests.post(PLACES_API_URL, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("places", [])
    except requests.HTTPError as e:
        logger.warning("HTTP error dla typu %s: %s", included_type, e)
        return []
    except Exception as e:
        logger.warning("Błąd dla typu %s: %s", included_type, e)
        return []


# ---------------------------------------------------------------------------
# Parsowanie odpowiedzi API do płaskiego słownika
# ---------------------------------------------------------------------------


def parse_place(p: dict) -> dict:
    loc = p.get("location", {})
    price_range = p.get("priceRange", {})

    opening_hours_raw = p.get("regularOpeningHours", {}).get("weekdayDescriptions", [])

    reviews_raw = p.get("reviews", [])
    all_reviews = " | ".join(
        r.get("text", {}).get("text", "") for r in reviews_raw if r.get("text")
    )

    return {
        "place_id": p.get("id"),
        "name": p.get("displayName", {}).get("text"),
        "address": p.get("formattedAddress"),
        "lat": loc.get("latitude"),
        "lon": loc.get("longitude"),
        "address_context": p.get("shortFormattedAddress"),
        "rating": p.get("rating"),
        "user_rating_count": p.get("userRatingCount"),
        "price_range_start": price_range.get("startPrice", {}).get("units"),
        "price_range_end": price_range.get("endPrice", {}).get("units"),
        "price_level": p.get("priceLevel"),
        "primary_type": p.get("primaryType"),
        "types": ",".join(p.get("types", [])),
        "opening_hours": " | ".join(opening_hours_raw),
        "all_reviews": all_reviews,
        "editorial_summary": p.get("editorialSummary", {}).get("text"),
        "maps_url": p.get("googleMapsUri"),
        "website": p.get("websiteUri"),
        # Atrybuty boolowskie
        "serves_vegetarian": p.get("servesVegetarianFood"),
        "serves_coffee": p.get("servesCoffee"),
        "serves_beer": p.get("servesBeer"),
        "serves_wine": p.get("servesWine"),
        "serves_cocktails": p.get("servesCocktails"),
        "serves_breakfast": p.get("servesBreakfast"),
        "serves_lunch": p.get("servesLunch"),
        "serves_dinner": p.get("servesDinner"),
        "serves_dessert": p.get("servesDessert"),
        "outdoor_seating": p.get("outdoorSeating"),
        "good_for_groups": p.get("goodForGroups"),
        "reservable": p.get("reservable"),
        "takeout": p.get("takeout"),
        "dine_in": p.get("dineIn"),
        "menu_for_children": p.get("menuForChildren"),
        # Pola własne — uzupełniane w preprocessing.ipynb
        "district": None,
        "menu_url": None,
        "main_category": None,
        "sub_category": None,
    }


# ---------------------------------------------------------------------------
# Główna logika
# ---------------------------------------------------------------------------


def main():
    os.makedirs("data", exist_ok=True)

    # 1. Załaduj granice dzielnic
    if not os.path.exists(GEOJSON_PATH):
        raise FileNotFoundError(
            f"Brak pliku {GEOJSON_PATH}.\n"
            "Pobierz granice dzielnic Warszawy np. z:\n"
            "https://github.com/andilabs/warszawa-dzielnice-geojson\n"
            "i zapisz jako data/warsaw_districts.geojson"
        )
    districts = load_districts(GEOJSON_PATH)

    # 2. Zbierz miejsca
    all_places: dict[str, dict] = {}  # place_id -> parsed dict (deduplication)

    total_calls = len(SEARCH_CENTERS) * len(INCLUDED_TYPES)
    with tqdm(total=total_calls, desc="Pobieranie miejsc") as pbar:
        for lat, lon, label in SEARCH_CENTERS:
            for ptype in INCLUDED_TYPES:
                pbar.set_postfix({"centrum": label, "typ": ptype})
                places = fetch_places_for_center(
                    lat, lon, SEARCH_RADIUS, ptype, API_KEY
                )
                for p in places:
                    pid = p.get("id")
                    if pid and pid not in all_places:
                        all_places[pid] = parse_place(p)
                pbar.update(1)
                time.sleep(0.05)  # grzeczność wobec API

    logger.info("Pobrano %d unikalnych miejsc przed filtrowaniem", len(all_places))

    # 3. Przypisz dzielnice i odfiltruj miejsca poza Warszawą
    records = []
    outside_warsaw = 0
    for place in all_places.values():
        lat, lon = place["lat"], place["lon"]
        if lat is None or lon is None:
            outside_warsaw += 1
            continue
        district = assign_district(lat, lon, districts)
        if district is None:
            outside_warsaw += 1
            continue
        place["district"] = district
        records.append(place)

    logger.info(
        "Po filtrowaniu: %d miejsc w Warszawie, %d poza (%d łącznie)",
        len(records),
        outside_warsaw,
        len(all_places),
    )

    # 4. Zapisz do CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.info(
        "Zapisano do %s (%d wierszy, %d kolumn)", OUTPUT_PATH, len(df), len(df.columns)
    )

    # 5. Podsumowanie
    print("\n--- Podsumowanie ---")
    print(f"Łącznie miejsc: {len(df)}")
    print(f"Dzielnice:\n{df['district'].value_counts().to_string()}")
    print(
        f"\nTypy (primary_type):\n{df['primary_type'].value_counts().head(20).to_string()}"
    )


if __name__ == "__main__":
    main()
