import os
import json
import time
import logging
import requests
import pandas as pd
from shapely.geometry import shape, Point
from tqdm import tqdm


API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY environment variable")

GEOJSON_PATH = "data/warsaw_districts.geojson"
OUTPUT_PATH = "data/warsaw_places.csv"

SEARCH_CENTERS = [
    (52.2297, 21.0122, "centrum"),
    (52.2700, 20.9800, "wola-zoliborz"),
    (52.2700, 21.0600, "praga-polnoc"),
    (52.1900, 21.0600, "praga-poludnie"),
    (52.1600, 21.0200, "mokotow-ursynow"),
    (52.2000, 20.9400, "ochota-wlochy"),
    (52.3000, 21.0500, "bialoleka"),
    (52.2100, 21.1200, "rembertow-wawer"),
]
SEARCH_RADIUS = 8000

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


def load_districts(geojson_path: str) -> list[dict]:
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    districts = []
    for feature in data["features"]:
        name = (
            feature["properties"].get("name")
            or feature["properties"].get("nazwa")
            or feature["properties"].get("NAZWA")
            or feature["properties"].get("district")
            or "Unknown"
        )
        geom = shape(feature["geometry"])
        districts.append({"name": name, "geometry": geom})

    logger.info("Loaded %d districts from %s", len(districts), geojson_path)
    return districts


def assign_district(lat: float, lon: float, districts: list[dict]) -> str | None:
    point = Point(lon, lat)
    for d in districts:
        if d["geometry"].contains(point):
            return d["name"]
    return None


def fetch_places_for_center(
    lat: float,
    lon: float,
    radius: int,
    included_type: str,
    api_key: str,
) -> list[dict]:
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
        logger.warning("HTTP error for type %s: %s", included_type, e)
        return []
    except Exception as e:
        logger.warning("Error for type %s: %s", included_type, e)
        return []


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
        "district": None,
        "menu_url": None,
        "main_category": None,
        "sub_category": None,
    }


def main():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(GEOJSON_PATH):
        raise FileNotFoundError(
            f"Missing file {GEOJSON_PATH}.\n"
            "Download Warsaw district boundaries from:\n"
            "https://github.com/andilabs/warszawa-dzielnice-geojson\n"
            "and save as data/warsaw_districts.geojson"
        )
    districts = load_districts(GEOJSON_PATH)

    all_places: dict[str, dict] = {}

    total_calls = len(SEARCH_CENTERS) * len(INCLUDED_TYPES)
    with tqdm(total=total_calls, desc="Fetching places") as pbar:
        for lat, lon, label in SEARCH_CENTERS:
            for ptype in INCLUDED_TYPES:
                pbar.set_postfix({"center": label, "type": ptype})
                places = fetch_places_for_center(
                    lat, lon, SEARCH_RADIUS, ptype, API_KEY
                )
                for p in places:
                    pid = p.get("id")
                    if pid and pid not in all_places:
                        all_places[pid] = parse_place(p)
                pbar.update(1)
                time.sleep(0.05)

    logger.info("Fetched %d unique places before filtering", len(all_places))

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
        "After filtering: %d places in Warsaw, %d outside (%d total)",
        len(records),
        outside_warsaw,
        len(all_places),
    )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.info(
        "Saved to %s (%d rows, %d columns)", OUTPUT_PATH, len(df), len(df.columns)
    )

    print("\n--- Summary ---")
    print(f"Total places: {len(df)}")
    print(f"Districts:\n{df['district'].value_counts().to_string()}")
    print(
        f"\nTypes (primary_type):\n{df['primary_type'].value_counts().head(20).to_string()}"
    )


if __name__ == "__main__":
    main()
