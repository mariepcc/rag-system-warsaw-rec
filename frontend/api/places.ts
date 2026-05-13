import apiClient from "./client";

export type Place = {
  id: string;
  name: string;
  address?: string;
  district?: string;
  rating?: number;
  user_rating_count?: number;
  price_level?: string;
  main_category?: string;
  sub_category?: string;
  opening_hours?: string;
  website?: string;
  maps_url?: string;
  menu_url?: string;
  serves_vegetarian?: boolean;
  outdoor_seating?: boolean;
  serves_coffee?: boolean;
  serves_breakfast?: boolean;
  serves_lunch?: boolean;
  serves_dinner?: boolean;
  serves_dessert?: boolean;
  serves_beer?: boolean;
  serves_wine?: boolean;
  serves_cocktails?: boolean;
  dine_in?: boolean;
  takeout?: boolean;
  reservable?: boolean;
  live_music?: boolean;
  good_for_groups?: boolean;
  menu_for_children?: boolean;
  lat?: number;
  lon?: number;
  price_range_start?: number;
  price_range_end?: number;
  google_maps_direct_link?: string;
  session_id?: string;
  saved_at?: string;
  is_favourite?: boolean;
  editorial_summary?: string;
};

export async function getAllPlaces(): Promise<Place[]> {
  const response = await apiClient.get("/places/all");
  return response.data;
}

export async function getFavouritePlaces(category?: string) {
  const response = await apiClient.get("/places/favourites", {
    params: category ? { category } : undefined,
  });
  return response.data;
}

export async function toggleFavourite(
  place: Place,
  sessionId?: string,
): Promise<boolean> {
  const response = await apiClient.post<{ is_favourite: boolean }>(
    `/places/favourites/${encodeURIComponent(place.id)}/toggle`,
    { sessionId },
  );
  return response.data.is_favourite;
}

export async function getFavouriteNames(): Promise<string[]> {
  const response = await apiClient.get<{ names: string[] }>(
    "/places/favourite-names",
  );
  return response.data.names;
}
