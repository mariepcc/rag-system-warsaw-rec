import { useState, useCallback, useRef } from "react";
import { toggleFavourite } from "@/api/places";
import { useFavouritesStore } from "@/store/favouritesStore";
import type { Place } from "@/api/places";

export function useFavourite(place: Place, sessionId?: string) {
  const { favouriteNames, toggle: storeToggle } = useFavouritesStore();
  const isFav = favouriteNames.has(place.name);
  const [loading, setLoading] = useState(false);
  const placeRef = useRef(place);
  placeRef.current = place;

  const toggle = useCallback(async () => {
    storeToggle(place.name);
    setLoading(true);
    try {
      await toggleFavourite(placeRef.current, sessionId);
    } catch {
      storeToggle(place.name);
    } finally {
      setLoading(false);
    }
  }, [place.name, storeToggle, sessionId]);

  return { isFav, loading, toggle };
}
