import React, { useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Platform,
} from "react-native";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import { Place } from "@/api/places";
import { useFavourite } from "@/hooks/useFavourite";

const CATEGORY_THEMES: Record<
  string,
  { icon: keyof typeof Ionicons.glyphMap; color: string; bg: string }
> = {
  Gastronomia: { icon: "restaurant", color: "#E8622A", bg: "#FEF0EA" },
  "Kawa i Słodycze": { icon: "cafe", color: "#7C5CBF", bg: "#F2EEFB" },
  "Kultura & Rozrywka": {
    icon: "color-palette",
    color: "#2A8BE8",
    bg: "#EAF2FE",
  },
  "Życie Nocne": { icon: "moon", color: "#3D3D5C", bg: "#EEEEF5" },
  "Natura & Rekreacja": { icon: "leaf", color: "#2A9E6A", bg: "#EAF7F2" },
};

const TAG_CONFIG: Partial<
  Record<keyof Place, { icon: keyof typeof MaterialCommunityIcons.glyphMap }>
> = {
  takeout: { icon: "shopping-outline" },
  dine_in: { icon: "silverware-variant" },
  reservable: { icon: "calendar-check-outline" },
  outdoor_seating: { icon: "weather-sunny" },
  serves_breakfast: { icon: "egg-outline" },
  serves_lunch: { icon: "food-outline" },
  serves_dinner: { icon: "food-steak" },
  serves_dessert: { icon: "cupcake" },
  serves_beer: { icon: "beer-outline" },
  serves_wine: { icon: "glass-wine" },
  serves_cocktails: { icon: "glass-cocktail" },
  serves_coffee: { icon: "coffee-outline" },
  serves_vegetarian: { icon: "leaf" },
  good_for_groups: { icon: "account-group-outline" },
  live_music: { icon: "music-note-outline" },
  menu_for_children: { icon: "baby-face-outline" },
};

const TAG_COLOR = "#999";

function getActiveTags(place: Place) {
  return (Object.entries(TAG_CONFIG) as [keyof Place, { icon: any }][])
    .filter(([key]) => place[key] === true)
    .map(([, config]) => config)
    .slice(0, 5);
}

export function PlaceCard({
  place,
  onPress,
  sessionId,
}: {
  place: Place;
  onPress?: () => void;
  sessionId?: string;
}) {
  const { isFav, loading, toggle } = useFavourite(place, sessionId);
  const heartScale = useRef(new Animated.Value(1)).current;
  const activeTags = getActiveTags(place);

  const theme = CATEGORY_THEMES[place.main_category ?? ""] || {
    icon: "location",
    color: "#888",
    bg: "#F5F5F5",
  };

  function handleHeart() {
    Animated.sequence([
      Animated.spring(heartScale, {
        toValue: 1.4,
        useNativeDriver: true,
        speed: 50,
      }),
      Animated.spring(heartScale, {
        toValue: 1,
        useNativeDriver: true,
        speed: 20,
      }),
    ]).start();
    toggle();
  }

  return (
    <TouchableOpacity style={s.card} onPress={onPress} activeOpacity={0.9}>
      <View style={[s.imagePlaceholder, { backgroundColor: theme.bg }]}>
        <Ionicons name={theme.icon} size={36} color={theme.color} />

        {place.sub_category && (
          <View style={s.subBadge}>
            <Text style={s.subBadgeText}>{place.sub_category}</Text>
          </View>
        )}

        {!loading && (
          <TouchableOpacity style={s.heartBtn} onPress={handleHeart}>
            <Animated.View style={{ transform: [{ scale: heartScale }] }}>
              <Ionicons
                name={isFav ? "heart" : "heart-outline"}
                size={18}
                color={isFav ? "#E8622A" : "#888"}
              />
            </Animated.View>
          </TouchableOpacity>
        )}
      </View>

      <View style={s.body}>
        <Text style={s.name} numberOfLines={1}>
          {place.name}
        </Text>

        <View style={s.metaRow}>
          {place.rating != null && (
            <View style={s.ratingBox}>
              <Ionicons name="star" size={13} color="#F5A623" />
              <Text style={s.ratingText}>{place.rating.toFixed(1)}</Text>
              {place.user_rating_count != null && (
                <Text style={s.reviewCount}>({place.user_rating_count})</Text>
              )}
            </View>
          )}
          <Text style={s.district} numberOfLines={1}>
            • {place.district || "Warszawa"}
          </Text>
        </View>

        <View style={s.footer}>
          <View style={s.tags}>
            {activeTags.map((tag, i) => (
              <MaterialCommunityIcons
                key={i}
                name={tag.icon}
                size={16}
                color={TAG_COLOR}
              />
            ))}
          </View>
          <Ionicons
            name="chevron-forward-circle"
            size={20}
            color={theme.color}
          />
        </View>
      </View>
    </TouchableOpacity>
  );
}

const s = StyleSheet.create({
  card: {
    width: 200,
    borderRadius: 20,
    backgroundColor: "#fff",
    marginRight: 12,
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
      },
      android: { elevation: 3 },
    }),
    borderWidth: 1,
    borderColor: "#F0F0F0",
    overflow: "hidden",
  },
  imagePlaceholder: {
    height: 100,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  subBadge: {
    position: "absolute",
    bottom: 8,
    left: 8,
    backgroundColor: "rgba(255,255,255,0.92)",
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  subBadgeText: {
    color: "#444",
    fontSize: 9,
    fontWeight: "800",
    textTransform: "uppercase",
  },
  heartBtn: {
    position: "absolute",
    top: 8,
    right: 8,
    backgroundColor: "#fff",
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  body: {
    padding: 10,
  },
  name: {
    fontSize: 14,
    fontWeight: "700",
    color: "#1a1a1a",
    marginBottom: 4,
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    marginBottom: 8,
  },
  ratingBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 2,
  },
  ratingText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#1a1a1a",
  },
  reviewCount: {
    fontSize: 11,
    color: "#bbb",
  },
  district: {
    fontSize: 11,
    color: "#999",
    flex: 1,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    minHeight: 20,
  },
  tags: {
    flexDirection: "row",
    gap: 5,
    alignItems: "center",
  },
});
