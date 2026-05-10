import { memo, useMemo } from "react";
import { View, Text, ScrollView, StyleSheet } from "react-native";
import Markdown from "react-native-markdown-display";
import { PlaceCard } from "@/components/chat/PlaceCard";
import { Place } from "@/api/places";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  places?: Place[];
  isFollowup?: boolean;
};

export const MessageItem = memo(
  ({
    item,
    onPlacePress,
  }: {
    item: Message;
    onPlacePress: (place: Place) => void;
  }) => {
    const isUser = item.role === "user";
    const places = useMemo(() => item.places ?? [], [item.id]);
    const showPlaces = !isUser && !item.isFollowup && places.length > 0;

    return (
      <View style={[s.row, isUser ? s.rowUser : s.rowAssistant]}>
        <View style={[s.col, isUser && { alignItems: "flex-end" }]}>
          <View style={isUser ? s.bubbleUser : s.assistantTextContainer}>
            {isUser ? (
              <Text style={[s.text, s.textUser]}>{item.content}</Text>
            ) : (
              <Markdown style={markdownStyle}>{item.content}</Markdown>
            )}
          </View>
          {showPlaces && (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              style={{ marginTop: 16 }}
              contentContainerStyle={{ paddingRight: 16 }}
            >
              {places.map((place) => (
                <PlaceCard
                  key={place.id}
                  place={place}
                  onPress={() => onPlacePress(place)}
                />
              ))}
            </ScrollView>
          )}
        </View>
      </View>
    );
  },
  (prev, next) =>
    prev.item.id === next.item.id && prev.item.content === next.item.content,
);

const markdownStyle = {
  body: {
    fontSize: 14,
    lineHeight: 24,
    color: "#3f3b3b",
    fontWeight: "400" as const,
  },
  strong: { fontWeight: "bold" as const },
  paragraph: { marginTop: 0, marginBottom: 8 },
};

const s = StyleSheet.create({
  row: { marginBottom: 24 },
  rowUser: { alignItems: "flex-end" },
  rowAssistant: { alignItems: "flex-start" },
  col: { flex: 1 },
  bubbleUser: {
    backgroundColor: "rgba(228, 150, 119, 0.36)",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    alignSelf: "flex-end",
    maxWidth: "85%",
  },
  assistantTextContainer: {
    alignSelf: "flex-start",
    maxWidth: "100%",
    paddingHorizontal: 4,
  },
  text: { fontSize: 14, lineHeight: 24, color: "#3f3b3b" },
  textUser: { fontWeight: "600" },
  textAssistant: { fontWeight: "400" },
});
