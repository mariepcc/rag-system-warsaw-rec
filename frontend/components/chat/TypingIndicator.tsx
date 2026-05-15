import { useEffect, useRef, useState } from "react";
import { View, Text, StyleSheet, Animated } from "react-native";

const MESSAGES = [
  "Szukam najlepszych miejsc...",
  "Analizuję Twoje preferencje...",
  "Przeglądam lokalne perełki...",
  "Już prawie gotowe...",
];

export function TypingIndicator({ index }: { index: number }) {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(1)).current;
  const [displayedIndex, setDisplayedIndex] = useState(index);

  useEffect(() => {
    const animate = (dot: Animated.Value, delay: number) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(dot, {
            toValue: -6,
            duration: 300,
            useNativeDriver: true,
          }),
          Animated.timing(dot, {
            toValue: 0,
            duration: 300,
            useNativeDriver: true,
          }),
          Animated.delay(600),
        ]),
      );

    const a1 = animate(dot1, 0);
    const a2 = animate(dot2, 150);
    const a3 = animate(dot3, 300);
    a1.start();
    a2.start();
    a3.start();

    return () => {
      a1.stop();
      a2.stop();
      a3.stop();
    };
  }, []);

  useEffect(() => {
    Animated.sequence([
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start(() => setDisplayedIndex(index));
  }, [index]);

  return (
    <View style={s.wrap}>
      <View style={s.bubble}>
        <View style={s.dots}>
          {[dot1, dot2, dot3].map((dot, i) => (
            <Animated.View
              key={i}
              style={[s.dot, { transform: [{ translateY: dot }] }]}
            />
          ))}
        </View>
        <Animated.Text style={[s.text, { opacity: fadeAnim }]}>
          {MESSAGES[displayedIndex % MESSAGES.length]}
        </Animated.Text>
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    alignItems: "flex-start",
  },
  bubble: {
    backgroundColor: "rgba(255,255,255,0.7)",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    maxWidth: "85%",
    shadowColor: "#000",
    shadowOpacity: 0.06,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
  },
  dots: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "#E8622A",
    opacity: 0.8,
  },
  text: {
    fontSize: 14,
    color: "#555",
    fontStyle: "italic",
    flex: 1,
  },
});
