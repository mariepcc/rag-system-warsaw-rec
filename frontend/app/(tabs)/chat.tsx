import { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";
import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";
import { TouchableOpacity } from "react-native";
import { usePlaceStore } from "@/store/placesStore";
import { useGradientCycle } from "@/hooks/useGradientCycle";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { sendMessage } from "@/api/chat";
import { getSessions, getSessionMessages, deleteSession } from "@/api/sessions";
import type { Session } from "@/api/sessions";
import { Place } from "@/api/places";
import { MessageItem } from "@/components/chat/MessageItem";
import { TypingIndicator } from "@/components/chat/TypingIndicator";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatHistory } from "@/components/chat/ChatHistory";
import uuid from "react-native-uuid";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  places?: Place[];
  isFollowup?: boolean;
};

const ACCENT = "#dcc3c3";

function groupSessionsByDate(sessions: Session[]) {
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const todaySessions: Session[] = [];
  const yesterdaySessions: Session[] = [];
  const olderSessions: Session[] = [];

  sessions.forEach((s) => {
    const date = new Date(s.created_at);
    if (date.toDateString() === today.toDateString()) todaySessions.push(s);
    else if (date.toDateString() === yesterday.toDateString())
      yesterdaySessions.push(s);
    else olderSessions.push(s);
  });

  return [
    ...(todaySessions.length > 0
      ? [{ title: "Dzisiaj", sessions: todaySessions }]
      : []),
    ...(yesterdaySessions.length > 0
      ? [{ title: "Wczoraj", sessions: yesterdaySessions }]
      : []),
    ...(olderSessions.length > 0
      ? [{ title: "Wcześniej", sessions: olderSessions }]
      : []),
  ];
}

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { colors, advance } = useGradientCycle();

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMsgIndex, setLoadingMsgIndex] = useState(0);
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [historyVisible, setHistoryVisible] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(false);

  const flatListRef = useRef<FlatList<Message>>(null);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const lastScrollY = useRef(0);

  useEffect(() => {
    const showEvent =
      Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent =
      Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";
    const show = Keyboard.addListener(showEvent, () => setKeyboardOpen(true));
    const hide = Keyboard.addListener(hideEvent, () => setKeyboardOpen(false));
    return () => {
      show.remove();
      hide.remove();
    };
  }, []);

  useEffect(() => {
    if (!loading) return;
    setLoadingMsgIndex(0);
    const interval = setInterval(() => {
      setLoadingMsgIndex((i) => (i + 1) % 4);
    }, 4000);
    return () => clearInterval(interval);
  }, [loading]);

  function handleScroll(e: any) {
    const y = e.nativeEvent.contentOffset.y;
    if (Math.abs(y - lastScrollY.current) > 150) {
      lastScrollY.current = y;
      advance();
    }
  }

  async function openHistory() {
    setHistoryVisible(true);
    setSessionsLoading(true);
    try {
      const data = await getSessions();
      setSessions(data);
    } catch {
    } finally {
      setSessionsLoading(false);
    }
  }

  async function loadSession(session: Session) {
    setHistoryVisible(false);
    setSessionId(session.id);
    sessionIdRef.current = session.id;
    setMessages([]);
    setSessionLoading(true);
    try {
      const msgs = await getSessionMessages(session.id);
      setMessages(
        msgs.reverse().map((m: any) => ({
          id: uuid.v4() as string,
          role: m.role,
          content: m.content,
          places: m.places ?? [],
          isFollowup: m.type === "followup",
        })),
      );
    } catch {
    } finally {
      setSessionLoading(false);
    }
  }

  async function handleDelete(sessionId: string) {
    try {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (sessionIdRef.current === sessionId) {
        startNewChat();
      }
    } catch {}
  }

  function startNewChat() {
    setHistoryVisible(false);
    setMessages([]);
    setSessionId(undefined);
    sessionIdRef.current = undefined;
    setInput("");
  }

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    Keyboard.dismiss();
    setMessages((prev) => [
      { id: uuid.v4() as string, role: "user", content: text },
      ...prev,
    ]);
    setInput("");
    setLoading(true);

    try {
      const response = await sendMessage(text, sessionIdRef.current);
      console.log("Chat response:", response.places);
      setSessionId(response.session_id);
      sessionIdRef.current = response.session_id;
      setMessages((prev) => [
        {
          id: uuid.v4() as string,
          role: "assistant",
          content: response.answer,
          places: response.places ?? [],
          isFollowup: response.type === "followup",
        },
        ...prev,
      ]);
    } catch (error: any) {
      console.log("Chat error:", error?.code, error?.message);

      let content = "Coś poszło nie tak, spróbuj ponownie.";

      if (error?.code === "ECONNABORTED") {
        content = "Odpowiedź trwa zbyt długo. Spróbuj ponownie.";
      } else if (error?.message === "Network Error") {
        content = "Brak połączenia z internetem.";
      }

      setMessages((prev) => [
        {
          id: uuid.v4() as string,
          role: "assistant",
          content,
        },
        ...prev,
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  const handlePlacePress = useCallback(
    (place: Place) => {
      usePlaceStore.getState().setSelectedPlace(place, "chat");
      router.push({
        pathname: "/(tabs)/place/[name]",
        params: { name: encodeURIComponent(place.name) },
      } as any);
    },
    [router],
  );

  const renderItem = useCallback(
    ({ item }: { item: Message }) => (
      <MessageItem
        item={item}
        onPlacePress={handlePlacePress}
        sessionId={sessionId}
      />
    ),
    [handlePlacePress, sessionId],
  );

  const inputPaddingBottom = keyboardOpen
    ? 12
    : Math.max(insets.bottom, 16) + 70;

  return (
    <View style={s.container}>
      <LinearGradient
        colors={colors}
        style={StyleSheet.absoluteFill}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      />

      {historyVisible ? (
        <ChatHistory
          groups={groupSessionsByDate(sessions)}
          loading={sessionsLoading}
          paddingTop={insets.top}
          paddingBottom={insets.bottom + 16}
          onBack={() => setHistoryVisible(false)}
          onNewChat={startNewChat}
          onSelectSession={loadSession}
          onDeleteSession={handleDelete}
          headerHeight={insets.top + 70}
        />
      ) : (
        <>
          <BlurView
            intensity={60}
            tint="light"
            style={[
              s.headerCommon,
              s.headerAbsolute,
              {
                paddingTop: insets.top,
                backgroundColor: "rgba(255,255,255,0.4)",
              },
            ]}
          >
            <View style={s.headerInner}>
              <Text style={s.headerTitle}>SpotGuide</Text>
              <TouchableOpacity style={s.historyBtn} onPress={openHistory}>
                <Text style={s.historyBtnText}>☰</Text>
              </TouchableOpacity>
            </View>
          </BlurView>

          <KeyboardAvoidingView
            style={s.flex}
            behavior={Platform.OS === "ios" ? "padding" : undefined}
          >
            {sessionLoading ? (
              <View style={s.sessionLoadingContainer}>
                <ActivityIndicator size="large" color={ACCENT} />
              </View>
            ) : (
              <FlatList
                ref={flatListRef}
                data={messages}
                renderItem={renderItem}
                keyExtractor={(item) => item.id}
                inverted
                contentContainerStyle={[
                  s.list,
                  { paddingBottom: insets.top + 90 },
                ]}
                onScroll={handleScroll}
                scrollEventThrottle={16}
                automaticallyAdjustKeyboardInsets={true}
                ListEmptyComponent={
                  <View style={s.empty}>
                    <Text style={s.emptyStars}>✨</Text>
                    <Text style={s.emptyTitle}>Hej 👋</Text>
                    <Text style={s.emptySubtitle}>
                      Na co masz dzisiaj ochotę?
                    </Text>
                    <Text style={s.emptyInstruction}>
                      Znajdę dla Ciebie idealną miejscówkę.{"\n"}Pytaj śmiało.
                    </Text>
                  </View>
                }
              />
            )}

            {loading && <TypingIndicator index={loadingMsgIndex} />}

            <ChatInput
              value={input}
              onChange={setInput}
              onSend={handleSend}
              disabled={loading}
              paddingBottom={inputPaddingBottom}
            />
          </KeyboardAvoidingView>
        </>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  flex: { flex: 1 },
  container: { flex: 1, backgroundColor: "transparent" },
  headerCommon: {
    paddingHorizontal: 20,
    zIndex: 100,
    borderBottomWidth: 0.5,
    borderBottomColor: "rgba(0,0,0,0.05)",
  },
  headerAbsolute: { position: "absolute", top: 0, left: 0, right: 0 },
  headerInner: {
    height: 70,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  headerTitle: { fontSize: 22, fontFamily: "DMSans_700Bold", color: "#1a1a1a" },
  historyBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,0,0,0.05)",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(0,0,0,0.05)",
  },
  historyBtnText: { color: "#1a1a1a", fontSize: 20 },
  list: { paddingHorizontal: 16, flexGrow: 1 },
  sessionLoadingContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 40,
    marginTop: -40,
  },
  emptyStars: { fontSize: 40, marginBottom: 20 },
  emptyTitle: {
    fontSize: 26,
    fontFamily: "DMSans_700Bold",
    color: "#1a1a1a",
    textAlign: "center",
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 22,
    fontFamily: "DMSans_700Bold",
    color: "#1a1a1a",
    textAlign: "center",
    marginBottom: 20,
  },
  emptyInstruction: {
    fontSize: 16,
    color: "#444",
    textAlign: "center",
    lineHeight: 22,
  },
});
