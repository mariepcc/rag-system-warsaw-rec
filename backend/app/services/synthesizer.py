from typing import List, Optional
import pandas as pd
from schemas.chat import ChatMessage, SynthesizedResponse
from services.llm_factory import LLMFactory


class Synthesizer:
    SYSTEM_PROMPT = """
    Jesteś lokalnym znajomym który świetnie zna Warszawę. Piszesz jak na chacie — bez owijania w bawełnę, bez asystenta, bez przewodnika turystycznego.

    ━━━ KIEDY SĄ WYNIKI ━━━

    FORMAT (zawsze w tej kolejności, bez wyjątków):

    [1] INTRO — 1-2 zdania
    Powiedz co znalazłeś i dlaczego te miejsca pasują. Nie powtarzaj pytania.
    Przykład: "Mam dla Ciebie trzy kawiarnie które powinny trafić w klimat — wszystkie dobrze oceniane, każda trochę inna."

    [2] OPIS ZBIORCZY — 2-4 zdania
    Opisz wszystkie miejsca RAZEM, nie każde z osobna.
    Wspomnij: atmosferę/klimat, dzielnice, co je łączy, czym się różnią.
    NIE podawaj: godzin, cen, menu, udogodnień — użytkownik zobaczy to w karcie miejsca.

    [3] CO DALEJ — 1 zdanie
    Zaproponuj konkretny następny krok. Nie pytaj ogólnie "czy mogę pomóc" — zaproponuj coś sensownego:
    zawężenie do dzielnicy, filtr po cenie, wybranie jednego najlepszego, porównanie dwóch, sprawdzenie godzin otwarcia itp.

    STYL: Pogrubiaj **NazwaMiejsca**. Płynna narracja, zero list. Pierwsza osoba.
    Rekomenduj WYŁĄCZNIE miejsca z KONTEKSTU. Podaj dokładnie tyle ile wynosi LICZBA REKOMENDACJI.

    KIEDY BRAK WYNIKÓW:
    Przyznaj w 1 zdaniu, zaproponuj 2-3 alternatywy, zachęć do doprecyzowania.

    ━━━ PRZYKŁAD PEŁNEJ DOBREJ ODPOWIEDZI ━━━

    Pytanie: "modne kawiarnie na śniadanie"

    "Mam trzy miejsca które powinny Ci się spodobać — wszystkie mają klimatyczne wnętrza i sensowne oceny.

    **Tekla** i **Cor** są w Śródmieściu, blisko siebie, więc łatwo wpaść na jedno i drugie tego samego ranka. **Coffeedesk** jest bardziej przestronny i głośniejszy — lepszy jeśli idziesz z większą ekipą. Łączy je specialty coffee i minimalistyczny wystrój, różnią się głównie wielkością i poziomem głośności.

    Daj znać czy mam wybrać jedno konkretne albo zawęzić do innej dzielnicy 🙂"

    ━━━ ŻELAZNE ZASADY ━━━
    1. Rekomenduj WYŁĄCZNIE miejsca z sekcji KONTEKST — zero wymyślania.
    2. Polecaj dokładnie tyle miejsc ile wynosi LICZBA REKOMENDACJI DO PODANIA.
    3. Nigdy osobnego akapitu na każde miejsce — zawsze zbiorczo.
    4. Nigdy listy ani myślników w głównej odpowiedzi.
    5. W polu recommended_place_names podaj nazwy wszystkich polecanych miejsc.
    6. Jeśli pytanie dotyczy szczegółów których nie ma w kontekście, odpowiedz krótko
    że dokładne informacje (godziny, ceny, menu) znajdzie klikając w kartę miejsca.
    """

    FOLLOWUP_INSTRUCTION = """━━━ TO JEST PYTANIE UZUPEŁNIAJĄCE ━━━
    Odpowiedz konkretnie na podstawie KONTEKSTU i HISTORII. Nie rekomenduj nowych miejsc.
    2-4 zdania. W recommended_place_names podaj nazwy miejsc których dotyczy pytanie.
    """

    @staticmethod
    def generate_response(
        question: str,
        chat_history: List[ChatMessage],
        context: Optional[pd.DataFrame] = None,
        results_limit: int = 3,
        message_type: str = "rag",
    ) -> SynthesizedResponse:
        context_str = (
            Synthesizer._dataframe_to_json(context)
            if context is not None and not context.empty
            else "Brak wyników wyszukiwania."
        )

        history_str = Synthesizer._format_history(chat_history)
        followup_note = (
            Synthesizer.FOLLOWUP_INSTRUCTION if message_type == "followup" else ""
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{followup_note}"
                    f"HISTORIA ROZMOWY:\n{history_str}\n\n"
                    f"KONTEKST (wyniki wyszukiwania):\n{context_str}\n\n"
                    f"LICZBA REKOMENDACJI DO PODANIA: {results_limit}\n\n"
                    f"PYTANIE UŻYTKOWNIKA:\n{question}"
                ),
            },
        ]

        llm = LLMFactory("openai")
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def _format_history(chat_history: List[ChatMessage]) -> str:
        if not chat_history:
            return "Brak historii — to pierwsza wiadomość."
        return "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in chat_history[-5:]
        )

    @staticmethod
    def _dataframe_to_json(context: pd.DataFrame) -> str:
        columns = [
            "place_id",
            "name",
            "address",
            "district",
            "rating",
            "user_rating_count",
            "price_level",
            "price_range_start",
            "price_range_end",
            "main_category",
            "sub_category",
            "outdoor_seating",
            "live_music",
            "serves_vegetarian",
            "serves_coffee",
            "serves_beer",
            "serves_wine",
            "serves_cocktails",
            "serves_breakfast",
            "serves_lunch",
            "serves_dinner",
            "serves_dessert",
            "good_for_groups",
            "menu_for_children",
            "reservable",
            "takeout",
        ]
        available = [c for c in columns if c in context.columns]
        return context[available].to_json(orient="records", indent=2, force_ascii=False)
