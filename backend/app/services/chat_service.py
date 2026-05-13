import pandas as pd
import io
from schemas.chat import SynthesizedResponse
from config.settings import get_settings
from database.vector_store import VectorStore
from database.chat_repository import ChatRepository
from database.places_repository import PlacesRepository
from services.message_classifier import MessageClassifier
from services.metadata_extractor import MetadataExtractor
from services.synthesizer import Synthesizer


class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.vec = VectorStore()
        self.chat_repo = ChatRepository(self.settings.database.service_url)
        self.places_repo = PlacesRepository(self.settings.database.service_url)

    def handle_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
    ) -> SynthesizedResponse:
        self.chat_repo.create_session(user_id, session_id)
        history = self.chat_repo.get_history(session_id)
        classification = MessageClassifier.classify(message, history)

        extraction = None

        if classification.message_type in ("rag", "hybrid"):
            query = classification.reformulated_query or message
            print(f"Reformulated query for retrieval: {query}")

            extraction = MetadataExtractor.extract(query)
            predicates = MetadataExtractor.build_predicates(extraction)
            print(f"Cleaned query: {extraction.clean_query}")
            expanded_query = MetadataExtractor.expand_query_with_hyde(
                extraction.clean_query
            )
            print(f"Extracted predicates: {predicates}")
            print(f"Expanded query for retrieval: {expanded_query}")

            buffer = max(5, extraction.results_limit * 2)
            search_kwargs = {"limit": extraction.results_limit + buffer}
            if predicates is not None:
                search_kwargs["predicates"] = predicates

            context = self.vec.search(expanded_query, **search_kwargs)

            context = MetadataExtractor.filter_by_opening_hours(context, extraction)
            available = len(context) if context is not None else 0
            limit = min(max(extraction.results_limit, min(available, 5)), 5)

        elif classification.message_type == "followup":
            context_json = self.chat_repo.get_last_rag_context(session_id)
            if context_json:
                context = pd.read_json(io.StringIO(context_json), orient="records")
                limit = len(context)

        print(f"Number of results to search for: {limit}")

        response = Synthesizer.generate_response(
            question=message,
            chat_history=history,
            context=context,
            results_limit=limit,
            message_type=classification.message_type,
        )

        recommended_places = []
        if response.recommended_place_names and context is not None:
            recommended = set(response.recommended_place_names)
            id_col = "place_id" if "place_id" in context.columns else "id"
            place_ids = context[context["name"].isin(recommended)][id_col].tolist()
            recommended_places = self.places_repo.get_places_by_ids(place_ids)

        response.recommended_places = recommended_places
        response._context = context

        self.chat_repo.save_message(
            session_id, "user", message, message_type=classification.message_type
        )
        self.chat_repo.save_message(
            session_id,
            "assistant",
            response.answer,
            message_type=classification.message_type,
            recommended_places=[p.id for p in recommended_places],
        )
        return response
