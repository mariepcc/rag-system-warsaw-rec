from typing import Any, Dict, List, Type
import instructor
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel
from config.settings import get_settings
import logging
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)


class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        client_initializers = {
            "openai": lambda s: instructor.from_openai(OpenAI(api_key=s.api_key)),
            "anthropic": lambda s: instructor.from_anthropic(
                Anthropic(api_key=s.api_key)
            ),
            "llama": lambda s: instructor.from_openai(
                OpenAI(base_url=s.base_url, api_key=s.api_key),
                mode=instructor.Mode.JSON,
            ),
        }

        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_completion_tokens": kwargs.get(
                "max_completion_tokens", self.settings.max_completion_tokens
            ),
            "response_model": response_model,
            "messages": messages,
        }
        try:
            return self.client.chat.completions.create(**completion_params)
        except APITimeoutError as e:
            logger.error(f"OpenAI timeout: {e}", exc_info=True)
            raise
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit: {e}", exc_info=True)
            raise
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}", exc_info=True)
            raise
        except APIStatusError as e:
            logger.error(f"OpenAI API error {e.status_code}: {e}", exc_info=True)
            raise
