import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

WEAVEL_DEFAULT_API_URL = "https://api.weavel.ai"


def get_last_user_message(messages: List[BaseMessage]) -> str | None:
    # find last HumanMessage
    messages = list(reversed(messages))
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content)
    return None


class WeavelCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for Weavel.

    Args:
        user_id (str): User ID. You should handle this from your system.
        trace_id (str): Trace(Conversation) ID. You should handle this from your system.
        user_message_id (str, Optional): User message ID.
            If you want to add some metadata to the user message, you can use this.
        user_message_metadata (Dict[str, Any], Optional): User message metadata.
            If you want to add some metadata to the user message, you can use this.
        assistant_message_id (str, Optional): Assistant message ID.
            If you want to add some metadata to the assistant message, you can use this.
        assistant_message_metadata (Dict[str, Any], Optional):
            Assistant message metadata.
            If you want to add some metadata to the assistant message, you can use this.
    """

    def __init__(
        self,
        user_id: str,
        trace_id: str,
        user_message_id: Optional[str] = None,
        user_message_metadata: Optional[Dict[str, Any]] = None,
        assistant_message_id: Optional[str] = None,
        assistant_message_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.user_id = user_id
        self.trace_id = trace_id
        self.user_message_id = user_message_id
        self.user_message_metadata = user_message_metadata
        self.assistant_message_id = assistant_message_id
        self.assistant_message_metadata = assistant_message_metadata
        if not user_id or not trace_id:
            raise ValueError("user_id and trace_id must be passed in kwargs")

        self.weavel_api_key = os.getenv("WEAVEL_API_KEY")
        self.saved_user_messages: List[str] = []
        self.saved_assistant_messages: List[str] = []

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            self.user_message: str | None = get_last_user_message(messages[0])
            # uses_id and trace_id must be passed in kwargs
            if self.user_message is None:
                return
            if self.user_message in self.saved_user_messages:
                return

            requests.post(
                f"{WEAVEL_DEFAULT_API_URL}/capture/trace_data",
                headers={"Authorization": f"Bearer {self.weavel_api_key}"},
                json={
                    "user_id": self.user_id,
                    "trace_id": self.trace_id,
                    "role": "user",
                    "content": self.user_message,
                    "trace_data_id": self.user_message_id,
                    "metadata": self.user_message_metadata,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.saved_user_messages.append(self.user_message)

        except Exception as e:
            logger.error(f"[Weavel] An error occurred in on_chat_model_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            for generation in response.generations:
                if generation[0].text in self.saved_assistant_messages:
                    return

                requests.post(
                    f"{WEAVEL_DEFAULT_API_URL}/capture/trace_data",
                    headers={"Authorization": f"Bearer {self.weavel_api_key}"},
                    json={
                        "user_id": self.user_id,
                        "trace_id": self.trace_id,
                        "role": "assistant",
                        "content": generation[0].text,
                        "trace_data_id": self.assistant_message_id,
                        "metadata": self.assistant_message_metadata,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                self.saved_assistant_messages.append(generation[0].text)

        except Exception as e:
            logger.error(f"[Weavel] An error occurred in on_llm_end: {e}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self.tool_name = serialized.get("name", None)
            self.tool_inputs = inputs

        except Exception as e:
            logger.error(f"[Weavel] An error occurred in on_tool_start: {e}")

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if self.tool_name is None:
                raise ValueError("name must be passed in serialized")

            properties = {"inputs": self.tool_inputs, "output": output}

            requests.post(
                f"{WEAVEL_DEFAULT_API_URL}/capture/track_event",
                headers={"Authorization": f"Bearer {self.weavel_api_key}"},
                json={
                    "user_id": self.user_id,
                    "trace_id": self.trace_id,
                    "name": self.tool_name,
                    "properties": properties,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"[Weavel] An error occurred in on_tool_end: {e}")
