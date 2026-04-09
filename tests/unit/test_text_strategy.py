"""Phase 1 Testing Gate — test_text_strategy.py

Tests the ThinkingStrategy ABC, TextStrategy (V1 Anthropic API), and
LatentStrategy (V2 stub).

No real API key or network calls are made.  The Anthropic AsyncAnthropic
client is mocked via unittest.mock.AsyncMock at the module-level binding
in libucks.thinking.text_strategy, following the same safe-patching
pattern established in test_embedding_service.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libucks.thinking.base import Representation, ThinkingStrategy
from libucks.thinking.latent_strategy import LatentStrategy  # still tested via ABC checks
from libucks.thinking.text_strategy import TextStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str = "mocked response") -> MagicMock:
    """Return a mock AsyncAnthropic client whose messages.create returns
    a response whose first content block has .text == response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=message)
    return mock_client


# ---------------------------------------------------------------------------
# ThinkingStrategy ABC
# ---------------------------------------------------------------------------

class TestThinkingStrategyABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ThinkingStrategy()  # type: ignore[abstract]

    def test_text_strategy_is_subclass(self):
        assert issubclass(TextStrategy, ThinkingStrategy)

    def test_latent_strategy_is_subclass(self):
        assert issubclass(LatentStrategy, ThinkingStrategy)

    def test_representation_type_alias_exists(self):
        # Representation must be importable and be a type or alias
        assert Representation is not None


# ---------------------------------------------------------------------------
# TextStrategy — encode()
# ---------------------------------------------------------------------------

class TestTextStrategyEncode:
    async def test_encode_returns_text_unchanged(self):
        svc = TextStrategy(client=_make_mock_client())
        result = await svc.encode("hello world")
        assert result == "hello world"

    async def test_encode_empty_string(self):
        svc = TextStrategy(client=_make_mock_client())
        result = await svc.encode("")
        assert result == ""

    async def test_encode_multiline_text(self):
        text = "line one\nline two\nline three"
        svc = TextStrategy(client=_make_mock_client())
        result = await svc.encode(text)
        assert result == text

    async def test_encode_does_not_call_api(self):
        mock_client = _make_mock_client()
        svc = TextStrategy(client=mock_client)
        await svc.encode("no API call expected")
        mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# TextStrategy — reason()
# ---------------------------------------------------------------------------

class TestTextStrategyReason:
    async def test_reason_returns_api_response_text(self):
        mock_client = _make_mock_client("The auth middleware validates JWTs.")
        svc = TextStrategy(client=mock_client)
        result = await svc.reason("how does auth work?", "JWT context here")
        assert result == "The auth middleware validates JWTs."

    async def test_reason_calls_messages_create_once(self):
        mock_client = _make_mock_client()
        svc = TextStrategy(client=mock_client)
        await svc.reason("query", "context")
        mock_client.messages.create.assert_called_once()

    async def test_reason_result_is_str(self):
        mock_client = _make_mock_client("some string response")
        svc = TextStrategy(client=mock_client)
        result = await svc.reason("q", "ctx")
        assert isinstance(result, str)

    async def test_reason_prompt_includes_query(self):
        mock_client = _make_mock_client()
        svc = TextStrategy(client=mock_client)
        await svc.reason("unique_query_marker", "some context")
        call_kwargs = mock_client.messages.create.call_args
        # The prompt passed to the API must contain the query
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        prompt_text = str(messages)
        assert "unique_query_marker" in prompt_text

    async def test_reason_prompt_includes_context(self):
        mock_client = _make_mock_client()
        svc = TextStrategy(client=mock_client)
        await svc.reason("query", "unique_context_marker")
        call_kwargs = mock_client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        prompt_text = str(messages)
        assert "unique_context_marker" in prompt_text

    async def test_reason_empty_context_does_not_raise(self):
        mock_client = _make_mock_client("response")
        svc = TextStrategy(client=mock_client)
        result = await svc.reason("query", "")
        assert isinstance(result, str)

    async def test_reason_returns_representation(self):
        """result must be usable as a Representation (str in V1)."""
        mock_client = _make_mock_client("a valid representation")
        svc = TextStrategy(client=mock_client)
        result = await svc.reason("q", "ctx")
        # In V1, Representation is str
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TextStrategy — decode()
# ---------------------------------------------------------------------------

class TestTextStrategyDecode:
    async def test_decode_returns_str_unchanged(self):
        svc = TextStrategy(client=_make_mock_client())
        result = await svc.decode("already a string")
        assert result == "already a string"

    async def test_decode_empty_string(self):
        svc = TextStrategy(client=_make_mock_client())
        assert await svc.decode("") == ""

    async def test_decode_does_not_call_api(self):
        mock_client = _make_mock_client()
        svc = TextStrategy(client=mock_client)
        await svc.decode("some result")
        mock_client.messages.create.assert_not_called()

    async def test_decode_returns_str_type(self):
        svc = TextStrategy(client=_make_mock_client())
        result = await svc.decode("text")
        assert type(result) is str


# ---------------------------------------------------------------------------
# TextStrategy — construction
# ---------------------------------------------------------------------------

class TestTextStrategyConstruction:
    def test_can_construct_with_injected_client(self):
        svc = TextStrategy(client=_make_mock_client())
        assert isinstance(svc, TextStrategy)

    def test_default_model_is_set(self):
        svc = TextStrategy(client=_make_mock_client())
        assert svc.model is not None
        assert isinstance(svc.model, str)
        assert len(svc.model) > 0

    def test_custom_model_is_stored(self):
        svc = TextStrategy(client=_make_mock_client(), model="claude-3-5-sonnet-20241022")
        assert svc.model == "claude-3-5-sonnet-20241022"

    def test_build_from_env_uses_anthropic_client(self):
        """TextStrategy.from_env() constructs an AsyncAnthropic client.
        We patch the constructor in the module namespace so no real key is needed."""
        import libucks.thinking.text_strategy as ts_module

        with patch.object(ts_module, "AsyncAnthropic") as MockClient:
            MockClient.return_value = _make_mock_client()
            svc = TextStrategy.from_env()
            MockClient.assert_called_once()
        assert isinstance(svc, TextStrategy)

