"""OpenCode Console gateway (OpenCode Go / pay-as-you-go) as an LLM provider.

Console serves every text model over Chat Completions, asks clients to identify
themselves with their own user agent, and needs a stable session ID per
conversation (required on a Go plan). DeepSeek IDs reached through the gateway
keep their reasoning_content round-trip.
"""

import pytest

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAIClient,
    OpenCodeChatOpenAI,
)
from tradingagents.llm_clients.validators import validate_model


@pytest.mark.unit
def test_registry_entry():
    spec = OPENAI_COMPATIBLE_PROVIDERS["opencode"]
    assert spec.base_url == "https://opencode.ai/inference/go/openai/v1"
    assert spec.base_url_env == "OPENCODE_BASE_URL"
    assert spec.chat_class is OpenCodeChatOpenAI
    assert get_api_key_env("opencode") == "OPENCODE_API_KEY"


@pytest.mark.unit
def test_factory_routes_to_openai_client():
    client = create_llm_client(provider="opencode", model="deepseek-v4.1-flash")
    assert isinstance(client, OpenAIClient)


@pytest.mark.unit
def test_chat_completions_against_go_endpoint(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    monkeypatch.delenv("OPENCODE_BASE_URL", raising=False)
    llm = create_llm_client(provider="opencode", model="deepseek-v4.1-flash").get_llm()

    assert isinstance(llm, OpenCodeChatOpenAI)
    assert str(llm.openai_api_base) == "https://opencode.ai/inference/go/openai/v1"
    assert getattr(llm, "use_responses_api", False) in (False, None)
    key = llm.openai_api_key.get_secret_value() if hasattr(
        llm.openai_api_key, "get_secret_value"
    ) else llm.openai_api_key
    assert key == "oc_sk_test"


@pytest.mark.unit
def test_headers_identify_client_and_session(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    llm = create_llm_client(provider="opencode", model="gpt-6-luna").get_llm()

    assert llm.default_headers["User-Agent"].startswith("tradingagents/")
    assert llm.default_headers["x-opencode-session"].startswith("ses_")


@pytest.mark.unit
def test_session_id_is_unique_per_client(monkeypatch):
    # One client instance backs a whole run/conversation, so the ID is stable
    # for that run; separate runs get separate IDs.
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    first = create_llm_client(provider="opencode", model="gpt-6-luna").get_llm()
    second = create_llm_client(provider="opencode", model="gpt-6-luna").get_llm()
    assert first.default_headers["x-opencode-session"] != second.default_headers["x-opencode-session"]
    assert first.default_headers["x-opencode-session"].startswith("ses_")


@pytest.mark.unit
def test_base_url_env_override_for_pay_as_you_go(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://opencode.ai/inference/openai/v1")
    llm = create_llm_client(provider="opencode", model="gpt-6-luna").get_llm()
    assert str(llm.openai_api_base) == "https://opencode.ai/inference/openai/v1"


@pytest.mark.unit
def test_deepseek_models_keep_reasoning_roundtrip():
    from tradingagents.llm_clients.capabilities import get_capabilities

    assert get_capabilities("deepseek-v4.1-flash").requires_reasoning_content_roundtrip
    assert get_capabilities("deepseek-v4-pro").requires_reasoning_content_roundtrip


@pytest.mark.unit
def test_any_model_accepted():
    assert validate_model("opencode", "some-new-go-model") is True


@pytest.mark.unit
def test_requires_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        create_llm_client(provider="opencode", model="gpt-6-luna").get_llm()
