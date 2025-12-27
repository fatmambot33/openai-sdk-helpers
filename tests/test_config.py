import pytest

from openai_sdk_helpers.config import OpenAISettings


def test_from_env_loads_dotenv(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=example-key",
                "OPENAI_ORG_ID=example-org",
                "OPENAI_PROJECT_ID=example-project",
                "OPENAI_BASE_URL=https://example.test",
                "OPENAI_MODEL=gpt-example",
            ]
        )
    )

    settings = OpenAISettings.from_env(dotenv_path=dotenv_path)

    assert settings.api_key == "example-key"
    assert settings.org_id == "example-org"
    assert settings.project_id == "example-project"
    assert settings.base_url == "https://example.test"
    assert settings.default_model == "gpt-example"
    assert settings.client_kwargs() == {
        "api_key": "example-key",
        "organization": "example-org",
        "project": "example-project",
        "base_url": "https://example.test",
    }


def test_overrides_take_precedence(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=unused")

    settings = OpenAISettings.from_env(
        dotenv_path=dotenv_path,
        api_key="override-key",
        default_model="override-model",
    )

    assert settings.api_key == "override-key"
    assert settings.default_model == "override-model"
    assert settings.client_kwargs() == {"api_key": "override-key"}


def test_create_client_uses_kwargs(monkeypatch):
    settings = OpenAISettings(api_key="another-key", base_url="http://localhost")

    kwargs = settings.client_kwargs()
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    client = settings.create_client()

    assert kwargs == {"api_key": "another-key", "base_url": "http://localhost"}
    assert client.api_key == "another-key"
    assert client.base_url == "http://localhost"


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        OpenAISettings.from_env()
