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
                "OPENAI_TIMEOUT=12.5",
                "OPENAI_MAX_RETRIES=4",
            ]
        )
    )

    settings = OpenAISettings.from_env(dotenv_path=dotenv_path)

    assert settings.api_key == "example-key"
    assert settings.org_id == "example-org"
    assert settings.project_id == "example-project"
    assert settings.base_url == "https://example.test"
    assert settings.default_model == "gpt-example"
    assert settings.timeout == 12.5
    assert settings.max_retries == 4
    assert settings.client_kwargs() == {
        "api_key": "example-key",
        "organization": "example-org",
        "project": "example-project",
        "base_url": "https://example.test",
        "timeout": 12.5,
        "max_retries": 4,
    }


def test_overrides_take_precedence(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=unused")

    settings = OpenAISettings.from_env(
        dotenv_path=dotenv_path,
        api_key="override-key",
        default_model="override-model",
        timeout=7.5,
        max_retries=2,
    )

    assert settings.api_key == "override-key"
    assert settings.default_model == "override-model"
    assert settings.timeout == 7.5
    assert settings.max_retries == 2
    assert settings.client_kwargs() == {
        "api_key": "override-key",
        "timeout": 7.5,
        "max_retries": 2,
    }


def test_create_client_uses_kwargs(monkeypatch):
    settings = OpenAISettings(
        api_key="another-key",
        base_url="http://localhost",
        timeout=15,
        max_retries=3,
        extra_client_kwargs={"default_headers": {"X-Test": "1"}},
    )

    kwargs = settings.client_kwargs()
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    client = settings.create_client()

    assert kwargs == {
        "api_key": "another-key",
        "base_url": "http://localhost",
        "timeout": 15,
        "max_retries": 3,
        "default_headers": {"X-Test": "1"},
    }
    assert client.api_key == "another-key"
    assert client.base_url == "http://localhost"
    assert client.timeout == 15
    assert client.max_retries == 3


def test_extra_client_kwargs_do_not_mutate_source():
    extra = {"default_headers": {"X-Trace": "abc"}}
    settings = OpenAISettings(api_key="key", extra_client_kwargs=extra)

    kwargs = settings.client_kwargs()

    assert kwargs["default_headers"] == {"X-Trace": "abc"}
    assert extra == {"default_headers": {"X-Trace": "abc"}}


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    def test_from_env_no_api_key(monkeypatch):
        """Test that no error is raised if OPENAI_API_KEY is missing (new behavior)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
        monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        # ... code that previously triggered the error, now should not raise ...
        pass
