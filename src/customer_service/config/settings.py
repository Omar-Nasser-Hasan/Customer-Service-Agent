"""Single environment boundary for current and planned application services."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(StrEnum):
    GOOGLE_GENAI = "google_genai"


class NodeModelConfig(BaseModel):
    provider: ModelProvider
    model: str
    temperature: float = Field(default=0, ge=0, le=2)


class Settings(BaseSettings):
    """Settings are loaded from the process environment or an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    persona_name: str = "Layla"
    store_name: str = "Souqly"
    app_environment: str = "development"
    frontend_origin: str = "http://localhost:3000"

    google_api_key: SecretStr | None = None
    node_models: dict[str, NodeModelConfig] = Field(
        default_factory=lambda: {
            "assistant": NodeModelConfig(
                provider=ModelProvider.GOOGLE_GENAI,
                model="gemini-3.1-flash-lite",
            ),
            "promo_judge": NodeModelConfig(
                provider=ModelProvider.GOOGLE_GENAI,
                model="gemini-3-flash-preview",
            ),
        }
    )

    staff_session_secret: SecretStr | None = None
    staff_session_ttl_seconds: int = Field(default=28_800, ge=60)
    websocket_token_ttl_seconds: int = Field(default=300, ge=30)
    admin_allowed_emails: list[str] = Field(default_factory=list)
    google_oauth_client_id: str | None = None

    postgres_dsn: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    meta_verify_token: SecretStr | None = None
    meta_app_secret: SecretStr | None = None
    meta_phone_number_id: str | None = None
    meta_access_token: SecretStr | None = None

    def model_for(self, node_name: str) -> NodeModelConfig:
        try:
            return self.node_models[node_name]
        except KeyError as error:
            raise ValueError(f"No model configuration exists for node '{node_name}'") from error

    def require_secret(self, name: str, value: SecretStr | None) -> str:
        if value is None or not value.get_secret_value():
            raise ValueError(f"Missing required configuration: {name}")
        return value.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
