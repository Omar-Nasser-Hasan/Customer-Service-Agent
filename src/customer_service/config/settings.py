"""Single environment boundary for current and planned application services."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(StrEnum):
    GOOGLE_GENAI = "google_genai"


class GeminiApiTier(StrEnum):
    FREE = "free"
    PAID = "paid"


class NodeModelConfig(BaseModel):
    provider: ModelProvider
    model: str
    temperature: float = Field(default=0, ge=0, le=2)


class WhatsAppTemplate(BaseModel):
    """A deploy-time-approved template allowed outside WhatsApp's service window."""

    template_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=512)
    language: str = Field(min_length=2, max_length=32)
    components: list[str] = Field(default_factory=lambda: ["body"])
    parameter_count: int = Field(default=0, ge=0, le=20)


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
    gemini_api_tier: GeminiApiTier = GeminiApiTier.FREE
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
            "safety_check": NodeModelConfig(
                provider=ModelProvider.GOOGLE_GENAI,
                model="gemini-3.1-flash-lite",
            ),
            "eval_judge": NodeModelConfig(
                provider=ModelProvider.GOOGLE_GENAI,
                model="gemini-3.1-flash-lite",
            ),
        }
    )

    staff_session_secret: SecretStr | None = None
    staff_session_ttl_seconds: int = Field(default=28_800, ge=60)
    websocket_token_ttl_seconds: int = Field(default=300, ge=30)
    admin_allowed_emails: list[str] = Field(default_factory=list)
    google_oauth_client_id: str | None = None
    admin_cookie_domain: str | None = None
    admin_cookie_secure: bool = False
    admin_api_url: str = "http://localhost:8000"
    csrf_cookie_name: str = "customer_service_csrf"
    staff_session_cookie_name: str = "customer_service_staff_session"
    whatsapp_phone_encryption_key: SecretStr | None = None
    whatsapp_thread_hmac_key: SecretStr | None = None
    meta_graph_api_version: str = "v26.0"
    whatsapp_template_catalog: list[WhatsAppTemplate] = Field(default_factory=list)
    worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=60)
    worker_max_attempts: int = Field(default=8, ge=1, le=50)
    whatsapp_service_window_hours: int = Field(default=24, ge=1, le=72)

    postgres_dsn: SecretStr | None = None
    langgraph_strict_msgpack: bool = True
    checkpoint_retention_days: int = Field(default=90, ge=1)
    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-3.5-lite"
    voyage_embedding_dimensions: int = Field(default=1024, ge=1, le=2048)
    faq_corpus_version: str = "v1"
    faq_min_similarity: float | None = Field(default=None, ge=-1, le=1)
    faq_sync_on_startup: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "customer-service-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_sample_rate: float = Field(default=1.0, ge=0, le=1)
    trace_thread_hash_salt: SecretStr | None = None
    meta_verify_token: SecretStr | None = None
    meta_app_secret: SecretStr | None = None
    meta_phone_number_id: str | None = None
    meta_access_token: SecretStr | None = None

    @model_validator(mode="after")
    def ensure_default_node_models(self) -> "Settings":
        """Merge newly introduced node defaults with partial env mappings.

        Pydantic-settings treats a nested dictionary supplied through one or
        more environment variables as a replacement mapping. Keeping this
        merge at the settings boundary means adding an evaluator or future
        node cannot break deployments whose .env only overrides existing
        production nodes.
        """

        defaults = {
            "assistant": NodeModelConfig(provider=ModelProvider.GOOGLE_GENAI, model="gemini-3.1-flash-lite"),
            "promo_judge": NodeModelConfig(provider=ModelProvider.GOOGLE_GENAI, model="gemini-3-flash-preview"),
            "safety_check": NodeModelConfig(provider=ModelProvider.GOOGLE_GENAI, model="gemini-3.1-flash-lite"),
            "eval_judge": NodeModelConfig(provider=ModelProvider.GOOGLE_GENAI, model="gemini-3.1-flash-lite"),
        }
        for name, default in defaults.items():
            self.node_models.setdefault(name, default)
        return self

    def model_for(self, node_name: str) -> NodeModelConfig:
        try:
            return self.node_models[node_name]
        except KeyError as error:
            raise ValueError(f"No model configuration exists for node '{node_name}'") from error

    def require_secret(self, name: str, value: SecretStr | None) -> str:
        if value is None or not value.get_secret_value():
            raise ValueError(f"Missing required configuration: {name}")
        return value.get_secret_value()

    @field_validator("langgraph_strict_msgpack")
    @classmethod
    def strict_msgpack_must_remain_enabled(cls, value: bool) -> bool:
        if not value:
            raise ValueError("LANGGRAPH_STRICT_MSGPACK must remain enabled")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
