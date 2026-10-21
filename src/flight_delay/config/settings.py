"""Environment-backed settings shared by application components."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings with local-development defaults and environment overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    aviationstack_api_key: SecretStr | None = None
    kafka_bootstrap_servers: str = "localhost:9092"
    cassandra_host: str = "localhost"
    cassandra_port: int = Field(default=9042, ge=1, le=65535)
    mlflow_tracking_uri: str = "http://localhost:5000"
    api_port: int = Field(default=8000, ge=1, le=65535)
    kafka_external_port: int = Field(default=9092, ge=1, le=65535)
    spark_master_ui_port: int = Field(default=8080, ge=1, le=65535)
    minio_api_port: int = Field(default=9000, ge=1, le=65535)
    cassandra_port_published: int = Field(default=9042, ge=1, le=65535)
    mlflow_port: int = Field(default=5000, ge=1, le=65535)
    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    grafana_port: int = Field(default=3000, ge=1, le=65535)

    def require_aviationstack_api_key(self) -> SecretStr:
        """Return the configured source API key or explain how to provide one."""
        if (
            self.aviationstack_api_key is None
            or not self.aviationstack_api_key.get_secret_value()
            or self.aviationstack_api_key.get_secret_value().startswith("replace-with-")
        ):
            raise ValueError(
                "AVIATIONSTACK_API_KEY is required for AviationStack access. "
                "Set it in your local .env file."
            )
        return self.aviationstack_api_key


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
