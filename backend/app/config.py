"""Application configuration (environment variables prefixed with ``DL_``; see .env.example)."""

from __future__ import annotations

import contextlib
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
WEAK_PASSWORDS = {"", "admin", "password", "change-me-now", "admin123"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DL_", env_file=(str(REPO_ROOT / ".env"), ".env"),
                                      env_file_encoding="utf-8", extra="ignore")

    env: str = Field("development", description="development | production | test")
    secret_key: str = ""
    var_dir: Path = REPO_ROOT / "var"
    database_url: str = ""
    datasets_dir: Path = Path()  # unset -> <var_dir>/datasets
    models_dir: Path = Path()  # unset -> <var_dir>/models
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    serve_frontend: bool = True

    # comma-separated in the environment (NoDecode: do not JSON-parse, the validator splits it)
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cookie_secure: bool = False
    # CSP frame-ancestors. Empty = "'self'" in production, not restricted otherwise (embedded dev previews).
    frame_ancestors: str = ""
    access_token_minutes: int = 480

    admin_username: str = "admin"
    admin_password: str = "change-me-now"  # noqa: S105 - documented first-run default, forced change warning
    seed_demo_users: bool = True

    max_upload_mb: int = 20
    max_zip_mb: int = 500
    max_zip_files: int = 2000
    max_zip_uncompressed_mb: int = 4000
    rate_limit_per_min: int = 600
    login_rate_limit_per_min: int = 10

    worker_mode: str = "thread"  # thread | external | off
    warmup_models: bool = True  # load active models in the background at startup
    worker_poll_seconds: float = 1.0
    job_stale_seconds: int = 300
    device: str = "auto"
    model_cache_size: int = 3

    auto_migrate: bool = True
    log_format: str = "text"  # text | json
    log_level: str = "INFO"
    duplicate_window_hours: int = 24

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split(cls, v):
        if isinstance(v, str):
            if v.strip().startswith("["):  # JSON list form is accepted too
                import json

                return [str(x).strip() for x in json.loads(v)]
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def resolved(self) -> Settings:
        self.var_dir = Path(self.var_dir).resolve()
        self.var_dir.mkdir(parents=True, exist_ok=True)
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.var_dir / 'defectlens.db').as_posix()}"
        unset = Path()
        self.datasets_dir = (self.var_dir / "datasets" if self.datasets_dir == unset else Path(self.datasets_dir)).resolve()
        self.models_dir = (self.var_dir / "models" if self.models_dir == unset else Path(self.models_dir)).resolve()
        for d in (self.datasets_dir, self.models_dir, self.data_dir, self.reports_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.secret_key:
            if self.is_production:
                raise RuntimeError("DL_SECRET_KEY must be set in production")
            key_file = self.var_dir / "secret.key"
            if not key_file.exists():
                key_file.write_text(secrets.token_urlsafe(48), encoding="utf-8")
                with contextlib.suppress(OSError):
                    key_file.chmod(0o600)
            self.secret_key = key_file.read_text(encoding="utf-8").strip()
        if self.is_production:
            if self.admin_password in WEAK_PASSWORDS or len(self.admin_password) < 10:
                raise RuntimeError("Set a strong DL_ADMIN_PASSWORD (>=10 chars) in production")
            if len(self.secret_key) < 32:
                raise RuntimeError("DL_SECRET_KEY must be at least 32 characters")
        return self

    @property
    def data_dir(self) -> Path:
        return self.var_dir / "data"

    @property
    def reports_dir(self) -> Path:
        return self.var_dir / "reports"

    @property
    def tmp_dir(self) -> Path:
        return self.var_dir / "tmp"


@lru_cache
def get_settings() -> Settings:
    return Settings().resolved()  # type: ignore[call-arg]  # values come from env/.env
