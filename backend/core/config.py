"""
backend/core/config.py
----------------------
Centralised application settings loaded from .env via pydantic-settings.
All tuneable parameters live here — never scattered across modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_key: str = "dev-secret-key"
    require_api_key: bool = False          # set True in production
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: str = "openai"           # openai | gemini | ollama
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./mechsage.db"

    # ── Simulation ────────────────────────────────────────────────────────────
    simulation_max_cycles: int = 300
    simulation_sleep_seconds: float = 0.5   # seconds between cycles
    simulation_auto_start: bool = True       # start streaming on server boot
    simulation_seed: int = 42

    # ── Inference ─────────────────────────────────────────────────────────────
    serve_models_dir: Path = _REPO_ROOT / "serve_models"
    datasets: List[str] = ["FD001", "FD002", "FD003", "FD004"]

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_relevance_threshold: float = 0.30   # abstain below this score
    rag_top_k: int = 5

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True                   # structured JSON logs in prod

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"openai", "gemini", "ollama"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return v.lower()


settings = Settings()
