import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv, find_dotenv

@dataclass
class ObservabilityConfig:
    """Configuration for LLM observability"""

    # Required settings with defaults
    datahub_gms_url: str = "http://localhost:8080"
    datahub_frontend_url: str = "http://localhost:9002"

    # Optional settings
    datahub_token: Optional[str] = None
    datahub_secret: Optional[str] = None
    datahub_dry_run: bool = False

    # LangChain/LangSmith settings
    langsmith_api_key: Optional[str] = None
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_project: str = "default"
    langchain_verbose: bool = False
    langchain_handler: str = "langchain"

    # Platform settings
    enabled_platforms: list = field(default_factory=lambda: ["langchain", "langsmith"])
    default_emitter: str = "datahub"

    # Ingestion settings
    ingest_window_days: int = 7
    ingest_batch_size: int = 100
    ingest_limit: int = 1000

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'ObservabilityConfig':
        """Create config from environment variables"""
        if env_file:
            load_dotenv(env_file)
        else:
            env_file = find_dotenv()
            if env_file:
                load_dotenv(env_file)

        return cls(
            # DataHub settings
            datahub_gms_url=os.getenv("DATAHUB_GMS_URL", "http://localhost:8080"),
            datahub_frontend_url=os.getenv("DATAHUB_FRONTEND_URL", "http://localhost:9002"),
            datahub_token=os.getenv("DATAHUB_TOKEN"),
            datahub_secret=os.getenv("DATAHUB_SECRET"),
            datahub_dry_run=os.getenv("DATAHUB_DRY_RUN", "false").lower() == "true",

            # LangSmith settings
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true",
            langchain_endpoint=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            langchain_project=os.getenv("LANGCHAIN_PROJECT", "default"),
            langchain_verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
            langchain_handler=os.getenv("LANGCHAIN_HANDLER", "langchain"),

            # Platform settings
            enabled_platforms=os.getenv("ENABLED_PLATFORMS", "langsmith,langchain").split(","),
            default_emitter=os.getenv("DEFAULT_EMITTER", "datahub"),

            # Ingestion settings
            ingest_window_days=int(os.getenv("INGEST_WINDOW_DAYS", "7")),
            ingest_batch_size=int(os.getenv("INGEST_BATCH_SIZE", "100")),
            ingest_limit=int(os.getenv("INGEST_LIMIT", "1000"))
        )

    def validate(self) -> None:
        """Validate configuration"""
        if not self.datahub_dry_run:
            if not self.datahub_gms_url:
                raise ValueError("DataHub GMS URL is required when not in dry run mode")
            if not self.datahub_token:
                raise ValueError("DataHub token is required when not in dry run mode")

        if "langsmith" in self.enabled_platforms and not self.langsmith_api_key:
            raise ValueError("LangSmith API key is required when LangSmith platform is enabled")

    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific configuration"""
        configs = {
            "langsmith": {
                "api_key": self.langsmith_api_key,
                "project": self.langchain_project,
                "endpoint": self.langchain_endpoint,
                "tracing_v2": self.langchain_tracing_v2,
                "verbose": self.langchain_verbose,
                "handler": self.langchain_handler
            },
            "datahub": {
                "gms_url": self.datahub_gms_url,
                "frontend_url": self.datahub_frontend_url,
                "token": self.datahub_token,
                "secret": self.datahub_secret,
                "dry_run": self.datahub_dry_run
            }
        }
        return configs.get(platform, {})

class ObservabilitySetup:
    """Setup and initialization for LLM observability"""

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.emitters = {}
        self.connectors = {}
        self.observers = {}

    def setup(self) -> None:
        """Setup all components based on configuration"""
        self._setup_emitters()
        self._setup_connectors()
        self._setup_observers()

    def _setup_emitters(self) -> None:
        """Setup configured emitters"""
        from .emitters.console import ConsoleEmitter
        from .emitters.datahub import DataHubEmitter

        # Always setup console emitter
        self.emitters["console"] = ConsoleEmitter()

        # Setup DataHub emitter if not in dry run mode
        if not self.config.datahub_dry_run:
            self.emitters["datahub"] = DataHubEmitter(
                gms_server=self.config.datahub_gms_url
            )

    def _setup_connectors(self) -> None:
        """Setup platform connectors"""
        if "langchain" in self.config.enabled_platforms:
            from .platforms.langchain import LangChainConnector
            self.connectors["langchain"] = LangChainConnector()

        if "langsmith" in self.config.enabled_platforms:
            from .platforms.langsmith import LangSmithConnector
            self.connectors["langsmith"] = LangSmithConnector(
                config=self.config
            )

        if "openai" in self.config.enabled_platforms:
            from .platforms.openai import OpenAIConnector
            self.connectors["openai"] = OpenAIConnector(
                api_key=self.config.openai_api_key
            )

        if "anthropic" in self.config.enabled_platforms:
            from .platforms.anthropic import AnthropicConnector
            self.connectors["anthropic"] = AnthropicConnector(
                api_key=self.config.anthropic_api_key
            )

    def _setup_observers(self) -> None:
        """Setup observers for each platform"""
        if "langchain" in self.connectors:
            from .platforms.langchain import LangChainObserver
            self.observers["langchain"] = LangChainObserver(
                config=self.config,
                emitter=self.emitters[self.config.default_emitter]
            )

    def get_observer(self, platform: str):
        """Get observer for a specific platform"""
        return self.observers.get(platform)

    def get_emitter(self, name: str = None):
        """Get emitter by name"""
        return self.emitters.get(name or self.config.default_emitter)

    def get_connector(self, platform: str):
        """Get connector for a specific platform"""
        return self.connectors.get(platform)
