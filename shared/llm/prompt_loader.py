"""Jinja2-based prompt template loader from config/prompts/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, BaseLoader, StrictUndefined, TemplateSyntaxError, UndefinedError

from shared.logging import get_logger
from shared.llm.config_loader import _resolve_config_dir

logger = get_logger("prompt_loader")


class PromptNotFoundError(Exception):
    """Raised when a prompt YAML file does not exist."""

    def __init__(self, prompt_name: str) -> None:
        super().__init__(f"Prompt not found: {prompt_name}")
        self.prompt_name = prompt_name


class PromptRenderError(Exception):
    """Raised when Jinja2 rendering fails (missing variable, syntax error)."""

    def __init__(self, prompt_name: str, cause: Exception) -> None:
        super().__init__(f"Prompt render failed for '{prompt_name}': {cause}")
        self.prompt_name = prompt_name
        self.cause = cause


@dataclass
class RenderedPrompt:
    """Result of loading and rendering a prompt template."""

    text: str
    model_key: str
    name: str
    version: str


class PromptLoader:
    """Loads YAML prompt files and renders Jinja2 templates."""

    def __init__(self, config_dir: Path) -> None:
        self._prompts_dir = config_dir / "prompts"
        self._cache: dict[str, dict] = {}
        self._jinja_env = Environment(loader=BaseLoader(), undefined=StrictUndefined)

    def load(self, prompt_name: str, **variables: object) -> RenderedPrompt:
        """Load a prompt YAML and render the Jinja2 template with variables.

        Args:
            prompt_name: Dot-separated or slash-separated path under config/prompts/,
                         e.g. "orchestrator/synthesis" -> config/prompts/orchestrator/synthesis.yaml
            **variables: Template variables to pass to Jinja2.

        Returns:
            RenderedPrompt with rendered text, model_key, name, and version.

        Raises:
            PromptNotFoundError: If the YAML file doesn't exist.
            PromptRenderError: If Jinja2 rendering fails.
        """
        cache_hit = prompt_name in self._cache
        if cache_hit:
            data = self._cache[prompt_name]
        else:
            yaml_path = self._prompts_dir / f"{prompt_name}.yaml"
            if not yaml_path.is_file():
                raise PromptNotFoundError(prompt_name)
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            self._cache[prompt_name] = data

        logger.info(
            "prompt_load",
            component="prompt_loader",
            prompt_name=prompt_name,
            cache_hit=cache_hit,
        )

        template_str = data.get("template", "")
        try:
            template = self._jinja_env.from_string(template_str)
            rendered = template.render(**variables)
        except (UndefinedError, TemplateSyntaxError) as exc:
            raise PromptRenderError(prompt_name, exc) from exc

        return RenderedPrompt(
            text=rendered,
            model_key=data.get("model_key", ""),
            name=data.get("name", ""),
            version=data.get("version", ""),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_loader: PromptLoader | None = None


def get_prompt_loader() -> PromptLoader:
    """Return the module-level singleton PromptLoader, creating it if needed."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader(_resolve_config_dir())
    return _default_loader


def reset_prompt_loader() -> None:
    """Reset the singleton — useful for test isolation."""
    global _default_loader
    _default_loader = None


def load_prompt(name: str, **kwargs: object) -> RenderedPrompt:
    """Module-level convenience function to load and render a prompt."""
    return get_prompt_loader().load(name, **kwargs)
