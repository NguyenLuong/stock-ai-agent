"""Tests for shared.llm.prompt_loader — Jinja2 template loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from shared.llm.prompt_loader import (
    PromptLoader,
    PromptNotFoundError,
    PromptRenderError,
    RenderedPrompt,
    reset_prompt_loader,
)


@pytest.fixture(autouse=True)
def _reset_loader():
    reset_prompt_loader()
    yield
    reset_prompt_loader()


def _make_prompt_yaml(
    name: str = "test_prompt",
    version: str = "1.0",
    model_key: str = "triage",
    template: str = "Hello {{ name }}!",
) -> str:
    data = {
        "name": name,
        "version": version,
        "model_key": model_key,
        "template": template,
    }
    return yaml.dump(data)


class TestPromptLoaderLoad:
    def test_load_success_renders_template(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        yaml_file = prompts_dir / "greeting.yaml"
        yaml_file.write_text(
            _make_prompt_yaml(
                name="greeting",
                model_key="triage",
                template="Xin chào {{ user }}! Hôm nay là {{ date }}.",
            )
        )

        loader = PromptLoader(tmp_path)
        result = loader.load("greeting", user="Luong", date="2026-03-06")

        assert isinstance(result, RenderedPrompt)
        assert "Xin chào Luong!" in result.text
        assert "Hôm nay là 2026-03-06" in result.text
        assert result.model_key == "triage"
        assert result.name == "greeting"
        assert result.version == "1.0"

    def test_prompt_not_found_error(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        loader = PromptLoader(tmp_path)
        with pytest.raises(PromptNotFoundError) as exc_info:
            loader.load("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_prompt_render_error_missing_variable(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        yaml_file = prompts_dir / "strict.yaml"
        yaml_file.write_text(
            _make_prompt_yaml(template="Hello {{ required_var }}!")
        )

        loader = PromptLoader(tmp_path)
        with pytest.raises(PromptRenderError) as exc_info:
            loader.load("strict")  # No variables passed
        assert "strict" in str(exc_info.value)

    def test_cache_hit_reads_file_once(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        yaml_file = prompts_dir / "cached.yaml"
        yaml_file.write_text(
            _make_prompt_yaml(template="Hello {{ name }}!")
        )

        loader = PromptLoader(tmp_path)

        result1 = loader.load("cached", name="A")
        result2 = loader.load("cached", name="B")

        assert result1.text == "Hello A!"
        assert result2.text == "Hello B!"
        # Cache should have the entry
        assert "cached" in loader._cache

    def test_nested_path(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts" / "orchestrator"
        prompts_dir.mkdir(parents=True)
        yaml_file = prompts_dir / "synthesis.yaml"
        yaml_file.write_text(
            _make_prompt_yaml(
                name="orchestrator_synthesis",
                model_key="orchestrator_synthesis",
                template="Synthesize {{ ticker }}: {{ agent_results }}",
            )
        )

        loader = PromptLoader(tmp_path)
        result = loader.load("orchestrator/synthesis", ticker="HPG", agent_results="bullish")

        assert "Synthesize HPG: bullish" in result.text
        assert result.model_key == "orchestrator_synthesis"
        assert result.name == "orchestrator_synthesis"

    def test_rendered_prompt_fields(self, tmp_path: Path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        yaml_file = prompts_dir / "fields.yaml"
        yaml_file.write_text(
            _make_prompt_yaml(
                name="my_prompt",
                version="2.0",
                model_key="terminal_chat",
                template="Content: {{ data }}",
            )
        )

        loader = PromptLoader(tmp_path)
        result = loader.load("fields", data="test")

        assert result.text == "Content: test"
        assert result.model_key == "terminal_chat"
        assert result.name == "my_prompt"
        assert result.version == "2.0"


class TestLoadPromptModuleLevel:
    def test_load_prompt_delegates_to_singleton(self, tmp_path: Path, monkeypatch):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "greeting.yaml").write_text(
            _make_prompt_yaml(name="greeting", model_key="triage", template="Hi {{ user }}!")
        )
        monkeypatch.setenv("CONFIG_DIR", str(tmp_path))

        from shared.llm.prompt_loader import load_prompt
        result = load_prompt("greeting", user="Luong")

        assert isinstance(result, RenderedPrompt)
        assert result.text == "Hi Luong!"
        assert result.model_key == "triage"

    def test_load_prompt_is_not_coroutine(self):
        import inspect
        from shared.llm.prompt_loader import load_prompt
        assert not inspect.iscoroutinefunction(load_prompt)
