"""Source ingestion, script generation and provider wiring."""

from __future__ import annotations

import pytest

from videogen.config import Settings
from videogen.errors import ConfigError, IngestError, ScriptError
from videogen.ingest import load_source, loader_for
from videogen.ingest.plaintext import RawTextLoader, TextFileLoader, _strip_markdown
from videogen.ingest.web import WebLoader
from videogen.providers.llm.echo_llm import EchoLLM
from videogen.providers.registry import build_image, build_llm, build_tts
from videogen.script.generator import ScriptGenerator, ScriptRequest, _clean_narration
from videogen.script.prompts import SOURCE_DELIMITER, build_user_prompt, script_schema


class TestLoaderDispatch:
    def test_url_routes_to_the_web_loader(self):
        assert isinstance(loader_for("https://example.com/article"), WebLoader)

    def test_text_file_routes_to_the_file_loader(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Title\n\nSome content here.", encoding="utf-8")
        assert isinstance(loader_for(str(path)), TextFileLoader)

    def test_bare_text_routes_to_the_raw_loader(self, sample_text):
        assert isinstance(loader_for(sample_text), RawTextLoader)


class TestTextIngestion:
    def test_loads_raw_text(self, sample_text):
        doc = load_source(sample_text)
        assert doc.kind == "raw"
        assert doc.word_count > 40

    def test_rejects_text_that_is_too_short(self):
        with pytest.raises(IngestError):
            load_source("too short")

    def test_truncates_to_the_word_budget(self, sample_text):
        doc = load_source(sample_text, max_words=10)
        assert doc.word_count == 10
        assert doc.metadata["truncated_from_words"] > 10

    def test_reads_a_markdown_file_and_strips_markup(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text(
            "# Heading\n\nSome **bold** text with a [link](http://x.com) and `code`.\n"
            "\n- bullet one\n- bullet two\n",
            encoding="utf-8",
        )
        doc = load_source(str(path))
        assert "**" not in doc.text
        assert "http://x.com" not in doc.text
        assert "link" in doc.text
        assert doc.name == "doc"

    def test_normalises_smart_punctuation(self):
        doc = load_source("The researcher’s findings were “surprising” indeed here.")
        assert "’" not in doc.text
        assert "“" not in doc.text


class TestMarkdownStripping:
    def test_removes_fenced_code_blocks(self):
        assert "print" not in _strip_markdown("text\n```py\nprint(1)\n```\nmore")

    def test_keeps_link_labels(self):
        assert _strip_markdown("[label](http://x)").strip() == "label"

    def test_removes_images(self):
        assert "alt" not in _strip_markdown("![alt](http://x/img.png)")


class TestEchoLLM:
    def test_returns_a_schema_shaped_object(self, sample_text, settings):
        response = EchoLLM(settings).complete_json(
            system="s",
            user=build_user_prompt(sample_text, min_scenes=3, max_scenes=4, style="cinematic"),
            schema=script_schema(min_scenes=3, max_scenes=4),
        )
        data = response.data
        assert set(data) == {"title", "description", "hashtags", "scenes"}
        assert 3 <= len(data["scenes"]) <= 4
        assert all({"narration", "image_prompt"} <= set(s) for s in data["scenes"])

    def test_extracts_source_after_the_delimiter(self, settings):
        user = f"instructions here\n{SOURCE_DELIMITER}\nThe cat sat on the mat today happily."
        data = EchoLLM(settings).complete_json(system="", user=user, schema={}).data
        assert "instructions here" not in " ".join(s["narration"] for s in data["scenes"])

    def test_is_deterministic(self, sample_text, settings):
        llm = EchoLLM(settings)
        first = llm.complete_json(system="", user=sample_text, schema={}).data
        second = llm.complete_json(system="", user=sample_text, schema={}).data
        assert first == second

    def test_is_marked_offline(self, settings):
        assert EchoLLM(settings).offline is True


class TestScriptGenerator:
    def test_produces_a_validated_script(self, sample_text, settings):
        doc = load_source(sample_text)
        script, usage = ScriptGenerator(EchoLLM(settings), settings).generate(
            ScriptRequest(document=doc)
        )
        assert settings.min_scenes <= len(script.scenes) <= settings.max_scenes
        assert [s.index for s in script.scenes] == list(range(len(script.scenes)))
        assert all(s.narration and s.image_prompt for s in script.scenes)
        assert usage.llm_calls == 1

    def test_trims_scenes_beyond_the_maximum(self, sample_text, settings):
        class Chatty(EchoLLM):
            def complete_json(self, **kwargs):
                response = super().complete_json(**kwargs)
                response.data["scenes"] = response.data["scenes"] * 5
                return response

        script, _ = ScriptGenerator(Chatty(settings), settings).generate(
            ScriptRequest(document=load_source(sample_text))
        )
        assert len(script.scenes) == settings.max_scenes

    def test_raises_when_there_are_no_scenes(self, sample_text, settings):
        class Empty(EchoLLM):
            def complete_json(self, **kwargs):
                response = super().complete_json(**kwargs)
                response.data["scenes"] = []
                return response

        with pytest.raises(ScriptError):
            ScriptGenerator(Empty(settings), settings).generate(
                ScriptRequest(document=load_source(sample_text))
            )

    def test_derives_an_image_prompt_when_one_is_missing(self, sample_text, settings):
        class Partial(EchoLLM):
            def complete_json(self, **kwargs):
                response = super().complete_json(**kwargs)
                response.data["scenes"][0]["image_prompt"] = ""
                return response

        script, _ = ScriptGenerator(Partial(settings), settings).generate(
            ScriptRequest(document=load_source(sample_text))
        )
        assert script.scenes[0].image_prompt


class TestNarrationCleaning:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('Narrator: "Hello there"', "Hello there"),
            ("VOICEOVER: Something", "Something"),
            ("[upbeat] The story begins", "The story begins"),
            ('"Just quoted"', "Just quoted"),
            ("Plain sentence", "Plain sentence"),
        ],
    )
    def test_strips_labels_and_directions(self, raw, expected):
        assert _clean_narration(raw) == expected


class TestPromptBuilding:
    def test_includes_the_source_after_the_delimiter(self):
        prompt = build_user_prompt("BODY TEXT", min_scenes=3, max_scenes=6, style="cinematic")
        assert prompt.split(SOURCE_DELIMITER)[1].strip() == "BODY TEXT"

    def test_optional_directives_appear_only_when_given(self):
        without = build_user_prompt("x", min_scenes=3, max_scenes=6, style="s")
        with_tone = build_user_prompt("x", min_scenes=3, max_scenes=6, style="s", tone="urgent")
        assert "tone" not in without.lower().split(SOURCE_DELIMITER)[0]
        assert "urgent" in with_tone


class TestSchema:
    def test_is_strict_mode_compatible(self):
        schema = script_schema(min_scenes=3, max_scenes=6)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])

        scene = schema["properties"]["scenes"]["items"]
        assert scene["additionalProperties"] is False
        assert set(scene["required"]) == set(scene["properties"])


class TestRegistry:
    @pytest.mark.parametrize("name", ["echo", "offline", "none"])
    def test_offline_llm_aliases(self, settings, name):
        assert build_llm(settings, name).offline

    def test_offline_image_and_tts(self, settings):
        assert build_image(settings, "placeholder").offline
        assert build_tts(settings, "silent").offline

    @pytest.mark.parametrize(
        "builder,name", [(build_llm, "nope"), (build_image, "nope"), (build_tts, "nope")]
    )
    def test_unknown_names_raise(self, settings, builder, name):
        with pytest.raises(ConfigError):
            builder(settings, name)

    def test_missing_credentials_raise_a_config_error(self):
        bare = Settings(openai_api_key=None, llm_provider="openai")
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            build_llm(bare).health_check()

    def test_offline_settings_switch_every_provider(self, settings):
        offline = Settings(
            llm_provider="openai", image_provider="openai", tts_provider="elevenlabs"
        ).offline()
        assert (offline.llm_provider, offline.image_provider, offline.tts_provider) == (
            "echo", "placeholder", "silent",
        )
