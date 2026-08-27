"""Prompts and the output schema for script generation.

Two things changed from the original prompt:

1. Output is a JSON object validated against a schema, not prose parsed by line
   prefixes. The old format broke whenever the model added a preamble or reformatted.
2. The content rules are stated as constraints on the image prompts only, with the
   reason given, so the model does not over-apply them to the narration.

This module imports nothing from the rest of the package, so the offline `EchoLLM` can
read `SOURCE_DELIMITER` without creating an import cycle.
"""

from __future__ import annotations

from typing import Any

#: Marks where the source material begins in the user message.
SOURCE_DELIMITER = "===== SOURCE MATERIAL ====="

SYSTEM_PROMPT = """You are a senior short-form video writer. You turn source material \
into a tight, spoken-word script for a vertical short (YouTube Shorts, TikTok, Reels).

WRITING THE NARRATION
- Open with a hook in the first sentence. The viewer decides in two seconds.
- One idea per scene. Each narration line is a single spoken sentence.
- Write for the ear, not the page: short clauses, active voice, concrete nouns.
- Stay factually faithful to the source material. Do not invent statistics, quotes,
  dates or events that are not in it.
- End on a payoff or a question, never on a cliffhanger that the source does not answer.
- The narration is fed to a text-to-speech engine. Use plain ASCII punctuation only.
  Spell out symbols and numerals that would be misread ("percent", not "%").
- Real names, organisations and events are fine in the narration.

WRITING THE IMAGE PROMPTS
Each scene needs a prompt for a text-to-image model. These have stricter rules,
because the image API rejects prompts that violate them and the scene then fails:
- Never name a real person, living or dead, and never describe a recognisable
  likeness of one. Describe an anonymous figure by role instead ("a scientist in a
  lab coat", "a commuter on a platform").
- No logos, trademarks, brand names, or copyrighted characters.
- No sexual content, gore, or graphic violence.
- Do not ask for text, captions, watermarks or letters inside the image; captions are
  burned in afterwards and baked-in text will collide with them.
- Describe a single, static, photographable scene: subject, setting, lighting, mood,
  camera framing. Aim for 15 to 35 words.
- Keep a consistent visual style across all scenes so the video feels like one piece.
- Compose for a tall vertical frame with the subject centred.

Return your answer using the provided structured format."""


def build_user_prompt(
    source_text: str,
    *,
    min_scenes: int,
    max_scenes: int,
    style: str,
    tone: str | None = None,
    language: str | None = None,
    focus: str | None = None,
) -> str:
    """Render the user message for a script request."""
    lines = [
        f"Write a short-video script with between {min_scenes} and {max_scenes} scenes.",
        "Target a total runtime of 30 to 60 seconds when read aloud at a natural pace.",
        f"Apply this visual style to every image prompt: {style}.",
    ]
    if tone:
        lines.append(f"Narration tone: {tone}.")
    if language:
        lines.append(f"Write the narration in {language}. Keep image prompts in English.")
    if focus:
        lines.append(f"Focus the script on this angle: {focus}.")

    lines += [
        "",
        "Also return a title (under 70 characters), a one-paragraph description suitable "
        "for a video platform, and 3 to 6 relevant hashtags without the # symbol.",
        "",
        SOURCE_DELIMITER,
        source_text,
    ]
    return "\n".join(lines)


def script_schema(*, min_scenes: int, max_scenes: int) -> dict[str, Any]:
    """JSON schema for the script object.

    Written to satisfy OpenAI's strict structured-output mode, which requires
    ``additionalProperties: false`` and every property listed in ``required``. Claude's
    tool-input schema accepts the same document.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "description", "hashtags", "scenes"],
        "properties": {
            "title": {
                "type": "string",
                "description": "Punchy title for the video, under 70 characters.",
            },
            "description": {
                "type": "string",
                "description": "One-paragraph description for the video platform.",
            },
            "hashtags": {
                "type": "array",
                "description": "Between 3 and 6 hashtags, without the # symbol.",
                "items": {"type": "string"},
            },
            "scenes": {
                "type": "array",
                "description": (
                    f"Between {min_scenes} and {max_scenes} scenes, in order."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["narration", "image_prompt"],
                    "properties": {
                        "narration": {
                            "type": "string",
                            "description": (
                                "One spoken sentence. ASCII punctuation only. "
                                "No stage directions or speaker labels."
                            ),
                        },
                        "image_prompt": {
                            "type": "string",
                            "description": (
                                "Text-to-image prompt for this scene's background. "
                                "No real people, no logos, no text in the image."
                            ),
                        },
                    },
                },
            },
        },
    }


#: Appended to every image prompt at generation time. Keeping it out of the LLM's
#: output means we can tune the look without regenerating the script.
IMAGE_PROMPT_SUFFIX = (
    "Vertical composition filling the full frame. No text, no letters, no watermark, "
    "no logo, no signature."
)
