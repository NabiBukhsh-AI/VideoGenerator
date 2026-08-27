# VideoGenerator

Turn a PDF, a web article, or a block of text into a narrated, captioned short-form
video — vertical for Shorts/Reels/TikTok, square, or landscape.

```bash
videogen generate paper.pdf --voice Rachel --music bed.mp3
```

The pipeline reads your source material, writes a script with an LLM, generates a
background image per scene, narrates it with text-to-speech, and renders an H.264 MP4
with word-synchronised captions, Ken Burns motion, crossfades, and an auto-ducked music
bed. It also emits SRT/VTT subtitles, a thumbnail, and publishing metadata.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [CLI](#cli)
- [Python API](#python-api)
- [Web UI](#web-ui)
- [Configuration](#configuration)
- [Providers](#providers)
- [Architecture](#architecture)
- [Development](#development)
- [What changed in v2](#what-changed-in-v2)

---

## Install

Requires **Python 3.10+** and **ffmpeg**.

```bash
git clone https://github.com/NabiBukhsh-AI/VideoGenerator.git
cd VideoGenerator

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[all]"
```

**ffmpeg** (needed for rendering, not for script generation):

| Platform | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

Then check your setup:

```bash
videogen doctor
```

### Optional extras

Install only what you need instead of `[all]`:

```bash
pip install -e ".[openai,pdf]"       # OpenAI + PDF reading
pip install -e ".[anthropic]"        # Claude for script generation
pip install -e ".[elevenlabs]"       # ElevenLabs narration
pip install -e ".[web]"              # read articles from URLs
pip install -e ".[ui]"               # Streamlit interface
```

---

## Quick start

### 1. Try it with no API keys

Every provider has an offline stub, so you can see the whole pipeline work before
spending anything:

```bash
videogen generate --dry-run "The deep ocean is the least explored place on Earth. \
Sunlight fades below one thousand metres. Creatures there make their own light."
```

That produces a real MP4 with placeholder images and silent narration — enough to check
framing, captions, timing, and encoding.

### 2. Add your keys

```bash
videogen init          # writes .env
```

Edit `.env`, then:

```bash
videogen doctor        # confirms what's usable
```

### 3. Generate for real

```bash
videogen generate article.pdf --voice Rachel
videogen generate https://example.com/blog-post --aspect 16:9
videogen generate notes.md --tone urgent --scenes 5 --music bed.mp3
```

---

## How it works

```
  source            script              assets                  render
┌─────────┐      ┌──────────┐      ┌──────────────┐      ┌────────────────┐
│  PDF    │      │          │      │  images  ────┼──┐   │ Ken Burns      │
│  URL    ├─────▶│   LLM    ├─────▶│  (parallel)  │  ├──▶│ crossfades     │──▶ .mp4
│  text   │      │  + JSON  │      │  narration ──┼──┘   │ captions       │    .srt
│  .md    │      │  schema  │      │  (parallel)  │      │ ducked music   │    .jpg
└─────────┘      └──────────┘      └──────────────┘      └────────────────┘
```

1. **Ingest** — extract text from a PDF (stripping running headers and repairing
   hyphenation), an article URL, a Markdown/text file, or a raw string.
2. **Script** — one LLM call returns a *validated JSON object*: title, description,
   hashtags, and a list of scenes each with narration and an image prompt.
3. **Assets** — images and narration are generated **concurrently** and cached by
   content hash, so a re-run only pays for what changed.
4. **Align** — each word gets a start/end time, either heuristically (free) or from
   Whisper timestamps (`VIDEOGEN_ALIGNER=whisper`).
5. **Render** — frames are generated in memory and piped straight into ffmpeg as raw
   RGB, encoded once to H.264/`yuv420p` MP4.

---

## CLI

```
videogen generate SOURCE [OPTIONS]
videogen doctor                       # check keys, deps, ffmpeg
videogen providers                    # what's ready to run
videogen ui                           # launch the web interface
videogen init                         # scaffold .env
```

### Frequently used options

| Option | Default | Description |
|---|---|---|
| `--aspect` | `9:16` | `9:16`, `1:1`, or `16:9` |
| `--voice` | provider default | TTS voice name or ID |
| `--scenes / -n` | `6` | Maximum number of scenes |
| `--music / -m` | — | Background track, auto-ducked under narration |
| `--music-gain` | `-18.0` | Music bed level in dB |
| `--tone` | — | `urgent`, `wry`, `warm`, … |
| `--language` | — | Narration language |
| `--focus` | — | Angle to emphasise |
| `--motion` | `auto` | `zoom_in`, `pan_left`, `none`, … |
| `--transition` | `crossfade` | `crossfade`, `cut`, `fade_to_black` |
| `--no-captions` | off | Skip burned-in captions |
| `--caption-color` | `#FFFFFF` | Caption text colour |
| `--highlight-color` | `#FFD400` | Colour of the word being spoken |
| `--font` | auto-detected | Path to a `.ttf` |
| `--llm/--image/--tts` | from `.env` | Override a provider for one run |
| `--crf` | `20` | H.264 quality; lower is better |
| `--dry-run` | off | Offline providers only, no cost |
| `--script-only` | off | Stop after writing the script |
| `--resume` | — | Continue an existing project directory |

### Recipes

```bash
# Preview the script before spending anything on images or narration
videogen generate report.pdf --script-only

# A run died halfway. Resume it - cached assets are reused.
videogen generate report.pdf --resume output/20260827-143022-report

# Mix providers: Claude writes, OpenAI draws, ElevenLabs speaks
videogen generate notes.md --llm anthropic --image openai --tts elevenlabs

# Landscape explainer with static framing and larger caption chunks
videogen generate paper.pdf --aspect 16:9 --motion none --scenes 8
```

---

## Python API

```python
from videogen import GenerationRequest, generate
from videogen.models import AspectRatio, KenBurnsKind, VideoSpec

result = generate(
    GenerationRequest(
        source="research.pdf",
        spec=VideoSpec(
            aspect=AspectRatio.VERTICAL,
            ken_burns=KenBurnsKind.ZOOM_IN,
            caption_highlight_color="#00E5FF",
        ),
        voice="Rachel",
        tone="curious",
    )
)

print(result.video_path)                     # output/.../title.mp4
print(result.subtitle_path)                  # .srt
print(f"${result.usage.estimated_cost_usd:.3f}")
```

Two-step flow — review the script before generating assets:

```python
from videogen.config import get_settings
from videogen.ingest import load_source
from videogen.providers import build_llm
from videogen.script.generator import ScriptGenerator, ScriptRequest
from videogen.pipeline import GenerationRequest, Pipeline

settings = get_settings()
document = load_source("research.pdf")
script, usage = ScriptGenerator(build_llm(settings), settings).generate(
    ScriptRequest(document=document)
)

script.scenes[0].narration = "A punchier opening line."   # edit freely

result = Pipeline(settings).run(
    GenerationRequest(source="research.pdf", script=script)
)
```

---

## Web UI

```bash
videogen ui        # http://localhost:8501
```

Upload a file, paste a URL, or type text. The UI generates the script first and lets
you **edit every scene** before committing to image and narration generation.

---

## Configuration

Settings resolve from environment variables, then `.env`, then defaults. See
[`.env.example`](.env.example) for the full list.

Most-used values:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Script, images, TTS, Whisper alignment |
| `ANTHROPIC_API_KEY` | — | Script generation via Claude |
| `ELEVENLABS_API_KEY` | — | Narration |
| `VIDEOGEN_MAX_SCENES` | `6` | Scene cap |
| `VIDEOGEN_IMAGE_STYLE` | cinematic… | Style appended to every image prompt |
| `VIDEOGEN_MAX_WORKERS` | `4` | Concurrent API calls |
| `VIDEOGEN_ALIGNER` | `heuristic` | `heuristic` or `whisper` |
| `VIDEOGEN_CACHE_ENABLED` | `true` | Reuse assets across runs |
| `VIDEOGEN_FONT_PATH` | auto | Caption font |

> **Secrets never belong in source.** `.env` is gitignored. If a key has ever been
> committed — to this repo or any other — treat it as public and rotate it.

---

## Providers

Every capability is an interface with at least one online and one offline
implementation.

| Kind | Online | Offline stub |
|---|---|---|
| **LLM** | `openai` (structured output), `anthropic` (tool-forced JSON) | `echo` |
| **Image** | `openai` (`gpt-image-1`, `dall-e-3`) | `placeholder` |
| **TTS** | `elevenlabs`, `openai` | `silent` |

The stubs are what make `--dry-run` and the test suite work with no keys and no network.

### Adding a provider

Implement the interface and register it:

```python
# videogen/providers/image/my_provider.py
from videogen.providers.base import ImageProvider

class MyImage(ImageProvider):
    name = "mine"

    def generate(self, *, prompt: str, output_path: Path, size: str) -> Path:
        ...
        return output_path
```

Add one branch to `videogen/providers/registry.py` and a name to `KNOWN`. Nothing else
changes — the pipeline only knows the interface.

---

## Architecture

```
videogen/
├── config.py         Settings via pydantic-settings; secrets as SecretStr
├── models.py         Validated domain types (Script, Scene, VideoSpec, …)
├── pipeline.py       Stage orchestration
├── cli.py            Typer CLI
├── errors.py         Exception hierarchy, each with a fix hint
├── ingest/           PDF, Markdown/text, HTML, URL, raw text loaders
├── script/           Prompts, JSON schema, script generation
├── providers/        LLM / image / TTS interfaces + implementations
│   ├── base.py         the three ABCs
│   └── registry.py     name → instance
├── render/
│   ├── ffmpeg.py       binary wrapper, raw-frame pipe, probing
│   ├── frames.py       Ken Burns, transitions, the timeline
│   ├── captions.py     layout, word highlighting, cached overlays
│   ├── align.py        heuristic and Whisper word timing
│   ├── audio.py        narration concat, sidechain-ducked music
│   ├── subtitles.py    SRT / VTT
│   └── compositor.py   render orchestration
├── storage/          Project directories, manifests, content-hash caching
└── ui/               Streamlit app
```

**Design rules.** The pipeline depends only on the provider ABCs, never on a vendor SDK.
Optional dependencies are imported inside the functions that use them, so a partial
install still works. Every error is a `VideoGenError` carrying a hint. Nothing is
written to disk twice.

---

## Development

```bash
pip install -e ".[all,dev]"

pytest                       # full suite
pytest -m "not ffmpeg"       # skip tests needing the binary
pytest --cov=videogen

ruff check videogen tests
ruff format videogen tests
mypy videogen
```

Tests run entirely offline — no keys, no network, no cost. Tests that need ffmpeg are
marked `ffmpeg` and skip automatically when it is absent.

---

## What changed in v2

The original was a working prototype: a single Streamlit script with the whole flow
inlined twice. This release rebuilds it around a modular core. The substantive fixes:

**Security**
- API keys were hardcoded in three source files and committed. All credentials now come
  from the environment, are held as `SecretStr`, and `.env` is gitignored.

**Correctness**
- `response_text.replace(...)` discarded its result, so smart quotes reached the TTS
  engine unchanged. Text normalisation now returns the cleaned value.
- Caption lines were centred by measuring the whole string but drawn word-by-word with
  a fixed `+10` advance, so every line drifted right and long lines ran off-frame.
  Layout now measures the real space width in the actual font.
- Scene durations subtracted the fade length inconsistently, so picture and narration
  drifted apart. The timeline now places scenes at exact audio durations.
- `cv2.addWeighted(image1, 1, image2, 0, 0)` was a no-op that computed nothing.
- Background music used `10 * log10(1/strength)`, which divides by zero at strength 0.
- The Streamlit music slider was created *inside* the button handler, so it could never
  return anything but its default.

**Modernisation**
- `elevenlabs.set_api_key/generate/save` were removed in ElevenLabs 1.0; `moviepy.editor`
  was removed in MoviePy 2.0. Both are gone.
- Output was XVID `.avi`. It is now H.264 MP4 with `yuv420p` and `+faststart`.
- Script parsing relied on line prefixes (`Narrator: `, `[`), which broke on any
  reformat. The LLM now returns a schema-validated JSON object.
- OpenCV Hershey fonts (ASCII-only) replaced with PIL TrueType rendering.

**New**
- CLI, offline dry-run mode, provider abstraction with Anthropic support, URL and
  Markdown ingestion, Ken Burns motion, sidechain music ducking, Whisper caption
  alignment, SRT/VTT export, thumbnails, publishing metadata, content-hash caching with
  `--resume`, concurrent asset generation, cost estimation, structured logging, and a
  test suite.

---

## License

MIT — see [LICENSE](LICENSE).
