# Security

## Reporting a vulnerability

Open a [security advisory](https://github.com/NabiBukhsh-AI/VideoGenerator/security/advisories/new)
rather than a public issue.

## API keys

This tool talks to paid APIs. Handling of credentials:

- Keys are read from environment variables or a local `.env` file, never from source.
- They are held as pydantic `SecretStr`, so they do not appear in logs, `repr()` output,
  or tracebacks.
- `.env` is gitignored. `.env.example` contains only empty placeholders.
- `videogen doctor` reports whether a key is *set*, never its value.

### If a key is ever committed

Treat it as public the moment it is pushed, even to a private repo, and even if the
commit is later removed. Rotate it:

- **OpenAI** — https://platform.openai.com/api-keys
- **Anthropic** — https://console.anthropic.com/settings/keys
- **ElevenLabs** — https://elevenlabs.io/app/settings/api-keys

Rewriting git history does not un-leak a key. Anything pushed to GitHub may already
have been cloned, cached, or indexed by automated scrapers. Rotation is the only fix.

> **Note for this repository:** versions before v2.0.0 contained hardcoded OpenAI and
> ElevenLabs keys in `Code/main.py`, `Code/images.py`, and `Code/narration.py`. Those
> keys are present in the git history and must be considered compromised. They have no
> counterpart in the current codebase.

## Generated content

Image prompts are written by a language model and sent to an image API. The script
prompt instructs the model to avoid real people, logos, and sensitive imagery, because
those prompts are rejected by provider content filters. This reduces failures; it is not
a guarantee. Review output before publishing.
