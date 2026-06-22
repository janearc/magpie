"""The magpie pipeline: a voice-memo m4a in, a clean transcript out.

A run is a bento (the shared unit of work): a directory holding the source audio
under raw_data/ and the transcripts under outputs/. The steps are banchans --
transcribe, then cleanup -- and each one's lifecycle is what we eventually emit to
the bus via the good-citizen Go sidecar (blm#11 wiring is the next step).

Archival rule (Max): duplicates over loss. We COPY the source audio into the
bento; we never move or delete the operator's original.
"""

import json
import logging
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# whisper large-v3 (MLX) -- mlx_whisper resolves this from the shared, read-only
# HF cache; we do not vendor weights. ffmpeg (a hard dep of mlx_whisper) decodes
# the m4a. The model id is a logical name; the good-citizen model abstraction
# (blm#14) will own this resolution later, the same way it does for mistral/flan.
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# the cleanup model. mistral via the local ollama, kept in-enclave. This direct
# call is INTERIM -- it is exactly the seam the good-citizen model-client (blm#14)
# replaces, so cleanup routes through one model abstraction instead of a hand-rolled
# ollama POST. The prompt's whole job is to delete whisper's loop artifacts (the
# "okay" x337 / "Let's go." x40 pathology) WITHOUT rewriting the words.
CLEANUP_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434"
_CLEANUP_PROMPT = (
    "The following is a raw speech-to-text transcript. It may contain stutter-loop "
    "artifacts where a phrase repeats many times in a row -- that is a transcription "
    "bug, not the speaker. Remove only those runaway repetitions and obvious "
    "duplicate lines. Do NOT paraphrase, summarize, correct grammar, or change any "
    "wording. Return only the cleaned transcript.\n\n---\n"
)

# data root: code is public, data is private. Everything magpie writes lives under
# ${HOME}/var/magpie, which is gitignored -- the separation good-citizen enshrines.
DATA_ROOT = Path.home() / "var" / "magpie"
BENTOS_ROOT = DATA_ROOT / "bentos"


def transcribe(audio_path: Path, prompt: str = "") -> str:
    # run whisper large-v3 over the audio and return the raw text. Imported lazily
    # because mlx_whisper pulls in MLX/torch-scale deps we don't want loaded for a
    # plain `magpie --help`. `prompt` is the bento's per-bento context (names,
    # places, terms, what the recording is about); whisper uses it as
    # initial_prompt to bias decoding toward the right proper nouns.
    import mlx_whisper

    logger.info("transcribing %s with %s", audio_path, WHISPER_MODEL)
    kwargs = {"path_or_hf_repo": WHISPER_MODEL}
    if prompt:
        kwargs["initial_prompt"] = prompt
    result = mlx_whisper.transcribe(str(audio_path), **kwargs)
    return result.get("text", "").strip()


def cleanup(raw_text: str) -> str:
    # best-effort artifact cleanup via local mistral. If ollama is unreachable we
    # return the raw transcript rather than fail the run -- a raw transcript beats
    # no transcript (telemetry-never-blocks applied to a processing step).
    try:
        resp = httpx.post(
            OLLAMA_URL + "/api/generate",
            json={"model": CLEANUP_MODEL, "prompt": _CLEANUP_PROMPT + raw_text, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or raw_text
    except Exception as e:  # noqa: BLE001 - any failure degrades to the raw text
        logger.warning("cleanup model unavailable (%s); keeping raw transcript", e)
        return raw_text


def _sidecar_prompt(audio_path: Path) -> str:
    # a per-bento prompt can live in a sidecar file next to the audio:
    # "<name>.prompt.txt" (or .prompt / .prompt.md). Write it in vim; magpie reads
    # it as the bento's prompt. An explicit prompt argument overrides it.
    for suffix in (".prompt.txt", ".prompt", ".prompt.md"):
        cand = audio_path.parent / (audio_path.stem + suffix)
        if cand.is_file():
            return cand.read_text().strip()
    return ""


def process(audio_path: Path, prompt: str = "") -> dict:
    # the full run as a bento: scaffold the dir, copy the source in (dup-over-loss),
    # transcribe -> outputs/transcript.raw.txt, cleanup -> outputs/transcript.txt.
    # Returns a manifest (where the outputs are), NOT the transcript text itself --
    # magpie tells you it found your file and where the result will be, it does not
    # hand back the prose.
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"no such audio file: {audio_path}")

    # an explicit prompt wins; otherwise pick up a sidecar prompt file if present.
    if not prompt:
        prompt = _sidecar_prompt(audio_path)

    bento_id = str(uuid.uuid4())
    bento = BENTOS_ROOT / bento_id
    raw_dir = bento / "raw_data"
    out_dir = bento / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # COPY (never move) the operator's source into the bento.
    archived = raw_dir / audio_path.name
    shutil.copy2(audio_path, archived)

    raw_text = transcribe(archived, prompt=prompt)
    (out_dir / "transcript.raw.txt").write_text(raw_text)

    cleaned = cleanup(raw_text)
    cleaned_path = out_dir / "transcript.txt"
    cleaned_path.write_text(cleaned)

    manifest = {
        "bento_id": bento_id,
        "kind": "voice-memo",
        "source": str(archived),
        "transcript": str(cleaned_path),
        "raw_transcript": str(out_dir / "transcript.raw.txt"),
        "prompt": prompt,
    }
    (bento / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def ffmpeg_available() -> bool:
    # ffmpeg is required for m4a decoding; surface its absence clearly at startup.
    return shutil.which("ffmpeg") is not None or subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0
