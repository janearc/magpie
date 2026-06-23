# the magpie pipeline: a voice-memo m4a in, a clean transcript out.
#
# a run is a bento (the shared unit of work): a directory holding the source audio
# under raw_data/ and the transcripts under outputs/. The steps are banchans --
# transcribe, then cleanup -- and each one's lifecycle is what we eventually emit to
# the bus via the good-citizen sidecar (the next wiring step).
#
# archival rule (Max): duplicates over loss. We COPY the source audio into the
# bento; we never move or delete the operator's original.

import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class _Stage:
    # times a pipeline stage into a stats dict, and logs it.
    #
    # records wall and CPU seconds. Wall is the signal that matters for fan/heat:
    # GPU work (whisper on MLX) barely moves CPU time but pegs the device for the
    # whole wall duration, so the longest-wall stage is the one cooking the laptop.
    # these per-stage stats are what magpie reports as a good citizen (and will emit
    # to the bus via the sidecar).

    def __init__(self, name: str, stats: dict):
        self.name, self.stats = name, stats

    def __enter__(self):
        self._w = time.monotonic()
        self._c = time.process_time()
        return self

    def __exit__(self, *exc):
        wall = round(time.monotonic() - self._w, 2)
        cpu = round(time.process_time() - self._c, 2)
        self.stats[self.name] = {"wall_s": wall, "cpu_s": cpu}
        logger.info("stage %s: wall=%.1fs cpu=%.1fs", self.name, wall, cpu)

# whisper large-v3 (MLX) -- mlx_whisper resolves this from the shared, read-only
# HF cache; we do not vendor weights. ffmpeg (a hard dep of mlx_whisper) decodes
# the m4a. The model id is a logical name; the good-citizen model abstraction
# will own this resolution later, the same way it does for mistral/flan.
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# the cleanup model. mistral via the local ollama, kept in-enclave. This direct
# call is INTERIM -- it is exactly the seam the good-citizen model-client
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


def safe_name(filename: str) -> str:
    # magpie normalizes filenames: no spaces or shell-hostile characters. Lowercase,
    # runs of unsafe chars collapse to a single hyphen; the extension is preserved
    # lowercased. iOS voice memos arrive with spaces -- we do not propagate that.
    p = Path(filename)
    stem = re.sub(r"[^a-z0-9._-]+", "-", p.stem.lower()).strip("-_.") or "untitled"
    return stem + p.suffix.lower()


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

    # magpie normalizes filenames: declare it, and archive under a safe name.
    safe = safe_name(audio_path.name)
    if safe != audio_path.name:
        logger.warning("magpie: normalizing filename %r -> %r (no spaces/unsafe chars)", audio_path.name, safe)

    # an explicit prompt wins; else a sidecar prompt file -- match the safe/slug
    # name Max actually writes, falling back to the original.
    if not prompt:
        prompt = _sidecar_prompt(audio_path.parent / safe) or _sidecar_prompt(audio_path)

    bento_id = str(uuid.uuid4())
    bento = BENTOS_ROOT / bento_id
    raw_dir = bento / "raw_data"
    out_dir = bento / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # COPY (never move) the operator's source into the bento, under the safe name.
    archived = raw_dir / safe
    shutil.copy2(audio_path, archived)

    stats: dict = {}
    with _Stage("transcribe", stats):
        raw_text = transcribe(archived, prompt=prompt)
    (out_dir / "transcript.raw.txt").write_text(raw_text)

    with _Stage("cleanup", stats):
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
        "stats": stats,
    }
    (bento / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def ffmpeg_available() -> bool:
    # ffmpeg is required for m4a decoding; surface its absence clearly at startup.
    return shutil.which("ffmpeg") is not None or subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0
