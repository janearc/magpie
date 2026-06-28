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
import unicodedata
import uuid
from pathlib import Path

from bento.v1 import bento_pb2
from good_citizen import fsm, model

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

# the cleanup model -- a logical name the good-citizen model client resolves through
# service discovery (fail-closed), replacing the interim hand-rolled ollama POST so
# cleanup routes through one model abstraction. The prompt's whole job is to delete
# whisper's loop artifacts (the "okay" x337 / "Let's go." x40 pathology) WITHOUT
# rewriting the words.
CLEANUP_MODEL = "mistral"
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


def _clean_or_degrade(raw_text: str) -> tuple[str, bool]:
    # run cleanup, returning (text, degraded). degraded is True when the model was
    # unavailable (or the call failed) and we kept the raw transcript -- the bento's
    # PARTIAL outcome. on_cook uses the flag to pick DONE vs PARTIAL; cleanup() below is
    # the text-only view for callers that do not care which path produced it.
    try:
        cleaned = model.generate(CLEANUP_MODEL, _CLEANUP_PROMPT + raw_text, timeout=300)
        return (cleaned.strip() or raw_text), False
    except Exception as e:  # noqa: BLE001 - any failure (incl. ModelUnavailable) degrades to raw
        logger.warning("cleanup model unavailable (%s); keeping raw transcript", e)
        return raw_text, True


def cleanup(raw_text: str) -> str:
    # best-effort artifact cleanup through the good-citizen model client (resolves the
    # model via service discovery, fail-closed). Degrades to the raw transcript rather
    # than failing the run -- a raw transcript beats no transcript.
    return _clean_or_degrade(raw_text)[0]


# cap the slug well under the common 255-byte filesystem name limit. the bento's
# uuid dir keeps separate runs separate, so the human-facing name does not have to
# be unique -- only legible and writable everywhere.
_MAX_STEM = 200


def safe_name(filename: str) -> str:
    # magpie normalizes filenames to ascii, lowercase, no spaces or shell-hostile
    # characters; runs of unsafe chars collapse to a single hyphen. iOS hands us
    # spaces -- we do not propagate them. Path(...).stem drops any directory parts,
    # so traversal ("../../x") and separators cannot survive a name.
    p = Path(filename)
    # decompose and drop combining marks so accented latin folds to its base letter
    # (café -> cafe) the SAME way regardless of NFC vs NFD input form -- otherwise the
    # one visible name slugs two different ways and two memos collide differently.
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", p.stem) if not unicodedata.combining(c)
    )
    # anything still outside the safe set -- unicode punctuation, CJK, emoji, control,
    # bidi/zero-width -- collapses to a hyphen; a name with no ascii form (日本語,
    # Москва) falls through to "untitled" (a naming choice, not loss: see the uuid dir).
    stem = re.sub(r"[^a-z0-9._-]+", "-", folded.lower()).strip("-_.") or "untitled"
    # cap length, then re-strip in case the cut left a trailing separator.
    stem = stem[:_MAX_STEM].rstrip("-_.") or "untitled"
    suffix = "".join(
        c for c in unicodedata.normalize("NFKD", p.suffix) if not unicodedata.combining(c)
    ).lower()
    suffix = re.sub(r"[^a-z0-9.]+", "", suffix)
    return stem + suffix


def _sidecar_prompt(audio_path: Path) -> str:
    # a per-bento prompt can live in a sidecar file next to the audio:
    # "<name>.prompt.txt" (or .prompt / .prompt.md). Write it in vim; magpie reads
    # it as the bento's prompt. An explicit prompt argument overrides it.
    for suffix in (".prompt.txt", ".prompt", ".prompt.md"):
        cand = audio_path.parent / (audio_path.stem + suffix)
        if cand.is_file():
            return cand.read_text().strip()
    return ""


# magpie processes RAW AUDIO bentos -- the file class, not the source app. We do not
# care that iOS calls it a "voice memo"; a recording is a recording. A richer bento
# taxonomy across the birbhuis is coming; for now this one constant names the kind, in
# ONE place, so it is never a magic string sprinkled through the handlers.
KIND_RAW_AUDIO = "raw-audio"

# the elements a raw-audio bento is made of, named ONCE here so the kind and its
# banchans are never matched by a bare string out in the handlers.
_BANCHAN_AUDIO = "audio"
_BANCHAN_TRANSCRIPT = "transcript"

# the lifecycle states a magpie bento settles into; past these the FSM halts.
_TERMINAL = {bento_pb2.BENTO_STATE_DONE, bento_pb2.BENTO_STATE_FAILED}


class AudioBento:
    # a thin, behavior-bearing wrapper over a bento_pb2.Bento for raw-audio work. It owns
    # the two things that were ugly and scattered before: PATH composition (the layout
    # lives here, changes in one place, never `root / "raw_data"` inline) and BANCHAN
    # access (asked for by name through methods, never matched by a bare string). The
    # handlers wrap the bento they are handed and ask it for what they need.

    def __init__(self, pb: bento_pb2.Bento) -> None:
        self.pb = pb

    @classmethod
    def new(cls, audio_path: Path, prompt: str) -> "AudioBento":
        # a raw-audio bento in NOTICED. The audio banchan starts at the operator's
        # original file; on_noticed copies it in and repoints the location.
        bento_id = str(uuid.uuid4())
        pb = bento_pb2.Bento(
            id=bento_id,
            name=safe_name(audio_path.name),
            kind=KIND_RAW_AUDIO,
            state=bento_pb2.BENTO_STATE_NOTICED,
            root_path=str(BENTOS_ROOT / bento_id),
            prompt=prompt,
            banchans=[
                bento_pb2.Banchan(
                    guid=str(uuid.uuid4()), name=_BANCHAN_AUDIO, kind="source",
                    location=str(audio_path),
                )
            ],
        )
        return cls(pb)

    # --- paths: composed here, and nowhere else -------------------------------
    @property
    def root(self) -> Path:
        return Path(self.pb.root_path)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw_data"

    @property
    def out_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def raw_transcript_path(self) -> Path:
        return self.out_dir / "transcript.raw.txt"

    @property
    def transcript_path(self) -> Path:
        return self.out_dir / "transcript.txt"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def scaffold(self) -> None:
        # make the bento's directories (idempotent).
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # --- banchans: by name, through methods -----------------------------------
    def banchan(self, name: str) -> bento_pb2.Banchan | None:
        for ban in self.pb.banchans:
            if ban.name == name:
                return ban
        return None

    @property
    def audio(self) -> bento_pb2.Banchan | None:
        return self.banchan(_BANCHAN_AUDIO)

    @property
    def transcript(self) -> bento_pb2.Banchan | None:
        return self.banchan(_BANCHAN_TRANSCRIPT)

    def add_transcript(self, location: Path) -> None:
        self.pb.banchans.append(
            bento_pb2.Banchan(
                guid=str(uuid.uuid4()), name=_BANCHAN_TRANSCRIPT, kind="transcript",
                location=str(location),
            )
        )

    # --- the manifest mirrors the bento: where the outputs ARE, not the prose --
    def manifest(self, *, degraded: bool, error: str, stats: dict) -> dict:
        audio = self.audio
        transcript = self.transcript
        return {
            "bento_id": self.pb.id,
            "kind": self.pb.kind,
            "source": audio.location if audio else "",
            "transcript": transcript.location if transcript else "",
            "raw_transcript": str(self.raw_transcript_path),
            "prompt": self.pb.prompt,
            "degraded": degraded,
            "error": error,
            "stats": stats,
        }

    def write_manifest(self, manifest: dict) -> None:
        if self.root.is_dir():
            self.manifest_path.write_text(json.dumps(manifest, indent=2))


class AudioHandlers(fsm.Handlers):
    # magpie's behavior bound to the bento lifecycle. The stages (transcribe, cleanup)
    # run INSIDE on_cook and never appear on the wire as states -- the wire sees only
    # NOTICED -> COOK -> (DONE | PARTIAL -> DONE | FAILED). One instance per bento; it
    # carries the run's stats + final manifest (the durable outputs live on disk under
    # root_path, so a future distributed handler reads them from there, not from here).

    def __init__(self) -> None:
        self.stats: dict = {}
        self.manifest: dict | None = None
        self.error: str = ""

    def on_noticed(self, b: bento_pb2.Bento) -> int:
        # pre-flight: scaffold the bento dir and COPY the source in (dup-over-loss),
        # repointing the audio banchan at the archived copy. No source -> FAILED.
        bento = AudioBento(b)
        audio = bento.audio
        src = Path(audio.location) if audio else None
        if src is None or not src.is_file():
            self.error = f"no source audio: {src}"
            self._finish(bento)
            return bento_pb2.BENTO_STATE_FAILED
        bento.scaffold()
        archived = bento.raw_dir / b.name
        shutil.copy2(src, archived)
        audio.location = str(archived)
        return bento_pb2.BENTO_STATE_COOK

    def on_cook(self, b: bento_pb2.Bento) -> int:
        # the work. transcribe -> raw; cleanup -> clean (or raw, degraded). A transcribe
        # error FAILs; a degraded cleanup is PARTIAL, a clean one DONE.
        bento = AudioBento(b)
        try:
            with _Stage("transcribe", self.stats):
                raw_text = transcribe(Path(bento.audio.location), prompt=b.prompt)
        except Exception as e:  # noqa: BLE001 - transcription is the step that cannot degrade
            self.error = f"transcribe failed: {e}"
            logger.error("magpie: %s", self.error)
            self._finish(bento)
            return bento_pb2.BENTO_STATE_FAILED
        bento.raw_transcript_path.write_text(raw_text)

        with _Stage("cleanup", self.stats):
            cleaned, degraded = _clean_or_degrade(raw_text)
        bento.transcript_path.write_text(cleaned)
        bento.add_transcript(bento.transcript_path)
        self._finish(bento, degraded=degraded)
        return bento_pb2.BENTO_STATE_PARTIAL if degraded else bento_pb2.BENTO_STATE_DONE

    def on_partial(self, b: bento_pb2.Bento) -> int:
        # degraded cleanup: the raw transcript is written and IS the deliverable (raw
        # beats nothing). Convergence/retry is a substrate concern, not magpie's, so we
        # accept and finish.
        return bento_pb2.BENTO_STATE_DONE

    def on_done(self, b: bento_pb2.Bento) -> int:
        # terminal. Outputs + manifest were written in COOK; nothing to do locally.
        return bento_pb2.BENTO_STATE_UNSPECIFIED

    def on_failed(self, b: bento_pb2.Bento) -> int:
        # terminal. The manifest (with error) was written where the failure occurred.
        return bento_pb2.BENTO_STATE_UNSPECIFIED

    def _finish(self, bento: AudioBento, degraded: bool = False) -> None:
        # assemble + persist the manifest, in one place.
        self.manifest = bento.manifest(degraded=degraded, error=self.error, stats=self.stats)
        bento.write_manifest(self.manifest)


def process(audio_path: Path, prompt: str = "", emitter=None) -> dict:
    # the full run as a bento walked through the generated FSM. Builds a NOTICED bento
    # and drives it to a terminal state; each transition is relayed to `emitter` (the
    # good-citizen sidecar) when one is given -- the CLI passes None (local, no bus).
    # Returns the manifest (where the outputs are), NOT the transcript text. Raises if
    # the bento ends FAILED, so callers (cli, daemon) surface the error as before.
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"no such audio file: {audio_path}")

    safe = safe_name(audio_path.name)
    if safe != audio_path.name:
        logger.warning(
            "magpie: normalizing filename %r -> %r (no spaces/unsafe chars)", audio_path.name, safe
        )
    # an explicit prompt wins; else a sidecar prompt file next to the audio.
    if not prompt:
        prompt = _sidecar_prompt(audio_path.parent / safe) or _sidecar_prompt(audio_path)

    bento = AudioBento.new(audio_path, prompt)
    handlers = AudioHandlers()
    # the bus is the loop in production (one step per consumed event); for a single
    # local run we step until a terminal handler is reached.
    while bento.pb.state not in _TERMINAL:
        prev = bento.pb.state
        fsm.step(handlers, emitter, bento.pb)
        if bento.pb.state == prev:  # a handler that did not advance -- stop rather than spin
            break

    if bento.pb.state == bento_pb2.BENTO_STATE_FAILED:
        raise RuntimeError(handlers.error or "magpie bento failed")
    return handlers.manifest


def ffmpeg_available() -> bool:
    # ffmpeg is required for m4a decoding; surface its absence clearly at startup.
    return shutil.which("ffmpeg") is not None or subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0
