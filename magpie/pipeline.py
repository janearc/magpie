# the magpie pipeline: a voice-memo m4a in, a clean transcript out.
#
# a run is a bento (the shared unit of work): a directory holding the source audio under
# raw_data/ and the transcripts under outputs/. magpie is now a thin birb -- it declares
# its kind and banchans and implements ONE method, cook(); birblib (BirbBento /
# BirbHandlers) owns the scaffold, the dup-over-loss pre-flight, the FSM walk, the manifest
# envelope, and the lifecycle emit. what is left here is magpie's domain: transcribe with
# whisper, clean with a model (degrade to raw if it is down).
#
# archival rule (Max): duplicates over loss. birblib COPIES the source audio into the
# bento; the operator's original is never moved or deleted.

import logging
import shutil
import subprocess
from pathlib import Path

from birblib import BirbBento, BirbHandlers, CookResult, Manifest, Stage, driver, safe_name
from good_citizen import model

logger = logging.getLogger(__name__)

# whisper large-v3 (MLX) -- mlx_whisper resolves this from the shared, read-only HF cache;
# we do not vendor weights. ffmpeg (a hard dep of mlx_whisper) decodes the m4a.
WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"

# the cleanup model -- a logical name the good-citizen model client resolves through
# service discovery (fail-closed). The prompt's whole job is to delete whisper's loop
# artifacts (the "okay" x337 / "Let's go." x40 pathology) WITHOUT rewriting the words.
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

# magpie processes RAW AUDIO bentos -- the file class, not the source app. The kind and its
# banchans are named ONCE here, so they are never a bare string out in the handlers.
KIND_RAW_AUDIO = "raw-audio"
_BANCHAN_AUDIO = "audio"
_BANCHAN_TRANSCRIPT = "transcript"


def transcribe(audio_path: Path, prompt: str = "") -> str:
    # run whisper large-v3 over the audio and return the raw text. Imported lazily because
    # mlx_whisper pulls in MLX/torch-scale deps we don't want loaded for a plain
    # `magpie --help`. `prompt` is the bento's per-bento context (names, places, terms);
    # whisper uses it as initial_prompt to bias decoding toward the right proper nouns.
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
    # PARTIAL outcome. cook() uses the flag to pick DONE vs PARTIAL; cleanup() below is the
    # text-only view for callers that do not care which path produced it.
    try:
        cleaned = model.generate(CLEANUP_MODEL, _CLEANUP_PROMPT + raw_text, timeout=300)
        return (cleaned.strip() or raw_text), False
    except Exception as e:  # noqa: BLE001 - any failure (incl. ModelUnavailable) degrades to raw
        logger.warning("cleanup model unavailable (%s); keeping raw transcript", e)
        return raw_text, True


def cleanup(raw_text: str) -> str:
    # best-effort artifact cleanup through the good-citizen model client (resolves the model
    # via service discovery, fail-closed). Degrades to the raw transcript rather than
    # failing the run -- a raw transcript beats no transcript.
    return _clean_or_degrade(raw_text)[0]


def _sidecar_prompt(audio_path: Path) -> str:
    # a per-bento prompt can live in a sidecar file next to the audio: "<name>.prompt.txt"
    # (or .prompt / .prompt.md). Write it in vim; magpie reads it as the bento's prompt. An
    # explicit prompt argument overrides it.
    for suffix in (".prompt.txt", ".prompt", ".prompt.md"):
        cand = audio_path.parent / (audio_path.stem + suffix)
        if cand.is_file():
            return cand.read_text().strip()
    return ""


class AudioBento(BirbBento):
    # magpie's typed view over a BirbBento: the audio in and the transcript out, reached by
    # name, plus the two output paths magpie writes. the base owns path composition,
    # banchan access, scaffold, and persistence; this adds only the audio-specific names.

    @classmethod
    def new(cls, audio_path: Path, prompt: str = "") -> "AudioBento":
        # a raw-audio bento in NOTICED. the audio banchan starts at the operator's original
        # file; on_noticed copies it in (dup-over-loss) and repoints the location.
        return super().new(
            kind=KIND_RAW_AUDIO,
            bentos_root=BENTOS_ROOT,
            name=safe_name(audio_path.name),
            prompt=prompt,
            banchans=[(_BANCHAN_AUDIO, "source", audio_path)],
        )

    @property
    def audio(self):
        return self.banchan(_BANCHAN_AUDIO)

    @property
    def transcript(self):
        return self.banchan(_BANCHAN_TRANSCRIPT)

    @property
    def raw_transcript_path(self) -> Path:
        return self.out_dir / "transcript.raw.txt"

    @property
    def transcript_path(self) -> Path:
        return self.out_dir / "transcript.txt"

    def add_transcript(self, location) -> None:
        # declare the transcript element (used directly in tests; in the handler flow
        # birblib records the artifact banchan from the CookResult).
        self.add(_BANCHAN_TRANSCRIPT, "transcript", location)


class AudioHandlers(BirbHandlers):
    # magpie bound to the bento lifecycle: declare the kind and the source/artifact
    # banchans, implement cook(). birblib drives NOTICED -> COOK -> (DONE | PARTIAL -> DONE
    # | FAILED); the stages (transcribe, cleanup) run INSIDE cook() and never appear on the
    # wire as states.
    kind = KIND_RAW_AUDIO
    bento_cls = AudioBento
    source_banchan = _BANCHAN_AUDIO
    artifact_banchan = _BANCHAN_TRANSCRIPT

    def cook(self, b) -> CookResult:
        # the work. transcribe -> raw; cleanup -> clean (or raw, degraded). a transcribe
        # error raises (the step that cannot degrade) and birblib FAILs the bento; a
        # degraded cleanup returns ok=False (PARTIAL), a clean one ok=True (DONE).
        bento = AudioBento(b)
        with Stage("transcribe", self.stats):
            raw_text = transcribe(Path(bento.audio.location), prompt=b.prompt)
        bento.raw_transcript_path.write_text(raw_text)

        with Stage("cleanup", self.stats):
            cleaned, degraded = _clean_or_degrade(raw_text)
        bento.transcript_path.write_text(cleaned)
        return CookResult(
            artifact=str(bento.transcript_path),
            ok=not degraded,
            detail={
                "degraded": degraded,
                "source": bento.audio.location,
                "raw_transcript": str(bento.raw_transcript_path),
            },
        )

    # magpie's resolved request is just the per-bento prompt that biased whisper -- which
    # is exactly birblib's default request(), so on_noticed archives it to request.json and
    # params() surfaces it under the manifest with no override here.


def process(audio_path: Path, prompt: str = "", emitter=None) -> Manifest:
    # the full run as a bento walked through the generated FSM (via birblib.driver). Builds
    # a NOTICED bento and drives it to a terminal state; each transition is relayed to
    # `emitter` (the good-citizen sidecar) when one is given -- the CLI passes None (local,
    # no bus). Returns the manifest (where the outputs are), NOT the transcript text. Raises
    # if the bento ends FAILED, so callers (cli, daemon) surface the error.
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
    return driver.run(AudioHandlers(), bento, emitter=emitter)


def ffmpeg_available() -> bool:
    # ffmpeg is required for m4a decoding; surface its absence clearly at startup.
    return shutil.which("ffmpeg") is not None or subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0
