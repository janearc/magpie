"""Regression tests for the magpie pipeline.

These pin the failure modes we have actually hit, so they cannot come back:

  - safe_name() must not raise on a filename that needs normalization. The
    first daemon run crashed here -- safe_name called re.sub but `re` was never
    imported, so any file with a space (every iOS voice memo) raised NameError
    and killed the watch thread while the HTTP server stayed up (green /health,
    inbox never drained). That bug is a single assertion below.

  - cleanup() must DEGRADE to the raw transcript when the local model is down
    (ollama unreachable or 404), never fail the run. A raw transcript beats no
    transcript. Mocked -- we never touch a real ollama.

  - process() must scaffold a bento, COPY (never move) the operator's source in
    (dup-over-loss), and write the manifest + per-stage stats. The heavy/IO
    stages (whisper, ollama) are mocked so this runs without MLX or a model.
"""

import json
from pathlib import Path

import pytest

from magpie import pipeline


# --- safe_name: the exact regression for the missing `re` import ------------

def test_safe_name_normalizes_spaces():
    # the literal file that crashed the daemon. On the unfixed code (no `import
    # re`) this call raised NameError before it could return anything.
    assert (
        pipeline.safe_name("freezer food and handsome man.m4a")
        == "freezer-food-and-handsome-man.m4a"
    )


@pytest.mark.parametrize(
    "given, expected",
    [
        ("Voice Memo 12.m4a", "voice-memo-12.m4a"),   # spaces collapse, lowercased
        ("a  b   c.WAV", "a-b-c.wav"),                 # runs collapse to one hyphen, ext lowered
        ("weird!!!name???.mp3", "weird-name.mp3"),     # shell-hostile chars -> hyphen
        ("--leading.trailing--.flac", "leading.trailing.flac"),  # strip junk from both ends of the stem
        ("keep_me.ok-1.aac", "keep_me.ok-1.aac"),      # already-safe chars survive
    ],
)
def test_safe_name_cases(given, expected):
    assert pipeline.safe_name(given) == expected


def test_safe_name_empty_stem_falls_back_to_untitled():
    # a name that normalizes to nothing must not produce an extension-only file.
    assert pipeline.safe_name("???.m4a") == "untitled.m4a"


# --- cleanup: must degrade to raw when the model is unavailable -------------

def test_cleanup_degrades_when_ollama_unreachable(monkeypatch):
    # connection error (ollama not running at all) -> keep the raw transcript.
    def boom(*a, **k):
        raise pipeline.httpx.ConnectError("connection refused")

    monkeypatch.setattr(pipeline.httpx, "post", boom)
    raw = "the raw transcript, kept verbatim"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_degrades_on_http_error(monkeypatch):
    # the 404 we actually saw (/api/generate missing / model not pulled).
    # raise_for_status() raises -> we fall through to the raw transcript.
    class Resp:
        def raise_for_status(self):
            raise pipeline.httpx.HTTPStatusError("404", request=None, response=None)

        def json(self):  # pragma: no cover - never reached when status raises
            return {}

    monkeypatch.setattr(pipeline.httpx, "post", lambda *a, **k: Resp())
    raw = "still here even though the model 404'd"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_returns_model_output_on_success(monkeypatch):
    # the happy path: the model answers, we return its cleaned text.
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "cleaned text"}

    monkeypatch.setattr(pipeline.httpx, "post", lambda *a, **k: Resp())
    assert pipeline.cleanup("dirty text") == "cleaned text"


def test_cleanup_empty_model_output_falls_back_to_raw(monkeypatch):
    # an empty response is not an improvement -- keep the raw transcript.
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "   "}

    monkeypatch.setattr(pipeline.httpx, "post", lambda *a, **k: Resp())
    raw = "raw beats blank"
    assert pipeline.cleanup(raw) == raw


# --- process: bento scaffolding + dup-over-loss + manifest ------------------

@pytest.fixture
def mocked_stages(monkeypatch, tmp_path):
    # mock the heavy/IO stages so the orchestration is testable without MLX or a
    # model, and redirect the data root to a tmp dir so we never touch ~/var.
    monkeypatch.setattr(pipeline, "transcribe", lambda audio, prompt="": "raw words")
    monkeypatch.setattr(pipeline, "cleanup", lambda raw: "clean words")
    monkeypatch.setattr(pipeline, "BENTOS_ROOT", tmp_path / "bentos")
    return tmp_path


def _make_audio(tmp_path: Path, name: str = "a memo.m4a") -> Path:
    src = tmp_path / name
    src.write_bytes(b"not really audio, but a real file on disk")
    return src


def test_process_copies_source_and_does_not_move_it(mocked_stages):
    # dup-over-loss: the operator's original must still exist afterward.
    src = _make_audio(mocked_stages)
    manifest = pipeline.process(src)
    assert src.exists(), "process() must COPY the source, never move/delete it"
    archived = Path(manifest["source"])
    assert archived.exists()
    assert archived.read_bytes() == src.read_bytes()


def test_process_normalizes_archived_filename(mocked_stages):
    # the archived copy lands under the safe name even though the source had a space.
    src = _make_audio(mocked_stages, "a memo.m4a")
    manifest = pipeline.process(src)
    assert Path(manifest["source"]).name == "a-memo.m4a"


def test_process_writes_transcripts_and_manifest(mocked_stages):
    src = _make_audio(mocked_stages)
    manifest = pipeline.process(src)

    raw = Path(manifest["raw_transcript"])
    clean = Path(manifest["transcript"])
    assert raw.read_text() == "raw words"
    assert clean.read_text() == "clean words"

    # the manifest is persisted next to the outputs and carries per-stage stats.
    bento = clean.parent.parent
    on_disk = json.loads((bento / "manifest.json").read_text())
    assert on_disk["bento_id"] == manifest["bento_id"]
    assert set(on_disk["stats"]) == {"transcribe", "cleanup"}
    assert "wall_s" in on_disk["stats"]["transcribe"]


def test_process_uses_sidecar_prompt(mocked_stages, monkeypatch):
    # a "<name>.prompt.txt" next to the audio becomes the bento prompt, and is
    # passed through to whisper. We capture what transcribe() received.
    src = _make_audio(mocked_stages, "memo.m4a")
    (src.parent / "memo.prompt.txt").write_text("names: Will, Rae")

    seen = {}
    monkeypatch.setattr(
        pipeline, "transcribe",
        lambda audio, prompt="": seen.setdefault("prompt", prompt) or "raw words",
    )
    manifest = pipeline.process(src)
    assert seen["prompt"] == "names: Will, Rae"
    assert manifest["prompt"] == "names: Will, Rae"


def test_process_raises_on_missing_file(mocked_stages):
    with pytest.raises(FileNotFoundError):
        pipeline.process(mocked_stages / "does-not-exist.m4a")
