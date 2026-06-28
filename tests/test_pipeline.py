# regression tests for the magpie pipeline, now a thin birb on birblib.
#
# these pin the failure modes we have actually hit, so they cannot come back:
#
#   - safe_name() must not raise on a filename that needs normalization. The first daemon
#     run crashed here -- safe_name called re.sub but `re` was never imported, so any file
#     with a space (every iOS voice memo) raised NameError and killed the watch thread
#     while the HTTP server stayed up (green /health, inbox never drained). safe_name now
#     lives in birblib.names (its exhaustive case suite is birblib's test_names.py); the
#     one smoke assertion below pins that magpie is still wired to it.
#
#   - cleanup() must DEGRADE to the raw transcript when the local model is down (ollama
#     unreachable or 404), never fail the run. A raw transcript beats no transcript.
#
#   - process() must scaffold a bento, COPY (never move) the operator's source in
#     (dup-over-loss), and write the birblib manifest envelope + per-stage stats. The
#     heavy/IO stages (whisper, ollama) are mocked so this runs without MLX or a model.

import json
from pathlib import Path

import pytest
from bento.v1 import bento_pb2

from magpie import pipeline


# --- safe_name: wired to birblib (exhaustive cases live in birblib.test_names) ---

def test_safe_name_is_wired_and_normalizes():
    # the literal file that crashed the daemon. magpie re-exports birblib.names.safe_name;
    # this pins that the wiring is intact and the normalization still holds.
    assert (
        pipeline.safe_name("freezer food and handsome man.m4a")
        == "freezer-food-and-handsome-man.m4a"
    )


# --- cleanup: must degrade to raw when the model is unavailable -------------

def test_cleanup_degrades_when_model_unavailable(monkeypatch):
    # ModelUnavailable (delightd down, or nothing healthy serves the model) -> keep the raw
    # transcript. This is the fail-closed path of the good-citizen client.
    def boom(*a, **k):
        raise pipeline.model.ModelUnavailable("no backend serves 'mistral'")

    monkeypatch.setattr(pipeline.model, "generate", boom)
    raw = "the raw transcript, kept verbatim"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_degrades_on_model_error(monkeypatch):
    # any other failure mid-generate (a transport error, say) also falls through to the raw
    # transcript rather than failing the run.
    def boom(*a, **k):
        raise RuntimeError("connection reset mid-generate")

    monkeypatch.setattr(pipeline.model, "generate", boom)
    raw = "still here even though generate blew up"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_returns_model_output_on_success(monkeypatch):
    monkeypatch.setattr(pipeline.model, "generate", lambda *a, **k: "cleaned text")
    assert pipeline.cleanup("dirty text") == "cleaned text"


def test_cleanup_empty_model_output_falls_back_to_raw(monkeypatch):
    monkeypatch.setattr(pipeline.model, "generate", lambda *a, **k: "   ")
    raw = "raw beats blank"
    assert pipeline.cleanup(raw) == raw


# --- process: bento scaffolding + dup-over-loss + the manifest envelope ------

@pytest.fixture
def mocked_stages(monkeypatch, tmp_path):
    # mock the heavy/IO stages so the orchestration is testable without MLX or a model, and
    # redirect the data root to a tmp dir so we never touch ~/var.
    monkeypatch.setattr(pipeline, "transcribe", lambda audio, prompt="": "raw words")
    monkeypatch.setattr(pipeline, "_clean_or_degrade", lambda raw: ("clean words", False))
    monkeypatch.setattr(pipeline, "BENTOS_ROOT", tmp_path / "bentos")
    return tmp_path


def _make_audio(tmp_path: Path, name: str = "a memo.m4a") -> Path:
    src = tmp_path / name
    src.write_bytes(b"not really audio, but a real file on disk")
    return src


def test_process_copies_source_and_does_not_move_it(mocked_stages):
    # dup-over-loss: the operator's original must still exist afterward, and the archived
    # copy lands under the bento's raw_data (recorded in the manifest's detail.source).
    src = _make_audio(mocked_stages)
    manifest = pipeline.process(src)
    assert src.exists(), "process() must COPY the source, never move/delete it"
    archived = Path(manifest.detail["source"])
    assert archived.exists()
    assert archived.read_bytes() == src.read_bytes()
    assert "raw_data" in str(archived)


def test_process_normalizes_archived_filename(mocked_stages):
    # the archived copy lands under the safe name even though the source had a space.
    src = _make_audio(mocked_stages, "a memo.m4a")
    manifest = pipeline.process(src)
    assert Path(manifest.detail["source"]).name == "a-memo.m4a"


def test_process_writes_the_manifest_envelope(mocked_stages):
    src = _make_audio(mocked_stages)
    manifest = pipeline.process(src)

    # the birblib envelope: ok is the single success signal, the transcript is the artifact.
    assert manifest.ok is True
    assert manifest.state == "DONE"
    assert manifest.kind == pipeline.KIND_RAW_AUDIO
    raw = Path(manifest.detail["raw_transcript"])
    clean = Path(manifest.artifact)
    assert raw.read_text() == "raw words"
    assert clean.read_text() == "clean words"

    # the manifest is persisted next to the outputs and carries per-stage stats.
    bento = clean.parent.parent
    on_disk = json.loads((bento / "manifest.json").read_text())
    assert on_disk["bento_id"] == manifest.bento_id
    assert set(on_disk["stats"]) == {"transcribe", "cleanup"}
    assert "wall_s" in on_disk["stats"]["transcribe"]


def test_process_persists_the_bento_on_disk(mocked_stages):
    # the on-disk SOT birblib adds: the bento itself, recoverable without the bus.
    src = _make_audio(mocked_stages)
    manifest = pipeline.process(src)
    bento_dir = Path(manifest.artifact).parent.parent
    on_disk = json.loads((bento_dir / "bento.json").read_text())
    assert on_disk["id"] == manifest.bento_id
    assert on_disk["kind"] == pipeline.KIND_RAW_AUDIO


def test_process_uses_sidecar_prompt(mocked_stages, monkeypatch):
    # a "<name>.prompt.txt" next to the audio becomes the bento prompt, is passed through
    # to whisper, and lands under the manifest's params.
    src = _make_audio(mocked_stages, "memo.m4a")
    (src.parent / "memo.prompt.txt").write_text("names: Will, Rae")

    seen = {}
    monkeypatch.setattr(
        pipeline, "transcribe",
        lambda audio, prompt="": seen.setdefault("prompt", prompt) or "raw words",
    )
    manifest = pipeline.process(src)
    assert seen["prompt"] == "names: Will, Rae"
    # the prompt is archived to request.json by the base and surfaced under params.
    assert manifest.params["prompt"] == "names: Will, Rae"


def test_process_raises_on_missing_file(mocked_stages):
    with pytest.raises(FileNotFoundError):
        pipeline.process(mocked_stages / "does-not-exist.m4a")


# --- the FSM: state transitions + the lifecycle-emit seam -------------------

def test_process_emits_cook_then_done_on_clean(mocked_stages):
    # a clean run walks COOK -> DONE; each transition is relayed to the emitter (the seam
    # the sidecar plugs into). NOTICED is the seed state, not a transition.
    seen = []
    pipeline.process(_make_audio(mocked_stages), emitter=lambda b, s: seen.append(s))
    assert seen == [bento_pb2.BENTO_STATE_COOK, bento_pb2.BENTO_STATE_DONE]


def test_process_partial_then_done_when_cleanup_degrades(mocked_stages, monkeypatch):
    # a degraded cleanup (model down) is PARTIAL, then accepted as DONE -- the raw
    # transcript is the deliverable and the manifest records the degrade in detail.
    monkeypatch.setattr(pipeline, "_clean_or_degrade", lambda raw: (raw, True))
    seen = []
    manifest = pipeline.process(_make_audio(mocked_stages), emitter=lambda b, s: seen.append(s))
    assert seen == [
        bento_pb2.BENTO_STATE_COOK,
        bento_pb2.BENTO_STATE_PARTIAL,
        bento_pb2.BENTO_STATE_DONE,
    ]
    assert manifest.ok is True  # raw beats nothing: the run is usable
    assert manifest.detail["degraded"] is True
    assert Path(manifest.artifact).read_text() == "raw words"


def test_process_raises_when_transcribe_fails(mocked_stages, monkeypatch):
    # transcription is the one step that cannot degrade: a failure FAILs the bento and
    # process() raises, so the cli/daemon surface it rather than reporting success.
    def boom(audio, prompt=""):
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    with pytest.raises(RuntimeError, match="whisper exploded"):
        pipeline.process(_make_audio(mocked_stages))


# --- AudioBento: paths and banchans compose in ONE place --------------------

def test_audio_bento_composes_paths_under_root(mocked_stages):
    # every path derives from root_path; the layout is the wrapper's business, not the
    # handlers'. AudioBento adds the audio-specific output paths over BirbBento's base.
    bento = pipeline.AudioBento.new(_make_audio(mocked_stages), prompt="")
    root = bento.root
    assert bento.raw_dir == root / "raw_data"
    assert bento.out_dir == root / "outputs"
    assert bento.raw_transcript_path == root / "outputs" / "transcript.raw.txt"
    assert bento.transcript_path == root / "outputs" / "transcript.txt"
    assert bento.manifest_path == root / "manifest.json"


def test_audio_bento_banchans_by_name_not_bare_string(mocked_stages):
    # the kind's elements are reached through methods; the transcript banchan does not
    # exist until it is produced, then add_transcript declares it.
    bento = pipeline.AudioBento.new(_make_audio(mocked_stages), prompt="")
    assert bento.pb.kind == pipeline.KIND_RAW_AUDIO
    assert bento.audio is not None
    assert bento.transcript is None
    bento.add_transcript(bento.transcript_path)
    assert bento.transcript is not None
    assert bento.transcript.location == str(bento.transcript_path)
