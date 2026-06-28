# regression tests for the magpie pipeline.
#
# these pin the failure modes we have actually hit, so they cannot come back:
#
#   - safe_name() must not raise on a filename that needs normalization. The
#     first daemon run crashed here -- safe_name called re.sub but `re` was never
#     imported, so any file with a space (every iOS voice memo) raised NameError
#     and killed the watch thread while the HTTP server stayed up (green /health,
#     inbox never drained). That bug is a single assertion below.
#
#   - cleanup() must DEGRADE to the raw transcript when the local model is down
#     (ollama unreachable or 404), never fail the run. A raw transcript beats no
#     transcript. Mocked -- we never touch a real ollama.
#
#   - process() must scaffold a bento, COPY (never move) the operator's source in
#     (dup-over-loss), and write the manifest + per-stage stats. The heavy/IO
#     stages (whisper, ollama) are mocked so this runs without MLX or a model.

import json
import unicodedata
from pathlib import Path

import pytest
from bento.v1 import bento_pb2

from magpie import pipeline


# --- safe_name: filenames are a source of many sad nights, so probe them hard --

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
        # the everyday iOS shapes
        ("Voice Memo 12.m4a", "voice-memo-12.m4a"),    # spaces collapse, lowercased
        ("a  b   c.WAV", "a-b-c.wav"),                  # runs collapse to one hyphen, ext lowered
        ("weird!!!name???.mp3", "weird-name.mp3"),      # shell-hostile chars -> hyphen
        ("--leading.trailing--.flac", "leading.trailing.flac"),  # strip junk both ends
        ("keep_me.ok-1.aac", "keep_me.ok-1.aac"),       # already-safe chars survive
        # control + invisible characters that have no business in a name
        ("memo\nwith\nnewlines.m4a", "memo-with-newlines.m4a"),  # newlines
        ("tab\tseparated.m4a", "tab-separated.m4a"),    # control whitespace
        ("null\x00byte.m4a", "null-byte.m4a"),          # nul byte
        ("zero​width.m4a", "zero-width.m4a"),      # zero-width space
        ("rtl‮override.m4a", "rtl-override.m4a"),  # bidi override (a filename-spoof trick)
        ("emoji \U0001f600 memo.m4a", "emoji-memo.m4a"),  # astral-plane chars drop out
        # unicode that should fold or hyphenate, not silently vanish or split
        ("café.m4a", "cafe.m4a"),                  # accented latin (NFC) -> ascii base
        ("résumé.m4a", "resume.m4a"),       # combining marks (NFD) fold the same
        ("smart’quote.m4a", "smart-quote.m4a"),    # curly apostrophe -> hyphen
        ("a—b.m4a", "a-b.m4a"),                     # em dash -> hyphen
    ],
)
def test_safe_name_cases(given, expected):
    assert pipeline.safe_name(given) == expected


def test_safe_name_empty_stem_falls_back_to_untitled():
    # a name that normalizes to nothing must not produce an extension-only file.
    assert pipeline.safe_name("???.m4a") == "untitled.m4a"


def test_safe_name_non_latin_falls_back_to_untitled():
    # names with no ascii form fold away to "untitled" -- a naming choice, not data
    # loss: the bento's uuid dir keeps separate recordings separate regardless.
    assert pipeline.safe_name("日本語.m4a") == "untitled.m4a"          # 日本語
    assert pipeline.safe_name("Москва.m4a") == "untitled.m4a"  # Москва


def test_safe_name_is_normalization_stable():
    # the sneaky one: the SAME visible name in NFC vs NFD must slug identically.
    # before the fix "café" gave "caf" (NFC, é dropped whole) but "cafe" (NFD, base
    # e survived) -- one name, two slugs, depending on which app encoded it.
    nfc = unicodedata.normalize("NFC", "café.m4a")
    nfd = unicodedata.normalize("NFD", "café.m4a")
    assert pipeline.safe_name(nfc) == pipeline.safe_name(nfd) == "cafe.m4a"


def test_safe_name_neutralizes_path_traversal():
    # a filename is a name, never a path: directory parts and traversal must not
    # survive, so a crafted memo name cannot escape the inbox or the bento.
    assert pipeline.safe_name("../../etc/passwd.m4a") == "passwd.m4a"
    assert pipeline.safe_name("a/b/c.m4a") == "c.m4a"


def test_safe_name_caps_length():
    # some filesystems reject names over 255 bytes; cap the slug so a pathological
    # name still writes, with the extension preserved.
    out = pipeline.safe_name("a" * 300 + ".m4a")
    assert out.endswith(".m4a")
    assert len(out) - len(".m4a") <= 200


# --- cleanup: must degrade to raw when the model is unavailable -------------

def test_cleanup_degrades_when_model_unavailable(monkeypatch):
    # ModelUnavailable (delightd down, or nothing healthy serves the model) -> keep
    # the raw transcript. This is the fail-closed path of the good-citizen client.
    def boom(*a, **k):
        raise pipeline.model.ModelUnavailable("no backend serves 'mistral'")

    monkeypatch.setattr(pipeline.model, "generate", boom)
    raw = "the raw transcript, kept verbatim"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_degrades_on_model_error(monkeypatch):
    # any other failure mid-generate (a transport error, say) also falls through to
    # the raw transcript rather than failing the run.
    def boom(*a, **k):
        raise RuntimeError("connection reset mid-generate")

    monkeypatch.setattr(pipeline.model, "generate", boom)
    raw = "still here even though generate blew up"
    assert pipeline.cleanup(raw) == raw


def test_cleanup_returns_model_output_on_success(monkeypatch):
    # the happy path: the model answers, we return its cleaned text.
    monkeypatch.setattr(pipeline.model, "generate", lambda *a, **k: "cleaned text")
    assert pipeline.cleanup("dirty text") == "cleaned text"


def test_cleanup_empty_model_output_falls_back_to_raw(monkeypatch):
    # an empty response is not an improvement -- keep the raw transcript.
    monkeypatch.setattr(pipeline.model, "generate", lambda *a, **k: "   ")
    raw = "raw beats blank"
    assert pipeline.cleanup(raw) == raw


# --- process: bento scaffolding + dup-over-loss + manifest ------------------

@pytest.fixture
def mocked_stages(monkeypatch, tmp_path):
    # mock the heavy/IO stages so the orchestration is testable without MLX or a
    # model, and redirect the data root to a tmp dir so we never touch ~/var.
    monkeypatch.setattr(pipeline, "transcribe", lambda audio, prompt="": "raw words")
    monkeypatch.setattr(pipeline, "_clean_or_degrade", lambda raw: ("clean words", False))
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


# --- the FSM: state transitions + the lifecycle-emit seam -------------------

def test_process_emits_cook_then_done_on_clean(mocked_stages):
    # a clean run walks COOK -> DONE; each transition is relayed to the emitter (the
    # seam the sidecar plugs into). NOTICED is the seed state, not a transition.
    seen = []
    pipeline.process(_make_audio(mocked_stages), emitter=lambda b, s: seen.append(s))
    assert seen == [bento_pb2.BENTO_STATE_COOK, bento_pb2.BENTO_STATE_DONE]


def test_process_partial_then_done_when_cleanup_degrades(mocked_stages, monkeypatch):
    # a degraded cleanup (model down) is PARTIAL, then accepted as DONE -- the raw
    # transcript is the deliverable and the manifest records the degrade.
    monkeypatch.setattr(pipeline, "_clean_or_degrade", lambda raw: (raw, True))
    seen = []
    manifest = pipeline.process(_make_audio(mocked_stages), emitter=lambda b, s: seen.append(s))
    assert seen == [
        bento_pb2.BENTO_STATE_COOK,
        bento_pb2.BENTO_STATE_PARTIAL,
        bento_pb2.BENTO_STATE_DONE,
    ]
    assert manifest["degraded"] is True
    assert Path(manifest["transcript"]).read_text() == "raw words"


def test_process_raises_when_transcribe_fails(mocked_stages, monkeypatch):
    # transcription is the one step that cannot degrade: a failure FAILs the bento and
    # process() raises, so the cli/daemon surface it rather than reporting success.
    def boom(audio, prompt=""):
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(pipeline, "transcribe", boom)
    with pytest.raises(RuntimeError, match="whisper exploded"):
        pipeline.process(_make_audio(mocked_stages))
