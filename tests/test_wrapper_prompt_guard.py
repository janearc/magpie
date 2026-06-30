# the skill wrapper's {prompt} guard, pinned so it cannot regress.
#
# the mcp.json `transcribe` handler templates `{prompt}` into the wrapper's argv. when the
# caller OMITS the prompt, the MCP harness leaves that token unsubstituted -- the wrapper
# sees either an empty string or the literal "{prompt}". neither is a real prompt, and
# neither may reach whisper: `--prompt ""` and `--prompt "{prompt}"` would both bias
# decoding toward junk. the wrapper MUST drop `--prompt` entirely in those cases, and pass
# it through only for a genuine prompt.
#
# we prove this hermetically: a shim stands in for the `magpie` CLI (via MAGPIE_BIN) and
# records the exact argv the wrapper forwarded. no MLX, no daemon, no network.

import subprocess
from pathlib import Path

import pytest

# the wrapper under test: <repo>/skill/magpie. this file is <repo>/tests/, so two up.
WRAPPER = Path(__file__).resolve().parent.parent / "skill" / "magpie"


@pytest.fixture
def shim(tmp_path):
    # a fake `magpie` that appends each received arg (one per line) to a sentinel file, so
    # the test can read back exactly what the wrapper forwarded. returns (bin, args_out).
    args_out = tmp_path / "argv.txt"
    fake = tmp_path / "magpie"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" >> "$MAGPIE_ARGS_OUT"\n'
    )
    fake.chmod(0o755)
    return fake, args_out


def _run(file_arg, prompt_arg, shim):
    # invoke the wrapper's transcribe through the shim and return the forwarded argv list.
    fake, args_out = shim
    argv = [str(WRAPPER), "transcribe", file_arg]
    if prompt_arg is not None:
        argv.append(prompt_arg)
    env = {
        "MAGPIE_BIN": str(fake),
        "MAGPIE_ARGS_OUT": str(args_out),
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run(argv, env=env, check=True)
    return args_out.read_text().splitlines()


def test_omitted_prompt_passes_no_junk(shim):
    # no prompt arg at all (the common case) -> the no-prompt path, no `--prompt`.
    forwarded = _run("memo.m4a", None, shim)
    assert forwarded == ["transcribe", "memo.m4a"]
    assert "--prompt" not in forwarded


def test_literal_placeholder_prompt_passes_no_junk(shim):
    # the harness left `{prompt}` unsubstituted -> still the no-prompt path. THIS is the
    # MUST-CLOSE edge: the literal "{prompt}" must never reach whisper.
    forwarded = _run("memo.m4a", "{prompt}", shim)
    assert forwarded == ["transcribe", "memo.m4a"]
    assert "--prompt" not in forwarded
    assert "{prompt}" not in forwarded


def test_empty_prompt_passes_no_junk(shim):
    # an explicitly empty prompt is also "no prompt" -- no `--prompt ""`.
    forwarded = _run("memo.m4a", "", shim)
    assert forwarded == ["transcribe", "memo.m4a"]
    assert "--prompt" not in forwarded


def test_real_prompt_is_forwarded(shim):
    # a genuine prompt DOES reach whisper, as `--prompt <value>`.
    forwarded = _run("memo.m4a", "names: Will, Rae", shim)
    assert forwarded == ["transcribe", "memo.m4a", "--prompt", "names: Will, Rae"]
