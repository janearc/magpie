# tests for the per-user launchd agent management (magpie's launchd citizen
# self-install, ported faithfully from paling/tests/test_launchagent.py). every
# test pins $HOME to a tmp dir and stubs launchctl so nothing touches the real
# launchd or the real ~/Library. the contract under test is: correct
# user-owned paths, idempotent install/uninstall, and a stable json-shaped
# status struct.
import plistlib

import pytest

from magpie import launchagent


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    # route every Path.home()-derived location into a throwaway tree.
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_launchctl(monkeypatch):
    # pretend launchctl is absent so install/uninstall stay pure-filesystem and
    # never shell out during the unit tests.
    monkeypatch.setattr(launchagent.shutil, "which", lambda _name: None)


def test_resolve_paths_are_user_owned(fake_home):
    paths = launchagent.resolve_paths(home=fake_home)
    # the policy bars system paths: everything must live under $HOME.
    assert str(paths.plist_source).startswith(str(fake_home / "etc" / "magpie"))
    assert str(paths.plist_installed).startswith(str(fake_home / "Library" / "LaunchAgents"))
    assert str(paths.stdout_log).startswith(str(fake_home / "var" / "log"))
    assert str(paths.stderr_log).startswith(str(fake_home / "var" / "log"))


def test_render_plist_has_keepalive_and_runatload(fake_home):
    paths = launchagent.resolve_paths(home=fake_home, uv_bin="/some/bin/uv")
    job = plistlib.loads(launchagent.render_plist(paths, host="127.0.0.1", port=8092))
    assert job["KeepAlive"] is True
    assert job["RunAtLoad"] is True
    assert job["Label"] == launchagent.DEFAULT_LABEL
    # absolute program path is mandatory under launchd's minimal env.
    assert job["ProgramArguments"][0] == "/some/bin/uv"
    assert job["ProgramArguments"][1:] == [
        "run", "magpie", "serve", "--host", "127.0.0.1", "--port", "8092",
    ]
    assert "/some/bin" in job["EnvironmentVariables"]["PATH"]


def test_install_writes_plist_and_symlink(fake_home, no_launchctl):
    result = launchagent.install(home=fake_home, uv_bin="/some/bin/uv")
    assert result.installed is True
    src = fake_home / "etc" / "magpie" / f"{launchagent.DEFAULT_LABEL}.plist"
    link = fake_home / "Library" / "LaunchAgents" / f"{launchagent.DEFAULT_LABEL}.plist"
    assert src.is_file()
    assert link.is_symlink()
    assert link.resolve() == src.resolve()
    # log dir is provisioned so launchd has somewhere to write.
    assert (fake_home / "var" / "log").is_dir()


def test_install_is_idempotent(fake_home, no_launchctl):
    first = launchagent.install(home=fake_home, uv_bin="/some/bin/uv")
    second = launchagent.install(home=fake_home, uv_bin="/some/bin/uv")
    # re-running must not raise and must converge to the same installed state.
    assert first.installed and second.installed
    link = fake_home / "Library" / "LaunchAgents" / f"{launchagent.DEFAULT_LABEL}.plist"
    assert link.is_symlink()


def test_uninstall_removes_symlink(fake_home, no_launchctl):
    launchagent.install(home=fake_home, uv_bin="/some/bin/uv")
    result = launchagent.uninstall(home=fake_home)
    link = fake_home / "Library" / "LaunchAgents" / f"{launchagent.DEFAULT_LABEL}.plist"
    assert result.installed is False
    assert result.action == "uninstalled"
    assert not link.exists()
    # source-of-truth plist is intentionally retained as a record.
    assert (fake_home / "etc" / "magpie" / f"{launchagent.DEFAULT_LABEL}.plist").is_file()


def test_uninstall_absent_is_noop(fake_home, no_launchctl):
    result = launchagent.uninstall(home=fake_home)
    assert result.installed is False
    assert result.action == "already-absent"


def test_status_reports_installed(fake_home, no_launchctl):
    before = launchagent.status(home=fake_home)
    assert before.installed is False
    launchagent.install(home=fake_home, uv_bin="/some/bin/uv")
    after = launchagent.status(home=fake_home)
    assert after.installed is True
    assert after.action == "status"


def test_install_reports_when_launchctl_missing(fake_home, no_launchctl):
    # with launchctl absent, load is requested but cannot happen: the plist is
    # still written and the status says so explicitly rather than lying.
    result = launchagent.install(home=fake_home, uv_bin="/some/bin/uv", load=True)
    assert result.installed is True
    assert result.loaded is False
    assert result.action == "installed-not-loaded"


def test_install_loads_via_launchctl(fake_home, monkeypatch):
    # simulate launchctl present and a successful load, asserting the load/list
    # handshake the install relies on.
    monkeypatch.setattr(launchagent.shutil, "which", lambda _name: "/bin/launchctl")
    calls = []

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(args, **_kwargs):
        calls.append(args)
        return _Result()

    monkeypatch.setattr(launchagent.subprocess, "run", fake_run)
    result = launchagent.install(home=fake_home, uv_bin="/some/bin/uv", load=True)
    assert result.loaded is True
    # a load command must have been issued against the installed plist.
    assert any("load" in c for c in calls)
