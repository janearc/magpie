# ==============================================================================
# Magpie LaunchAgent management (`magpie launchagent ...`)
# ==============================================================================
# `magpie serve` runs bare-metal: it drives mlx_whisper on Apple Silicon (Metal)
# and so cannot be containerized like the rest of the fleet. delightd's roster
# marks magpie's deploy.kind as launchd for exactly this reason, and the fleet
# expects every launchd citizen to self-install via `<tool> launchagent install`
# (the pattern this module ports, faithfully, from paling/paling/launchagent.py).
#
# This module renders, installs, and removes a per-user LaunchAgent (NOT a
# system-wide LaunchDaemon: no root, no /Library writes -- it stays inside the
# user-owned paths the agent_behavior_policy mandates). KeepAlive + RunAtLoad
# give the self-healing/always-up contract; install/uninstall are idempotent so
# re-running is a no-op rather than an error.
#
# Layout (all user-owned, per the var-directory standard):
#   plist source-of-truth : ~/etc/magpie/<label>.plist
#   launchd-loaded copy   : ~/Library/LaunchAgents/<label>.plist  (symlink)
#   stdout / stderr       : ~/var/log/magpie.out.log / magpie.err.log
#
# Every entry point returns a plain dict so the CLI can emit JSON by default
# (the agent-first mandate): a caller never has to scrape human text.
# ==============================================================================
import os
import plistlib
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# the reverse-dns label launchd keys the job on. neutral + fleet-namespaced;
# overridable for blue/green (magpie-dev) side-by-side installs.
DEFAULT_LABEL = os.environ.get("MAGPIE_LAUNCH_LABEL", "com.janearc.magpie")


class LaunchAgentPaths(BaseModel):
    # every path the agent touches, resolved and absolute. computed once so the
    # render / install / uninstall steps all agree on locations.
    label: str
    config_dir: Path
    plist_source: Path
    launchagents_dir: Path
    plist_installed: Path
    log_dir: Path
    stdout_log: Path
    stderr_log: Path
    working_dir: Path
    uv_bin: Path


class LaunchAgentStatus(BaseModel):
    # the JSON contract returned by every entry point.
    label: str
    installed: bool
    loaded: bool
    plist_source: str
    plist_installed: str
    action: Optional[str] = None
    message: Optional[str] = None


def _resolve_uv(uv_bin: Optional[str]) -> Path:
    # locate the uv launcher launchd will exec. launchd jobs get a minimal PATH,
    # so the plist must carry an absolute program path -- never bare "uv".
    if uv_bin:
        return Path(uv_bin).expanduser().resolve()
    found = shutil.which("uv")
    if found:
        return Path(found).resolve()
    # fall back to uv's standard astral install location.
    return Path.home() / ".local" / "bin" / "uv"


def resolve_paths(
    label: str = DEFAULT_LABEL,
    home: Optional[Path] = None,
    working_dir: Optional[str] = None,
    uv_bin: Optional[str] = None,
) -> LaunchAgentPaths:
    # derive the full path set from $HOME (injectable for tests). working_dir
    # defaults to the repo this module ships in so `uv run` resolves the venv.
    base = Path(home).expanduser() if home else Path.home()
    wd = (
        Path(working_dir).expanduser().resolve()
        if working_dir
        else Path(__file__).resolve().parent.parent
    )
    config_dir = base / "etc" / "magpie"
    launchagents_dir = base / "Library" / "LaunchAgents"
    log_dir = base / "var" / "log"
    return LaunchAgentPaths(
        label=label,
        config_dir=config_dir,
        plist_source=config_dir / f"{label}.plist",
        launchagents_dir=launchagents_dir,
        plist_installed=launchagents_dir / f"{label}.plist",
        log_dir=log_dir,
        stdout_log=log_dir / "magpie.out.log",
        stderr_log=log_dir / "magpie.err.log",
        working_dir=wd,
        uv_bin=_resolve_uv(uv_bin),
    )


def render_plist(paths: LaunchAgentPaths, host: str = "127.0.0.1", port: int = 8092) -> bytes:
    # build the launchd job dict and serialize it. KeepAlive=true gives the
    # self-healing contract (launchd restarts the daemon on crash); RunAtLoad
    # starts it at login. PATH is pinned because launchd does not inherit a
    # login shell environment.
    program_args = [
        str(paths.uv_bin),
        "run",
        "magpie",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    path_env = ":".join(
        [
            str(paths.uv_bin.parent),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]
    )
    job = {
        "Label": paths.label,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(paths.working_dir),
        "KeepAlive": True,
        "RunAtLoad": True,
        "ProcessType": "Interactive",
        "StandardOutPath": str(paths.stdout_log),
        "StandardErrorPath": str(paths.stderr_log),
        "EnvironmentVariables": {
            "PATH": path_env,
        },
    }
    return plistlib.dumps(job)


def _launchctl(args: list[str]) -> subprocess.CompletedProcess:
    # thin launchctl wrapper; never raises on a non-zero exit so the caller can
    # treat "job not loaded" (a normal idempotent state) as non-fatal.
    return subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def is_loaded(label: str) -> bool:
    # ask launchd whether the job is currently loaded. absence is not an error.
    if shutil.which("launchctl") is None:
        return False
    res = _launchctl(["list", label])
    return res.returncode == 0


def install(
    label: str = DEFAULT_LABEL,
    host: str = "127.0.0.1",
    port: int = 8092,
    home: Optional[Path] = None,
    working_dir: Optional[str] = None,
    uv_bin: Optional[str] = None,
    load: bool = True,
) -> LaunchAgentStatus:
    # idempotent install: write the plist to ~/etc/magpie, symlink it into
    # ~/Library/LaunchAgents, and (best-effort) load it into launchd. re-running
    # overwrites the rendered plist and reloads -- it never errors on "exists".
    paths = resolve_paths(label=label, home=home, working_dir=working_dir, uv_bin=uv_bin)

    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.launchagents_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    paths.plist_source.write_bytes(render_plist(paths, host=host, port=port))

    # symlink the loaded copy at the source-of-truth. replace any stale link or
    # file so the install is idempotent and always points at fresh content.
    if paths.plist_installed.is_symlink() or paths.plist_installed.exists():
        paths.plist_installed.unlink()
    paths.plist_installed.symlink_to(paths.plist_source)

    action = "installed"
    message = None
    loaded = False
    if load and shutil.which("launchctl") is not None:
        # bootout first so a re-install reloads cleanly (idempotent); ignore the
        # "not loaded" failure that the first install produces.
        _launchctl(["unload", str(paths.plist_installed)])
        res = _launchctl(["load", "-w", str(paths.plist_installed)])
        loaded = is_loaded(label)
        if res.returncode != 0 and not loaded:
            message = f"plist written but launchctl load failed: {res.stderr.strip()}"
            action = "installed-not-loaded"
    elif load:
        message = "launchctl not found; plist written but not loaded"
        action = "installed-not-loaded"

    return LaunchAgentStatus(
        label=label,
        installed=True,
        loaded=loaded,
        plist_source=str(paths.plist_source),
        plist_installed=str(paths.plist_installed),
        action=action,
        message=message,
    )


def uninstall(
    label: str = DEFAULT_LABEL,
    home: Optional[Path] = None,
    working_dir: Optional[str] = None,
) -> LaunchAgentStatus:
    # idempotent uninstall: unload from launchd (if loaded) and remove the
    # installed symlink. the source plist in ~/etc/magpie is left in place as a
    # record; removing an already-absent job is a no-op, not an error.
    paths = resolve_paths(label=label, home=home, working_dir=working_dir)

    if shutil.which("launchctl") is not None and paths.plist_installed.exists():
        _launchctl(["unload", str(paths.plist_installed)])

    removed = False
    if paths.plist_installed.is_symlink() or paths.plist_installed.exists():
        paths.plist_installed.unlink()
        removed = True

    return LaunchAgentStatus(
        label=label,
        installed=False,
        loaded=False,
        plist_source=str(paths.plist_source),
        plist_installed=str(paths.plist_installed),
        action="uninstalled" if removed else "already-absent",
    )


def status(
    label: str = DEFAULT_LABEL,
    home: Optional[Path] = None,
    working_dir: Optional[str] = None,
) -> LaunchAgentStatus:
    # report the current install/load state without mutating anything.
    paths = resolve_paths(label=label, home=home, working_dir=working_dir)
    installed = paths.plist_installed.is_symlink() or paths.plist_installed.exists()
    return LaunchAgentStatus(
        label=label,
        installed=installed,
        loaded=is_loaded(label),
        plist_source=str(paths.plist_source),
        plist_installed=str(paths.plist_installed),
        action="status",
    )
