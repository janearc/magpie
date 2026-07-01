# magpie.register -- magpie joins the mesh: it registers with delightd on startup as the first
# WATCHER frood. delightd is the source of truth for who is on the mesh; magpie presents its
# identity, the contracts it speaks, and the /health endpoint delightd will probe.
#
# Registration is ADDITIVE today (delightd does not yet require a frood to register), so a
# registration that does not complete must NOT stop magpie transcribing -- but it must be LOUD.
# A frood that silently failed to register would look joined when delightd has no record of it,
# and that false-membership is exactly what the mesh forbids. So: attempt once the server is up,
# log the outcome plainly, and keep serving either way (frood.register itself raises; magpie
# chooses to log-and-continue because the join is additive, not because the failure is swallowed).
import logging
import os
import threading
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version

from bento.v1 import bento_pb2
from frood.register import RegistrationError
from frood.register import register as send_registration
from frood.v1 import frood_pb2
from observability.v1 import observability_pb2
from registry.v1 import register_pb2

log = logging.getLogger(__name__)

# magpie's own name on the mesh: its service_name (heartbeat key + discovery key) and the
# delightd project it binds to are both "magpie".
SERVICE_NAME = "magpie"


def _magpie_version() -> str:
    # the frood's build/version string for its Identity. Read from installed package metadata so
    # it tracks the real version; fall back to "0+unknown" if magpie is run from a source tree
    # without an installed dist (a missing version must not crash registration).
    try:
        return version("magpie")
    except PackageNotFoundError:
        return "0+unknown"


def build_registration(endpoint_address: str):
    # Assemble magpie's (identity, contracts, endpoints) for a RegisterRequest.
    #
    # identity: service_name and project are both "magpie"; version tracks the installed dist.
    identity = frood_pb2.Identity(
        service_name=SERVICE_NAME, project=SERVICE_NAME, version=_magpie_version()
    )
    # contracts: magpie is a watcher, so it EMITS (it does not consume or serve a bus contract).
    # It emits two things, both via its Go sidecar: the liveness heartbeat delightd hard-requires,
    # and its own work output, the bento lifecycle. The subjects are taken from the generated
    # message descriptors (full_name), NOT hard-coded FQN strings -- a proto message rename then
    # moves the declared subject automatically instead of silently desyncing a literal.
    contracts = frood_pb2.ContractDescriptor()
    contracts.emits.add(subject=observability_pb2.ServiceHealthHeartbeat.DESCRIPTOR.full_name)
    contracts.emits.add(subject=bento_pb2.BentoLifecycleEvent.DESCRIPTOR.full_name)
    # endpoints: the address delightd will dial for its /health guarantee. It MUST be reachable
    # FROM delightd (a fleet DNS name, not localhost), so it is deployment-set via
    # MAGPIE_ENDPOINT_ADDRESS; the default is the fleet convention magpie:8092.
    endpoints = [register_pb2.Endpoint(scheme="http", address=endpoint_address)]
    return identity, contracts, endpoints


def register_with_delightd(endpoint_address: str | None = None, delightd_url: str | None = None):
    # Build and send magpie's registration. Returns the RegisterResponse on success, or None if
    # the join did not complete (logged loudly) -- magpie keeps serving regardless, because
    # registration is additive today.
    endpoint_address = endpoint_address or os.environ.get("MAGPIE_ENDPOINT_ADDRESS", "magpie:8092")
    identity, contracts, endpoints = build_registration(endpoint_address)
    try:
        resp = send_registration(identity, contracts, endpoints, delightd_url=delightd_url)
    except RegistrationError as e:
        # LOUD, but non-fatal: name why the join failed (delightd reports the reason) so an
        # operator sees "magpie is not on the registry, and here is why" rather than silence.
        log.warning(
            "magpie: registration with delightd did NOT complete "
            "(still serving; join is additive): %s",
            e,
        )
        return None
    log.info(
        "magpie: registered with delightd as %r (endpoint %s, lease_ttl=%ss)",
        SERVICE_NAME,
        endpoint_address,
        resp.lease_ttl_seconds,
    )
    return resp


def register_when_healthy(host: str, port: int, delightd_url: str | None = None) -> None:
    # Wait until magpie's OWN /health is accepting connections, THEN register -- delightd probes
    # /health during the register call, so registering before the server is up would fail the
    # reachability guarantee on a race. The wait is bounded (~30s): if the server never comes up,
    # give up rather than spin, and let register run once (it will fail loud, which is correct --
    # a server that never became healthy should not be on the registry).
    health_url = f"http://{host}:{port}/health"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:  # noqa: BLE001 - the server may simply not be accepting yet; keep waiting
            pass
        time.sleep(1)
    register_with_delightd(delightd_url=delightd_url)


def start_registration(host: str, port: int, delightd_url: str | None = None) -> threading.Thread:
    # Kick off registration on a daemon thread so it never blocks the server's own startup. The
    # thread waits for /health then registers; returned so a caller (or a test) can join it.
    t = threading.Thread(
        target=register_when_healthy, args=(host, port, delightd_url), daemon=True
    )
    t.start()
    return t
