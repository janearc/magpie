# tests for magpie.register: it builds an honest RegisterRequest (identity + both emit subjects +
# the /health endpoint) and registers on startup, failing LOUD but non-fatal when delightd
# declines (registration is additive -- magpie keeps transcribing either way).
from magpie import register
from registry.v1 import register_pb2


def test_build_registration_declares_identity_and_both_emits():
    identity, contracts, endpoints = register.build_registration("magpie.fleet:8092")
    # identity: magpie names itself and the project it binds to; version is non-empty.
    assert identity.service_name == "magpie"
    assert identity.project == "magpie"
    assert identity.version
    # emits: the required liveness heartbeat AND magpie's work output, the bento lifecycle. The
    # literal FQNs here are the ground-truth pin -- build_registration derives them from the
    # message descriptors, so a rename that moved them would fail this assertion.
    subjects = {ref.subject for ref in contracts.emits}
    assert "observability.v1.ServiceHealthHeartbeat" in subjects
    assert "bento.v1.BentoLifecycleEvent" in subjects
    # a watcher emits; it neither consumes nor serves a bus contract.
    assert len(contracts.consumes) == 0
    assert len(contracts.serves) == 0
    # endpoint: the address delightd will probe, as given.
    assert endpoints[0].scheme == "http"
    assert endpoints[0].address == "magpie.fleet:8092"


def test_register_until_joined_returns_response_on_success(monkeypatch):
    captured = {}

    def fake_send(identity, contracts, endpoints, delightd_url=None):
        captured["service_name"] = identity.service_name
        captured["url"] = delightd_url
        return register_pb2.RegisterResponse(lease_ttl_seconds=30)

    monkeypatch.setattr(register, "send_registration", fake_send)
    resp = register.register_until_joined(
        endpoint_address="magpie.fleet:8092", delightd_url="http://delightd:8088"
    )
    assert resp is not None
    assert resp.lease_ttl_seconds == 30
    assert captured["service_name"] == "magpie"
    assert captured["url"] == "http://delightd:8088"


def test_register_until_joined_stops_on_decline(monkeypatch):
    # a 4xx DECLINE (unknown project) is terminal: return None (logged loudly), do NOT raise and do
    # NOT retry -- retrying can't change a "you are not allowed", and magpie keeps serving.
    calls = {"n": 0}

    def fake_send(identity, contracts, endpoints, delightd_url=None):
        calls["n"] += 1
        raise register.RegistrationError(404, "project not found")

    monkeypatch.setattr(register, "send_registration", fake_send)
    resp = register.register_until_joined(endpoint_address="magpie.fleet:8092")
    assert resp is None
    assert calls["n"] == 1  # terminal: tried exactly once, did not retry a decline


def test_register_until_joined_retries_while_unreachable_then_succeeds(monkeypatch):
    # status 0 (delightd unreachable) is transient: keep retrying until delightd answers. Two
    # unreachable failures then a success -> the join completes without a restart. _sleep is stubbed
    # so the retry does not actually wait.
    attempts = {"n": 0}
    slept = []

    def fake_send(identity, contracts, endpoints, delightd_url=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise register.RegistrationError(0, "delightd unreachable: connection refused")
        return register_pb2.RegisterResponse(lease_ttl_seconds=15)

    monkeypatch.setattr(register, "send_registration", fake_send)
    resp = register.register_until_joined(
        endpoint_address="magpie.fleet:8092", _sleep=slept.append
    )
    assert resp is not None
    assert resp.lease_ttl_seconds == 15
    assert attempts["n"] == 3  # retried twice, joined on the third
    assert slept == [2.0, 4.0]  # backoff doubled between the two retries


def test_endpoint_address_defaults_from_env(monkeypatch):
    # unset -> the fleet-convention default; MAGPIE_ENDPOINT_ADDRESS overrides it.
    monkeypatch.delenv("MAGPIE_ENDPOINT_ADDRESS", raising=False)
    captured = {}

    def fake_send(identity, contracts, endpoints, delightd_url=None):
        captured["address"] = endpoints[0].address
        return register_pb2.RegisterResponse()

    monkeypatch.setattr(register, "send_registration", fake_send)
    register.register_once()
    assert captured["address"] == "magpie:8092"
    monkeypatch.setenv("MAGPIE_ENDPOINT_ADDRESS", "magpie.fleet:9999")
    register.register_once()
    assert captured["address"] == "magpie.fleet:9999"
