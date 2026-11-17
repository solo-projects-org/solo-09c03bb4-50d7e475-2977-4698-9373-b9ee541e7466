"""Tests for net-product computation, publishing, idempotency and reads."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.models import Batch, InitialData
from app.publish import (
    NetAmountOutOfRange,
    compute_net_product,
    request_content_hash,
)
from app.store import (
    PublishStore,
    RequestIdReused,
    VersionConflict,
)
from app.validate import InputError

from conftest import op_delete, op_insert, op_update

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


@pytest.fixture(autouse=True)
def publish_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "published.db"
    monkeypatch.setattr(config, "PUBLISH_DB_PATH", str(path))
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _initial(
    orders: list[dict] | None = None, refunds: list[dict] | None = None
) -> InitialData:
    return InitialData.model_validate(
        {
            "orders": orders
            if orders is not None
            else [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
            "refunds": refunds
            if refunds is not None
            else [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    )


def _batches() -> list[Batch]:
    return [
        Batch.model_validate(
            {
                "batch_id": 11,
                "changes": [op_insert("refunds", refund_id=20, order_id=1, amount=50)],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 12,
                "changes": [op_insert("orders", order_id=3, amount=800)],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 13,
                "changes": [
                    op_update(
                        "orders", {"order_id": 1}, {"order_id": 1, "amount": 1200}
                    )
                ],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 14,
                "changes": [op_delete("refunds", refund_id=20)],
            }
        ),
    ]


def _payload(
    request_id: str = "req-1",
    base_version: int = 0,
    *,
    initial: dict | None = None,
    batches: list[dict] | None = None,
) -> dict:
    if initial is None:
        initial = {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        }
    if batches is None:
        batches = [b.model_dump(mode="json", exclude_none=True) for b in _batches()]
    return {
        "request_id": request_id,
        "base_version": base_version,
        "initial": initial,
        "batches": batches,
    }


# ---------------------------------------------------------------------------
# Product derivation
# ---------------------------------------------------------------------------


def test_net_product_is_order_amount_minus_refund_total():
    rows = compute_net_product(_initial(), _batches())
    assert [(r.order_id, r.net_amount) for r in rows] == [
        (1, 1000),  # 1200 (updated) - 200 (initial refund 10); refund 20 deleted
        (2, 500),
        (3, 800),
    ]


def test_net_product_without_batches_uses_initial_state():
    rows = compute_net_product(_initial(), [])
    assert [(r.order_id, r.net_amount) for r in rows] == [(1, 800), (2, 500)]


def test_net_product_rejects_illegal_batch():
    batches = [
        Batch.model_validate(
            {
                "batch_id": 1,
                "changes": [op_delete("refunds", refund_id=999)],
            }
        )
    ]
    with pytest.raises(InputError):
        compute_net_product(_initial(), batches)


@pytest.mark.parametrize(
    ("amount", "refund"),
    [
        (INT64_MAX, None),
        (INT64_MIN, None),
        (0, None),
    ],
)
def test_net_product_accepts_int64_boundaries(
    amount: int, refund: int | None
):
    initial = _initial(
        orders=[{"order_id": 1, "amount": amount}],
        refunds=[]
        if refund is None
        else [{"refund_id": 10, "order_id": 1, "amount": refund}],
    )
    rows = compute_net_product(initial, [])
    assert rows[0].net_amount == amount


@pytest.mark.parametrize(
    ("amount", "refund"),
    [
        (INT64_MIN, 1),   # INT64_MIN - 1
        (INT64_MAX, -1),  # INT64_MAX + 1 (refund amounts may be negative)
    ],
)
def test_net_product_rejects_out_of_range(amount: int, refund: int):
    initial = _initial(
        orders=[{"order_id": 1, "amount": amount}],
        refunds=[{"refund_id": 10, "order_id": 1, "amount": refund}],
    )
    with pytest.raises(NetAmountOutOfRange) as excinfo:
        compute_net_product(initial, [])
    assert excinfo.value.order_id == 1
    assert excinfo.value.net_amount == amount - refund


def test_net_product_overflow_after_refund_total():
    # INT64_MIN - 1 is outside int64; refunds are subtracted.
    initial = _initial(
        orders=[{"order_id": 1, "amount": INT64_MIN}],
        refunds=[{"refund_id": 10, "order_id": 1, "amount": 1}],
    )
    with pytest.raises(NetAmountOutOfRange) as excinfo:
        compute_net_product(initial, [])
    assert excinfo.value.order_id == 1
    assert excinfo.value.net_amount == INT64_MIN - 1


def test_net_product_overflow_via_batch():
    batches = [
        Batch.model_validate(
            {
                "batch_id": 1,
                "changes": [
                    op_insert("orders", order_id=99, amount=INT64_MAX),
                ],
            }
        ),
        Batch.model_validate(
            {
                "batch_id": 2,
                "changes": [
                    op_insert("refunds", refund_id=98, order_id=99, amount=0)
                ],
            }
        ),
    ]
    # Legal and in range: INT64_MAX - 0.
    assert compute_net_product(_initial(), batches)[-1].net_amount == INT64_MAX

    overflowing = [
        batches[0],
        Batch.model_validate(
            {
                "batch_id": 2,
                "changes": [
                    op_insert("refunds", refund_id=98, order_id=99, amount=-1)
                ],
            }
        ),
    ]
    # INT64_MAX - (-1) = INT64_MAX + 1, which is outside int64.
    with pytest.raises(NetAmountOutOfRange):
        compute_net_product(_initial(), overflowing)


# ---------------------------------------------------------------------------
# Publish API
# ---------------------------------------------------------------------------


def test_publish_creates_version_one(client: TestClient):
    response = client.post("/v1/publish", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "version": 1,
        "replayed": False,
        "row_count": 3,
        "product": [
            {"order_id": 1, "net_amount": 1000},
            {"order_id": 2, "net_amount": 500},
            {"order_id": 3, "net_amount": 800},
        ],
    }


def test_publish_next_version_requires_current_base(client: TestClient):
    assert client.post("/v1/publish", json=_payload()).status_code == 200
    stale = client.post("/v1/publish", json=_payload("req-2", base_version=0))
    assert stale.status_code == 409
    assert "所见版本" in stale.json()["detail"]

    next_payload = _payload("req-2", base_version=1)
    next_payload["batches"] = next_payload["batches"][:1]
    fresh = client.post("/v1/publish", json=next_payload)
    assert fresh.status_code == 200
    assert fresh.json()["version"] == 2


def test_same_request_id_same_content_returns_first_version(client: TestClient):
    first = client.post("/v1/publish", json=_payload())
    assert first.json()["version"] == 1

    # Even after another publication advanced the current version, replaying
    # the same request id/content returns the *first* published version.
    other = _payload("req-2", base_version=1)
    other["batches"] = other["batches"][:1]
    assert client.post("/v1/publish", json=other).json()["version"] == 2

    replay = client.post("/v1/publish", json=_payload(base_version=0))
    assert replay.status_code == 200
    assert replay.json()["version"] == 1
    assert replay.json()["replayed"] is True
    assert client.get("/v1/products/versions").json()["current_version"] == 2

    # The product returned on replay is the stored one, not a re-derivation.
    assert replay.json()["product"] == first.json()["product"]


def test_same_request_id_different_content_rejected(client: TestClient):
    assert client.post("/v1/publish", json=_payload()).status_code == 200
    changed = _payload("req-1", base_version=0)
    changed["batches"] = changed["batches"][:1]
    response = client.post("/v1/publish", json=changed)
    assert response.status_code == 409
    assert "不同内容" in response.json()["detail"]
    # No new version and the existing request/product are untouched.
    versions = client.get("/v1/products/versions").json()
    assert versions["current_version"] == 1
    assert len(versions["versions"]) == 1


def test_failed_publish_does_not_change_products_or_request_log(
    client: TestClient, publish_db: Path
):
    assert client.post("/v1/publish", json=_payload()).status_code == 200
    before = client.get("/v1/products/current").json()

    store = PublishStore(publish_db)
    request_count_before = len(store.list_versions())

    # Illegal input (dangling delete), out-of-range net amount, stale version,
    # and request-id reuse all fail without writing anything.
    illegal = _payload("req-illegal", base_version=1)
    illegal["batches"] = [
        {"batch_id": 1, "changes": [op_delete("refunds", refund_id=999)]}
    ]
    assert client.post("/v1/publish", json=illegal).status_code == 422

    overflow = _payload(
        "req-overflow",
        base_version=1,
        initial={
            "orders": [{"order_id": 1, "amount": INT64_MIN}],
            "refunds": [{"refund_id": 5, "order_id": 1, "amount": 1}],
        },
        # Batch 13 updates order 1 to 1200, so INT64_MIN is no longer used.
        # Rebuild the payload so the overflowing order survives the replay.
        batches=[
            {
                "batch_id": 1,
                "changes": [
                    op_insert("orders", order_id=50, amount=INT64_MIN)
                ],
            },
            {
                "batch_id": 2,
                "changes": [
                    op_insert("refunds", refund_id=51, order_id=50, amount=1)
                ],
            },
        ],
    )
    assert client.post("/v1/publish", json=overflow).status_code == 422

    stale = _payload("req-stale", base_version=0)
    assert client.post("/v1/publish", json=stale).status_code == 409

    assert store.current_version() == 1
    assert len(store.list_versions()) == request_count_before
    assert client.get("/v1/products/current").json() == before
    for request_id in ("req-illegal", "req-overflow", "req-stale"):
        assert store.get_request(request_id) is None


def test_failed_request_id_can_be_retried_with_same_id(client: TestClient):
    # Another publisher wins version 1 first.
    assert client.post("/v1/publish", json=_payload("req-a")).status_code == 200

    # "retry-me" observed version 0, which is now stale -> conflict, and the
    # failed request must not be recorded.
    stale = _payload("retry-me", base_version=0)
    assert client.post("/v1/publish", json=stale).status_code == 409

    # Retrying the *same* id with the same content and the correct observed
    # version succeeds: it is treated as a fresh request, not a reuse.
    retry = _payload("retry-me", base_version=1)
    response = client.post("/v1/publish", json=retry)
    assert response.status_code == 200
    assert response.json()["version"] == 2
    assert response.json()["replayed"] is False


def test_validation_rejects_bad_request_ids_and_batches(client: TestClient):
    bad_id = _payload("bad id with spaces")
    assert client.post("/v1/publish", json=bad_id).status_code == 422

    empty_batches = _payload("req-x")
    empty_batches["batches"] = []
    assert client.post("/v1/publish", json=empty_batches).status_code == 422

    duplicate_ids = _payload("req-x")
    duplicate_ids["batches"][1]["batch_id"] = duplicate_ids["batches"][0]["batch_id"]
    assert client.post("/v1/publish", json=duplicate_ids).status_code == 422


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------


def test_read_current_and_by_version(client: TestClient):
    assert client.get("/v1/products/current").status_code == 404
    assert client.get("/v1/products/1").status_code == 404

    client.post("/v1/publish", json=_payload())
    second = _payload("req-2", base_version=1)
    second["batches"] = second["batches"][:1]
    client.post("/v1/publish", json=second)

    current = client.get("/v1/products/current").json()
    assert current["version"] == 2
    assert current["request_id"] == "req-2"
    assert current["row_count"] == len(current["product"]) == 2

    old = client.get("/v1/products/1").json()
    assert old["version"] == 1
    assert old["request_id"] == "req-1"
    assert old["product"] == [
        {"order_id": 1, "net_amount": 1000},
        {"order_id": 2, "net_amount": 500},
        {"order_id": 3, "net_amount": 800},
    ]

    versions = client.get("/v1/products/versions").json()
    assert versions["current_version"] == 2
    assert [v["version"] for v in versions["versions"]] == [1, 2]


def test_read_invalid_version(client: TestClient):
    assert client.get("/v1/products/0").status_code == 404
    assert client.get("/v1/products/999").status_code == 404


# ---------------------------------------------------------------------------
# Persistence and concurrency
# ---------------------------------------------------------------------------


def test_product_survives_new_store_instance(client: TestClient, publish_db: Path):
    client.post("/v1/publish", json=_payload())
    # A brand-new store/connection must see the committed version.
    reopened = PublishStore(publish_db)
    snapshot = reopened.get_product()
    assert snapshot is not None
    assert snapshot["version"] == 1
    assert len(snapshot["rows"]) == 3


def test_concurrent_publish_same_base_version_only_one_wins(publish_db: Path):
    store = PublishStore(publish_db)
    initial = _initial()
    batches = _batches()
    content_hash = request_content_hash(initial, batches)
    product = compute_net_product(initial, batches)

    barrier = threading.Barrier(8)
    outcomes: list[int | str] = []

    def attempt(index: int) -> None:
        barrier.wait()
        try:
            result = store.publish(
                base_version=0,
                request_id=f"race-{index}",
                content_hash=content_hash + f":{index}",
                product=product,
            )
            outcomes.append(result.version)
        except VersionConflict:
            outcomes.append("conflict")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(attempt, i) for i in range(8)]
        for future in futures:
            future.result()

    assert outcomes.count(1) == 1
    assert outcomes.count("conflict") == 7
    assert store.current_version() == 1
    assert len(store.list_versions()) == 1
    # Losers recorded no request entry.
    for index in range(8):
        record = store.get_request(f"race-{index}")
        assert record is None or record.version == 1


def test_concurrent_api_publish_only_one_succeeds(client: TestClient):
    barrier = threading.Barrier(6)
    statuses: list[int] = []

    def call(index: int) -> None:
        barrier.wait()
        response = client.post(
            "/v1/publish", json=_payload(f"api-race-{index}", base_version=0)
        )
        statuses.append(response.status_code)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(call, i) for i in range(6)]
        for future in futures:
            future.result()

    assert statuses.count(200) == 1
    assert statuses.count(409) == 5
    versions = client.get("/v1/products/versions").json()
    assert versions["current_version"] == 1
    assert len(versions["versions"]) == 1


def test_readers_only_see_complete_versions(publish_db: Path):
    store = PublishStore(publish_db)
    initial = _initial()
    base_batches = _batches()

    # Version 1: three orders.
    store.publish(
        base_version=0,
        request_id="v1",
        content_hash=request_content_hash(initial, base_batches),
        product=compute_net_product(initial, base_batches),
    )

    # Version 2: add many orders/refunds so a torn read would be obvious.
    extra_batches = base_batches + [
        Batch.model_validate(
            {
                "batch_id": 100 + i,
                "changes": [op_insert("orders", order_id=1000 + i, amount=i + 1)],
            }
        )
        for i in range(20)
    ]
    version_two = compute_net_product(initial, extra_batches)

    stop = threading.Event()
    errors: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            snapshot = store.get_product()
            assert snapshot is not None
            rows = snapshot["rows"]
            if snapshot["version"] == 1:
                if len(rows) != 3:
                    errors.append(f"v1 torn: {len(rows)} rows")
            elif snapshot["version"] == 2:
                if len(rows) != len(version_two):
                    errors.append(f"v2 torn: {len(rows)} rows")
            else:  # pragma: no cover
                errors.append(f"unexpected version {snapshot['version']}")

    reader_threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in reader_threads:
        thread.start()
    try:
        store.publish(
            base_version=1,
            request_id="v2",
            content_hash=request_content_hash(initial, extra_batches),
            product=version_two,
        )
    finally:
        stop.set()
        for thread in reader_threads:
            thread.join()

    assert not errors
    assert store.current_version() == 2


def test_store_rejects_reused_request_id_directly(publish_db: Path):
    store = PublishStore(publish_db)
    initial = _initial()
    batches = _batches()
    content_hash = request_content_hash(initial, batches)
    product = compute_net_product(initial, batches)
    store.publish(
        base_version=0,
        request_id="dup",
        content_hash=content_hash,
        product=product,
    )
    with pytest.raises(RequestIdReused):
        store.publish(
            base_version=1,
            request_id="dup",
            content_hash=content_hash + "x",
            product=product,
        )
    assert store.current_version() == 1
    # Identical content replays the first version under a newer base.
    replay = store.publish(
        base_version=1,
        request_id="dup",
        content_hash=content_hash,
        product=product,
    )
    assert replay.version == 1 and replay.replayed
    assert store.current_version() == 1


def test_swagger_lists_publish_and_read_endpoints(client: TestClient):
    schema = client.get("/openapi.json").json()
    for path in (
        "/v1/publish",
        "/v1/products/current",
        "/v1/products/versions",
        "/v1/products/{version}",
    ):
        assert path in schema["paths"]
    # Existing API remains available.
    assert "/v1/minimize" in schema["paths"]
    assert "/v1/validate" in schema["paths"]
    assert client.get("/docs").status_code == 200
