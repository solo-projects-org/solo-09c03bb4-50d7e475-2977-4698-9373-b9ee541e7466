"""Net-product computation and idempotency fingerprinting for publishing.

The published product is derived *only* from the replayed source rows:

* replay the submitted initial state and batches with the same legality rules
  used by the existing validation API (unique primary keys, update/delete
  targets exist, foreign keys valid after every batch);
* for every order in the final state compute ``order.amount - sum(refund
  amounts of that order)``;
* reject the request if any resulting net amount does not fit in a signed
  64-bit integer.

No partial product is ever produced.  A :class:`NetAmountOutOfRange` (or an
:class:`~app.validate.InputError`) means the caller must not publish.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .models import INT64_MAX, INT64_MIN, Batch, InitialData
from .validate import InputError, replay


class NetAmountOutOfRange(Exception):
    """At least one net amount cannot be represented as a signed 64-bit int."""

    def __init__(self, order_id: int, net_amount: int) -> None:
        super().__init__(
            f"order_id={order_id} 的净额 {net_amount} 超出 64 位整数范围 "
            f"[{INT64_MIN}, {INT64_MAX}]"
        )
        self.order_id = order_id
        self.net_amount = net_amount


@dataclass(frozen=True)
class ProductRow:
    order_id: int
    net_amount: int


def _check_order_id(order_id: int) -> None:
    # The source model only requires a positive integer.  The product is
    # persisted in a SQLite INTEGER column, so enforce the same 64-bit bound
    # here and treat a violation as illegal input (never publish).
    if order_id < INT64_MIN or order_id > INT64_MAX:
        raise InputError(f"order_id={order_id} 超出 64 位整数范围")


def compute_net_product(
    initial: InitialData, batches: Iterable[Batch]
) -> list[ProductRow]:
    """Replay the batches and return the complete, sorted net product.

    Raises :class:`~app.validate.InputError` for an illegal initial state or
    batch and :class:`NetAmountOutOfRange` for a net amount outside int64.
    The result is only returned after the *whole* product has been computed;
    callers may persist it afterwards.
    """
    batches = list(batches)
    initial_state, after_batches = replay(initial, batches)
    state = after_batches[-1] if after_batches else initial_state
    refund_totals: dict[int, int] = {}
    for refund in state.refunds.values():
        order_id = refund["order_id"]
        refund_totals[order_id] = refund_totals.get(order_id, 0) + refund["amount"]

    rows: list[ProductRow] = []
    for order_id in sorted(state.orders):
        _check_order_id(order_id)
        net_amount = state.orders[order_id]["amount"] - refund_totals.get(order_id, 0)
        if net_amount < INT64_MIN or net_amount > INT64_MAX:
            raise NetAmountOutOfRange(order_id, net_amount)
        rows.append(ProductRow(order_id=order_id, net_amount=net_amount))
    return rows


def request_content_hash(initial: InitialData, batches: Iterable[Batch]) -> str:
    """Stable fingerprint of the product-defining content of a request.

    ``request_id`` is the idempotency key and ``base_version`` is only a
    concurrency guard, so neither is part of the fingerprint.  Re-sending the
    same ``request_id`` with the same ``initial``/``batches`` always returns
    the first published version (even if the caller now sees a newer
    version); changing any content is rejected.  Failed attempts are not
    recorded, so a conflict or invalid request may be retried under the same
    id.
    """
    payload: dict[str, Any] = {
        "initial": initial.model_dump(mode="json"),
        "batches": [
            batch.model_dump(mode="json", exclude_none=True) for batch in batches
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
