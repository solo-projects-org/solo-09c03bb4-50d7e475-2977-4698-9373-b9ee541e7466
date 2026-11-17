"""FastAPI application: incremental-net SQL minimizer and net-product publisher.

Every request is handled synchronously and uses its own in-memory SQLite
databases created by the :class:`~app.executor.Executor`.  No global mutable
state, no temp files, no network access: a trial cannot affect another request
(and FastAPI runs sync endpoints in a worker thread pool, so two requests can
run concurrently without sharing a connection).

The publish/read endpoints additionally use a small on-disk SQLite store
(:mod:`app.store`) so a complete product survives restarts.  Publication is
one SQLite transaction with a compare-and-set on the observed version.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from . import __version__, config
from .executor import Executor
from .models import (
    Batch,
    DifferenceDetail,
    MinimizeRequest,
    MinimizeResponse,
    ProductResponse,
    PublishRequest,
    PublishResponse,
    ValidateRequest,
    ValidateResponse,
    VersionListResponse,
)
from .publish import (
    NetAmountOutOfRange,
    compute_net_product,
    request_content_hash,
)
from .reducer import Reducer
from .sqlguard import SqlPolicyError
from .store import PublishStore, RequestIdReused, VersionConflict
from .validate import InputError, validate_all

app = FastAPI(
    title="增量净额 SQL 反例精简器 / 净额成品发布",
    version=__version__,
    description=(
        "两部分能力：\n\n"
        "1. **反例精简**（`/v1/minimize`、`/v1/validate`）：给出 orders/refunds "
        "初始数据、至多 20 批增删改变更、全量 SELECT 与增量 SQL，在隔离的内存 "
        "SQLite 中逐批重放，按多重集比较并缩减批次。\n"
        "2. **净额成品发布与读取**（`/v1/publish`、`/v1/products/...`）：提交现有"
        "格式的初态、顺序批次、所见发布版本和请求标识；服务沿用批内合法性规则重放，"
        "逐订单计算 `订单金额 - 退款合计`，整份成品算完后以所见版本做原子比较并发布"
        "下一版本。\n\n"
        "**发布一致性**\n"
        "- 净额超出 64 位整数范围或输入非法时**不发布**；失败不会改变已有成品或请求记录。\n"
        "- 只有整份成品计算完成且所见版本仍是当前版本才持久发布；并发争同一版本只有"
        "一个成功，另一个返回 409。\n"
        "- 读取总是得到某个完整的旧版或新版（版本号与行在同一数据库快照中读取）。\n"
        "- 同一 `request_id` 重送相同内容返回首次发布版本（`replayed=true`）；内容不同"
        "返回 409；失败请求不记录，可用同一标识重试。"
    ),
    contact={"name": "结算数仓工具"},
)


def _publish_store() -> PublishStore:
    # Resolved per call so tests and deployments can point PUBLISH_DB_PATH at
    # a temp directory without import-time side effects.
    return PublishStore(Path(config.PUBLISH_DB_PATH))


@app.get("/health", summary="健康检查", tags=["运维"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "statement_timeout_seconds": config.STATEMENT_TIMEOUT_SECONDS,
        "max_result_rows": config.MAX_RESULT_ROWS,
        "max_trials": config.MAX_TRIALS,
        "max_request_seconds": config.MAX_REQUEST_SECONDS,
        "publish_db_path": str(config.PUBLISH_DB_PATH),
    }


def _prepare(request: MinimizeRequest | ValidateRequest) -> Executor:
    try:
        validate_all(request.initial, request.batches)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=f"输入非法: {exc}") from exc
    try:
        return Executor(request.sql.full_select, request.sql.incremental)
    except SqlPolicyError as exc:
        raise HTTPException(status_code=422, detail=f"SQL 不符合策略: {exc}") from exc


def _dump_batches(batches: list[Batch]) -> list[dict[str, Any]]:
    # ``exclude_none`` drops ``before`` from inserts and ``after`` from
    # deletes; the reduced input is still a valid request payload.
    return [b.model_dump(mode="json", exclude_none=True) for b in batches]


@app.post(
    "/v1/minimize",
    response_model=MinimizeResponse,
    summary="精简反例并返回首次差异",
    tags=["精简"],
    responses={
        422: {"description": "输入非法或 SQL 违反安全策略"},
    },
)
def minimize(request: MinimizeRequest) -> MinimizeResponse:
    executor = _prepare(request)
    try:
        outcome = Reducer(request.initial, request.batches, executor).run()
    except InputError as exc:  # pragma: no cover - validated above
        raise HTTPException(status_code=422, detail=f"输入非法: {exc}") from exc
    except SqlPolicyError as exc:  # pragma: no cover - validated above
        raise HTTPException(status_code=422, detail=f"SQL 不符合策略: {exc}") from exc

    difference = None
    if outcome.difference is not None:
        missing, extra = outcome.difference
        difference = DifferenceDetail(missing=missing, extra=extra)

    reduced = {
        "initial": request.initial.model_dump(mode="json"),
        "batches": _dump_batches(outcome.reduced_batches),
        "sql": request.sql.model_dump(mode="json"),
    }
    return MinimizeResponse(
        status=outcome.status,  # type: ignore[arg-type]
        reduced_input=reduced,
        first_difference_batch_id=outcome.first_difference_batch_id,
        difference=difference,
        notes=outcome.notes,
        blocked_by=outcome.blocked_by,
        evaluations=outcome.evaluations,
    )


@app.post(
    "/v1/validate",
    response_model=ValidateResponse,
    summary="完整输入一次重放（不缩减）",
    tags=["校验"],
)
def validate(request: ValidateRequest) -> ValidateResponse:
    executor = _prepare(request)
    result = executor.run(request.initial, request.batches)
    difference = None
    batch_id = None
    if result.first_difference is not None:
        batch_id = result.first_difference.batch_id
        difference = DifferenceDetail(
            missing=result.first_difference.missing,
            extra=result.first_difference.extra,
        )
    return ValidateResponse(
        outcome=result.outcome,  # type: ignore[arg-type]
        first_difference_batch_id=batch_id,
        difference=difference,
        notes=result.notes,
    )


@app.post(
    "/v1/publish",
    response_model=PublishResponse,
    summary="重放变更并原子发布下一版净额成品",
    tags=["成品"],
    responses={
        409: {
            "description": (
                "所见版本已失效（他人已发布），或同一请求标识被用于不同内容；"
                "已有成品与请求记录不变"
            )
        },
        422: {"description": "输入非法或净额超出 64 位整数范围；不发布任何内容"},
    },
)
def publish(request: PublishRequest) -> PublishResponse:
    try:
        product = compute_net_product(request.initial, request.batches)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=f"输入非法: {exc}") from exc
    except NetAmountOutOfRange as exc:
        raise HTTPException(
            status_code=422, detail=f"净额超出 64 位整数范围: {exc}"
        ) from exc

    content_hash = request_content_hash(request.initial, request.batches)
    store = _publish_store()
    try:
        outcome = store.publish(
            base_version=request.base_version,
            request_id=request.request_id,
            content_hash=content_hash,
            product=product,
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RequestIdReused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail=f"发布存储暂不可用，未改变已有成品: {exc}"
        ) from exc

    return PublishResponse(
        version=outcome.version,
        replayed=outcome.replayed,
        row_count=len(outcome.product),
        product=[
            {"order_id": row.order_id, "net_amount": row.net_amount}
            for row in outcome.product
        ],
    )


@app.get(
    "/v1/products/current",
    response_model=ProductResponse,
    summary="读取当前完整成品",
    tags=["成品"],
    responses={404: {"description": "尚未发布任何版本"}},
)
def get_current_product() -> ProductResponse:
    return _read_product(None)


@app.get(
    "/v1/products/versions",
    response_model=VersionListResponse,
    summary="列出已发布版本",
    tags=["成品"],
)
def list_versions() -> VersionListResponse:
    store = _publish_store()
    try:
        current, versions = store.version_overview()
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail=f"发布存储暂不可用: {exc}"
        ) from exc
    return VersionListResponse(
        current_version=current,
        versions=[
            {
                "version": info.version,
                "request_id": info.request_id,
                "row_count": info.row_count,
                "published_at": info.published_at,
            }
            for info in versions
        ],
    )


@app.get(
    "/v1/products/{version}",
    response_model=ProductResponse,
    summary="按版本号读取完整成品",
    tags=["成品"],
    responses={404: {"description": "该版本不存在"}},
)
def get_product_by_version(version: int) -> ProductResponse:
    return _read_product(version)


def _read_product(version: int | None) -> ProductResponse:
    store = _publish_store()
    try:
        snapshot = store.get_product(version)
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail=f"发布存储暂不可用: {exc}"
        ) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="成品版本不存在")
    return ProductResponse(
        version=snapshot["version"],  # type: ignore[arg-type]
        request_id=snapshot["request_id"],  # type: ignore[arg-type]
        row_count=snapshot["row_count"],  # type: ignore[arg-type]
        published_at=snapshot["published_at"],  # type: ignore[arg-type]
        product=snapshot["rows"],  # type: ignore[arg-type]
    )


@app.get("/", include_in_schema=False)
def root() -> Response:
    return Response(
        content='{"service":"incremental-net","docs":"/docs"}',
        media_type="application/json",
    )
