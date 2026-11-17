"""Durable, versioned storage for published net products.

Storage is a single SQLite database file (path from ``PUBLISH_DB_PATH``)
mounted on a Docker volume.  Every read and write uses a short-lived
connection opened in autocommit mode, with ``busy_timeout`` so concurrent
writers are serialised by SQLite's file lock instead of failing immediately.

Consistency model
-----------------

A publication is written in **one transaction**:

1. assert the observed ``base_version`` is still the current version;
2. bump ``state.current_version``;
3. insert one ``version`` row and all ``product`` rows for it;
4. record the ``request`` idempotency entry.

Because all four steps commit atomically, readers never observe a partially
published version: a read selects the current version number and its rows
from the same database snapshot, so it returns either the complete previous
version or the complete new one.

Concurrent publishers that observed the same ``base_version`` both attempt
``UPDATE state SET current_version = ... WHERE current_version = base``;
SQLite grants the write lock to one writer, and the loser re-checks inside
its own ``BEGIN IMMEDIATE`` transaction and raises
:class:`VersionConflict`.  Exactly one succeeds.

Failures (illegal input, out-of-range net amount, conflict, derivation error)
occur before or inside the rolled-back transaction, so the product and the
request log keep their previous contents.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .publish import ProductRow

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS state (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        current_version INTEGER NOT NULL
    )
    """,
    """
    INSERT OR IGNORE INTO state(id, current_version) VALUES (1, 0)
    """,
    """
    CREATE TABLE IF NOT EXISTS version (
        version      INTEGER PRIMARY KEY,
        request_id   TEXT    NOT NULL UNIQUE,
        content_hash TEXT    NOT NULL,
        row_count    INTEGER NOT NULL,
        published_at TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS product (
        version    INTEGER NOT NULL REFERENCES version(version),
        order_id   INTEGER NOT NULL,
        net_amount INTEGER NOT NULL,
        PRIMARY KEY (version, order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request (
        request_id   TEXT    PRIMARY KEY,
        content_hash TEXT    NOT NULL,
        version      INTEGER REFERENCES version(version),
        status       TEXT    NOT NULL,
        created_at   TEXT    NOT NULL
    )
    """,
)


class PublishError(Exception):
    """Base class for publication errors that must not change stored state."""


class VersionConflict(PublishError):
    """The observed version is no longer the current published version."""

    def __init__(self, expected_version: int, current_version: int) -> None:
        super().__init__(
            f"所见版本 {expected_version} 已失效，当前版本为 {current_version}"
        )
        self.expected_version = expected_version
        self.current_version = current_version


class RequestIdReused(PublishError):
    """The same request id was already used for different content."""


@dataclass(frozen=True)
class VersionInfo:
    version: int
    request_id: str
    row_count: int
    published_at: str


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    content_hash: str
    version: int | None
    status: str


@dataclass(frozen=True)
class PublishResult:
    version: int
    replayed: bool
    """True when an identical request id/content was already published."""
    product: list[ProductRow]


class PublishStore:
    """SQLite-backed published-product store.

    ``check_same_thread`` is left at its default (True); every operation
    opens its own connection on the calling thread, so no connection is
    shared between FastAPI worker threads.
    """

    #: Schema DDL is idempotent (``IF NOT EXISTS`` / ``INSERT OR IGNORE``);
    #: the process-wide lock only avoids concurrent ``journal_mode`` changes,
    #: which take a write lock and can otherwise contend at startup.
    _schema_lock = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- connection helpers ---------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        # Cheap per-connection check: the file may have been created by an
        # earlier process/container, so schema presence is decided by the
        # database itself, never by an in-memory flag alone.
        if _has_schema(conn):
            return
        with self._schema_lock:
            if _has_schema(conn):
                return
            mode = conn.execute("PRAGMA journal_mode").fetchone()
            if mode is None or str(mode[0]).lower() != "wal":
                conn.execute("PRAGMA journal_mode = WAL")
            for statement in SCHEMA:
                conn.execute(statement)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -- reads -----------------------------------------------------------
    def current_version(self) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT current_version FROM state WHERE id = 1"
            ).fetchone()
        return int(row[0]) if row else 0

    def list_versions(self) -> list[VersionInfo]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT version, request_id, row_count, published_at "
                "FROM version ORDER BY version"
            ).fetchall()
        return [VersionInfo(*row) for row in rows]

    def version_overview(self) -> tuple[int, list[VersionInfo]]:
        """Current version and the version list from one consistent snapshot."""
        with self._connection() as conn:
            conn.execute("BEGIN")
            try:
                row = conn.execute(
                    "SELECT current_version FROM state WHERE id = 1"
                ).fetchone()
                current = int(row[0]) if row else 0
                rows = conn.execute(
                    "SELECT version, request_id, row_count, published_at "
                    "FROM version ORDER BY version"
                ).fetchall()
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
        return current, [VersionInfo(*r) for r in rows]

    def get_product(self, version: int | None = None) -> dict[str, object] | None:
        """Return a complete product snapshot, or ``None`` if it does not exist.

        ``None`` means "read the current version".  The version number and
        its rows are read from one connection; ``BEGIN`` (deferred) gives a
        stable snapshot for the duration of the read, so concurrent
        publishers cannot mix rows from two versions.
        """
        with self._connection() as conn:
            conn.execute("BEGIN")
            try:
                if version is None:
                    row = conn.execute(
                        "SELECT current_version FROM state WHERE id = 1"
                    ).fetchone()
                    if row is None:
                        conn.execute("ROLLBACK")
                        return None
                    resolved_version = int(row[0])
                else:
                    resolved_version = int(version)
                meta = conn.execute(
                    "SELECT request_id, row_count, published_at "
                    "FROM version WHERE version = ?",
                    (resolved_version,),
                ).fetchone()
                if meta is None:
                    conn.execute("ROLLBACK")
                    return None
                rows = conn.execute(
                    "SELECT order_id, net_amount FROM product "
                    "WHERE version = ? ORDER BY order_id",
                    (resolved_version,),
                ).fetchall()
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
        if len(rows) != int(meta[1]):  # pragma: no cover - invariant check
            raise RuntimeError(
                f"版本 {resolved_version} 的成品行数不一致："
                f"元数据 {meta[1]}，实际 {len(rows)}"
            )
        return {
            "version": resolved_version,
            "request_id": meta[0],
            "row_count": int(meta[1]),
            "published_at": meta[2],
            "rows": [{"order_id": r[0], "net_amount": r[1]} for r in rows],
        }

    def get_request(self, request_id: str) -> RequestRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT request_id, content_hash, version, status "
                "FROM request WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return RequestRecord(*row) if row else None

    # -- publication -----------------------------------------------------
    def publish(
        self,
        *,
        base_version: int,
        request_id: str,
        content_hash: str,
        product: list[ProductRow],
    ) -> PublishResult:
        """Atomically publish ``product`` as the next version.

        Returns the already-published version when ``request_id`` was used
        before for byte-identical content.  Raises :class:`RequestIdReused`,
        :class:`VersionConflict` or :class:`PublishError` without changing
        stored state.
        """
        with self._write_transaction() as conn:
            previous = conn.execute(
                "SELECT content_hash, version FROM request WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if previous is not None:
                stored_hash, stored_version = previous
                if stored_hash != content_hash:
                    raise RequestIdReused(
                        f"请求标识 {request_id} 已用于不同内容（首次发布版本 "
                        f"{stored_version}），拒绝"
                    )
                return PublishResult(
                    version=int(stored_version),
                    replayed=True,
                    product=self._read_product(conn, int(stored_version)),
                )

            row = conn.execute(
                "SELECT current_version FROM state WHERE id = 1"
            ).fetchone()
            current = int(row[0]) if row else 0
            if current != base_version:
                raise VersionConflict(base_version, current)

            new_version = current + 1
            published_at = _utc_now()
            # Compare-and-set under SQLite's write lock: a publisher that lost
            # the race re-reads the committed current_version here and fails.
            updated = conn.execute(
                "UPDATE state SET current_version = ? "
                "WHERE id = 1 AND current_version = ?",
                (new_version, current),
            )
            if updated.rowcount != 1:  # pragma: no cover - IMMEDIATE serialises
                raise VersionConflict(base_version, self._current_in(conn))

            conn.execute(
                "INSERT INTO version(version, request_id, content_hash, "
                "row_count, published_at) VALUES (?, ?, ?, ?, ?)",
                (new_version, request_id, content_hash, len(product), published_at),
            )
            conn.executemany(
                "INSERT INTO product(version, order_id, net_amount) VALUES (?, ?, ?)",
                [(new_version, r.order_id, r.net_amount) for r in product],
            )
            conn.execute(
                "INSERT INTO request(request_id, content_hash, version, status, "
                "created_at) VALUES (?, ?, ?, 'published', ?)",
                (request_id, content_hash, new_version, published_at),
            )
            return PublishResult(
                version=new_version, replayed=False, product=list(product)
            )

    @staticmethod
    def _read_product(conn: sqlite3.Connection, version: int) -> list[ProductRow]:
        rows = conn.execute(
            "SELECT order_id, net_amount FROM product "
            "WHERE version = ? ORDER BY order_id",
            (version,),
        ).fetchall()
        return [ProductRow(order_id=r[0], net_amount=r[1]) for r in rows]

    @staticmethod
    def _current_in(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT current_version FROM state WHERE id = 1").fetchone()
        return int(row[0]) if row else 0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _has_schema(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name IN "
        "('state', 'version', 'product', 'request')"
    ).fetchone()
    return bool(row) and int(row[0]) == 4
