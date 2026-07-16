from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _normalize_capabilities(capabilities: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if capabilities is None:
        return ()
    if isinstance(capabilities, str):
        items = [item.strip() for item in capabilities.split(",")]
    else:
        items = [str(item).strip() for item in capabilities]
    return tuple(item for item in items if item)


@dataclass(frozen=True)
class ReleaseSpec:
    release_tag: str
    image_digest: str
    supported_api_versions: tuple[str, ...]
    capabilities: tuple[str, ...]
    notes: str = ""
    build_state: str = "built"
    health_state: str = "unknown"
    readiness_state: str = "unknown"


@dataclass(frozen=True)
class ProjectAssignment:
    project_id: str
    release_tag: str
    requested_capability: str
    requested_capabilities: tuple[str, ...]
    requested_api_version: str
    updated_at: str


@dataclass(frozen=True)
class AuditEvent:
    id: int
    actor_type: str
    actor_id: str
    action: str
    outcome: str
    message: str
    payload: dict[str, Any]
    created_at: str


class EngineRegistry:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._memory_mode = str(self.db_path) == ":memory:"
        self._memory_conn: sqlite3.Connection | None = None
        if self._memory_mode:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_mode:
            assert self._memory_conn is not None
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            if not self._memory_mode:
                conn.close()

    def initialize(self, default_release: ReleaseSpec) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS engine_releases (
                    release_tag TEXT PRIMARY KEY,
                    image_digest TEXT NOT NULL,
                    supported_api_versions_json TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    build_state TEXT NOT NULL DEFAULT 'built',
                    health_state TEXT NOT NULL DEFAULT 'unknown',
                    readiness_state TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL,
                    validated_at TEXT,
                    promoted_at TEXT,
                    previous_release_tag TEXT
                );
                CREATE TABLE IF NOT EXISTS engine_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    active_release_tag TEXT,
                    previous_release_tag TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_dependencies (
                    project_id TEXT PRIMARY KEY,
                    release_tag TEXT NOT NULL,
                    requested_capability TEXT NOT NULL,
                    requested_capabilities_json TEXT NOT NULL DEFAULT '[]',
                    requested_api_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(release_tag) REFERENCES engine_releases(release_tag)
                );
                CREATE TABLE IF NOT EXISTS dependency_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    requested_capability TEXT NOT NULL,
                    requested_capabilities_json TEXT NOT NULL DEFAULT '[]',
                    requested_api_version TEXT NOT NULL,
                    requested_release_tag TEXT,
                    resolved_release_tag TEXT,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS release_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    from_release_tag TEXT,
                    to_release_tag TEXT,
                    reason TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            for table_name in ("project_dependencies", "dependency_requests"):
                columns = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                }
                if "requested_capabilities_json" not in columns:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN requested_capabilities_json TEXT NOT NULL DEFAULT '[]'"
                    )
            if not self.get_release(default_release.release_tag, conn):
                self._upsert_release(conn, default_release, build_state="promoted", health_state="healthy", readiness_state="ready")
                now = _utc_now()
                conn.execute(
                    "INSERT OR REPLACE INTO engine_state (id, active_release_tag, previous_release_tag, updated_at) VALUES (1, ?, ?, ?)",
                    (default_release.release_tag, None, now),
                )
            elif not self.get_active_release(conn):
                now = _utc_now()
                conn.execute(
                    "INSERT OR REPLACE INTO engine_state (id, active_release_tag, previous_release_tag, updated_at) VALUES (1, ?, ?, ?)",
                    (default_release.release_tag, None, now),
                )
            conn.commit()

    def _upsert_release(
        self,
        conn: sqlite3.Connection,
        release: ReleaseSpec,
        build_state: str,
        health_state: str,
        readiness_state: str,
    ) -> None:
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO engine_releases (
                release_tag, image_digest, supported_api_versions_json, capabilities_json,
                notes, build_state, health_state, readiness_state, created_at,
                validated_at, promoted_at, previous_release_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(release_tag) DO UPDATE SET
                image_digest=excluded.image_digest,
                supported_api_versions_json=excluded.supported_api_versions_json,
                capabilities_json=excluded.capabilities_json,
                notes=excluded.notes,
                build_state=excluded.build_state,
                health_state=excluded.health_state,
                readiness_state=excluded.readiness_state,
                validated_at=excluded.validated_at,
                promoted_at=excluded.promoted_at,
                previous_release_tag=excluded.previous_release_tag
            """,
            (
                release.release_tag,
                release.image_digest,
                _dump_json(list(release.supported_api_versions)),
                _dump_json(list(release.capabilities)),
                release.notes,
                build_state,
                health_state,
                readiness_state,
                now,
                now if build_state in {"validated", "promoted"} else None,
                now if build_state == "promoted" else None,
                None,
            ),
        )

    def register_release(self, release: ReleaseSpec, actor_type: str = "ci", actor_id: str = "pipeline") -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            existing = self.get_release(release.release_tag, conn)
            if existing:
                if (
                    existing["image_digest"] != release.image_digest
                    or tuple(existing["supported_api_versions"]) != tuple(release.supported_api_versions)
                    or tuple(existing["capabilities"]) != tuple(release.capabilities)
                ):
                    raise ValueError(f"Release {release.release_tag} already exists with different immutable metadata.")
                return existing
            self._upsert_release(conn, release, build_state=release.build_state, health_state=release.health_state, readiness_state=release.readiness_state)
            self._audit(conn, actor_type, actor_id, "release.register", "ok", "Registered immutable release.", {
                "release_tag": release.release_tag,
                "image_digest": release.image_digest,
                "supported_api_versions": list(release.supported_api_versions),
                "capabilities": list(release.capabilities),
            })
            conn.commit()
            return self.get_release(release.release_tag, conn) or {}

    def mark_release_validated(
        self,
        release_tag: str,
        healthy: bool,
        ready: bool,
        validation_report: dict[str, Any] | None = None,
        actor_type: str = "ci",
        actor_id: str = "pipeline",
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            release = self.get_release(release_tag, conn)
            if not release:
                raise ValueError(f"Unknown release tag: {release_tag}")
            state = conn.execute(
                "SELECT active_release_tag, previous_release_tag FROM engine_state WHERE id = 1"
            ).fetchone()
            now = _utc_now()
            build_state = "validated" if healthy and ready else "failed"
            conn.execute(
                """
                UPDATE engine_releases
                SET build_state = ?,
                    health_state = ?,
                    readiness_state = ?,
                    validated_at = ?,
                    previous_release_tag = previous_release_tag
                WHERE release_tag = ?
                """,
                (
                    build_state,
                    "healthy" if healthy else "unhealthy",
                    "ready" if ready else "not_ready",
                    now,
                    release_tag,
                ),
            )
            rollback_result: dict[str, Any] | None = None
            if (not healthy or not ready) and state and state["active_release_tag"] == release_tag and state["previous_release_tag"]:
                rollback_result = self._rollback_previous_release_locked(
                    conn,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    reason="Automatic rollback triggered by failed validation.",
                )
            self._audit(
                conn,
                actor_type,
                actor_id,
                "release.validate",
                "ok" if healthy and ready else "failed",
                "Release validation recorded.",
                {
                    "release_tag": release_tag,
                    "healthy": healthy,
                    "ready": ready,
                    "validation_report": validation_report or {},
                    "rollback_triggered": bool(rollback_result),
                },
            )
            conn.commit()
            if rollback_result:
                rollback_result["validated_release_tag"] = release_tag
                rollback_result["rollback_triggered"] = True
                rollback_result["validation_report"] = validation_report or {}
                return rollback_result
            validated_release = self.get_release(release_tag, conn) or {}
            validated_release["rollback_triggered"] = False
            return validated_release

    def get_release(self, release_tag: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        own_conn = conn is None
        if own_conn:
            conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM engine_releases WHERE release_tag = ?", (release_tag,)).fetchone()
            if not row:
                return None
            return self._row_to_release(row)
        finally:
            if own_conn and not self._memory_mode:
                conn.close()

    def list_releases(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM engine_releases ORDER BY created_at DESC").fetchall()
        return [self._row_to_release(row) for row in rows]

    def get_active_release(self, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        own_conn = conn is None
        if own_conn:
            conn = self._connect()
        try:
            row = conn.execute(
                "SELECT active_release_tag, previous_release_tag, updated_at FROM engine_state WHERE id = 1"
            ).fetchone()
            if not row or not row["active_release_tag"]:
                return None
            release = self.get_release(row["active_release_tag"], conn)
            if not release:
                return None
            release["previous_release_tag"] = row["previous_release_tag"]
            release["active_since"] = row["updated_at"]
            return release
        finally:
            if own_conn and not self._memory_mode:
                conn.close()

    def list_projects_for_release(self, release_tag: str, exclude_project: str | None = None) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT project_id FROM project_dependencies WHERE release_tag = ? ORDER BY project_id",
                (release_tag,),
            ).fetchall()
        projects = [row["project_id"] for row in rows]
        if exclude_project:
            projects = [project_id for project_id in projects if project_id != exclude_project]
        return projects

    def assign_project(
        self,
        project_id: str,
        release_tag: str,
        requested_capability: str,
        requested_api_version: str,
        requested_capabilities: tuple[str, ...] | list[str] | str | None = None,
        actor_type: str = "engine",
        actor_id: str = "service",
    ) -> ProjectAssignment:
        with self._lock, self._connection() as conn:
            now = _utc_now()
            capabilities = _normalize_capabilities(requested_capabilities or requested_capability)
            primary_capability = capabilities[0] if capabilities else str(requested_capability).strip()
            conn.execute(
                """
                INSERT INTO project_dependencies (
                    project_id, release_tag, requested_capability, requested_capabilities_json, requested_api_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    release_tag=excluded.release_tag,
                    requested_capability=excluded.requested_capability,
                    requested_capabilities_json=excluded.requested_capabilities_json,
                    requested_api_version=excluded.requested_api_version,
                    updated_at=excluded.updated_at
                """,
                (project_id, release_tag, primary_capability, _dump_json(list(capabilities)), requested_api_version, now),
            )
            self._audit(
                conn,
                actor_type,
                actor_id,
                "project.assign",
                "ok",
                "Project assigned to release.",
                {
                    "project_id": project_id,
                    "release_tag": release_tag,
                    "requested_capability": primary_capability,
                    "requested_capabilities": list(capabilities),
                    "requested_api_version": requested_api_version,
                },
            )
            conn.commit()
            return ProjectAssignment(
                project_id=project_id,
                release_tag=release_tag,
                requested_capability=primary_capability,
                requested_capabilities=capabilities,
                requested_api_version=requested_api_version,
                updated_at=now,
            )

    def record_dependency_request(
        self,
        project_id: str,
        requested_capability: str,
        requested_api_version: str,
        requested_release_tag: str | None,
        status: str,
        message: str,
        *,
        requested_capabilities: tuple[str, ...] | list[str] | str | None = None,
        resolved_release_tag: str | None = None,
        actor_type: str = "engine",
        actor_id: str = "service",
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            now = _utc_now()
            capabilities = _normalize_capabilities(requested_capabilities or requested_capability)
            primary_capability = capabilities[0] if capabilities else str(requested_capability).strip()
            cur = conn.execute(
                """
                INSERT INTO dependency_requests (
                    project_id, requested_capability, requested_capabilities_json, requested_api_version,
                    requested_release_tag, resolved_release_tag, status, message,
                    created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    primary_capability,
                    _dump_json(list(capabilities)),
                    requested_api_version,
                    requested_release_tag,
                    resolved_release_tag,
                    status,
                    message,
                    now,
                    now if status in {"assigned", "blocked", "manual_required"} else None,
                ),
            )
            self._audit(
                conn,
                actor_type,
                actor_id,
                "dependency.request",
                status,
                message,
                {
                    "project_id": project_id,
                    "requested_capability": primary_capability,
                    "requested_capabilities": list(capabilities),
                    "requested_api_version": requested_api_version,
                    "requested_release_tag": requested_release_tag,
                    "resolved_release_tag": resolved_release_tag,
                },
            )
            conn.commit()
            return {
                "id": int(cur.lastrowid),
                "project_id": project_id,
                "requested_capability": primary_capability,
                "requested_capabilities": list(capabilities),
                "requested_api_version": requested_api_version,
                "requested_release_tag": requested_release_tag,
                "resolved_release_tag": resolved_release_tag,
                "status": status,
                "message": message,
                "created_at": now,
            }

    def list_pending_requests(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, requested_capability, requested_api_version,
                       requested_capabilities_json, requested_release_tag, resolved_release_tag, status, message,
                       created_at, resolved_at
                FROM dependency_requests
                ORDER BY id DESC
                """
            ).fetchall()
        pending_requests = []
        for row in rows:
            pending_requests.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "requested_capability": row["requested_capability"],
                    "requested_capabilities": _load_json(row["requested_capabilities_json"], [row["requested_capability"]]),
                    "requested_api_version": row["requested_api_version"],
                    "requested_release_tag": row["requested_release_tag"],
                    "resolved_release_tag": row["resolved_release_tag"],
                    "status": row["status"],
                    "message": row["message"],
                    "created_at": row["created_at"],
                    "resolved_at": row["resolved_at"],
                }
            )
        return pending_requests

    def compatible_releases(self, requested_capability: str, requested_api_version: str) -> list[dict[str, Any]]:
        requested_capabilities = _normalize_capabilities(requested_capability)
        if not requested_capabilities:
            return []
        matches = []
        for release in self.list_releases():
            if release["build_state"] not in {"validated", "promoted"}:
                continue
            if any(capability not in release["capabilities"] for capability in requested_capabilities):
                continue
            if requested_api_version not in release["supported_api_versions"]:
                continue
            matches.append(release)
        return matches

    def promote_release(
        self,
        release_tag: str,
        actor_type: str = "admin",
        actor_id: str = "admin",
        reason: str = "manual promotion",
        auto_reassign: bool = True,
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            release = self.get_release(release_tag, conn)
            if not release:
                raise ValueError(f"Unknown release tag: {release_tag}")
            if release["health_state"] != "healthy" or release["readiness_state"] != "ready":
                raise ValueError(f"Release {release_tag} has not passed health/readiness validation.")

            state = conn.execute(
                "SELECT active_release_tag, previous_release_tag FROM engine_state WHERE id = 1"
            ).fetchone()
            current_active = state["active_release_tag"] if state else None
            previous_known_good = current_active
            if current_active == release_tag:
                return {
                    "active_release": release,
                    "previous_release_tag": state["previous_release_tag"] if state else None,
                    "reassigned_projects": [],
                }

            if auto_reassign and current_active:
                rows = conn.execute(
                    """
                    SELECT project_id, requested_capability, requested_capabilities_json, requested_api_version
                    FROM project_dependencies
                    WHERE release_tag = ?
                    ORDER BY project_id
                    """,
                    (current_active,),
                ).fetchall()
                incompatible_projects = [
                    row["project_id"]
                    for row in rows
                    if any(
                        capability not in release["capabilities"]
                        for capability in _load_json(row["requested_capabilities_json"], [row["requested_capability"]])
                    )
                    or row["requested_api_version"] not in release["supported_api_versions"]
                ]
                if incompatible_projects:
                    self._audit(
                        conn,
                        actor_type,
                        actor_id,
                        "release.promote",
                        "blocked",
                        "Release promotion blocked by incompatible project requirements.",
                        {
                            "from_release_tag": current_active,
                            "to_release_tag": release_tag,
                            "incompatible_projects": incompatible_projects,
                            "reason": reason,
                        },
                    )
                    conn.commit()
                    raise ValueError(
                        "Promotion blocked because the release does not support all currently assigned project requirements."
                    )

            now = _utc_now()
            conn.execute(
                "INSERT OR REPLACE INTO engine_state (id, active_release_tag, previous_release_tag, updated_at) VALUES (1, ?, ?, ?)",
                (release_tag, current_active, now),
            )
            conn.execute(
                """
                UPDATE engine_releases
                SET build_state = 'promoted', promoted_at = ?, previous_release_tag = ?
                WHERE release_tag = ?
                """,
                (now, current_active, release_tag),
            )
            reassigned_projects: list[str] = []
            if auto_reassign and current_active:
                rows = conn.execute(
                    "SELECT project_id FROM project_dependencies WHERE release_tag = ? ORDER BY project_id",
                    (current_active,),
                ).fetchall()
                reassigned_projects = [row["project_id"] for row in rows]
                conn.execute(
                    "UPDATE project_dependencies SET release_tag = ?, updated_at = ? WHERE release_tag = ?",
                    (release_tag, now, current_active),
                )
                conn.execute(
                    """
                    UPDATE dependency_requests
                    SET status = 'assigned', resolved_release_tag = ?, resolved_at = ?
                    WHERE status = 'manual_required' AND requested_release_tag = ?
                    """,
                    (release_tag, now, release_tag),
                )
            conn.execute(
                """
                INSERT INTO release_history (action, from_release_tag, to_release_tag, reason, actor_type, actor_id, created_at)
                VALUES ('promote', ?, ?, ?, ?, ?, ?)
                """,
                (current_active, release_tag, reason, actor_type, actor_id, now),
            )
            self._audit(
                conn,
                actor_type,
                actor_id,
                "release.promote",
                "ok",
                "Release promoted.",
                {
                    "from_release_tag": current_active,
                    "to_release_tag": release_tag,
                    "reassigned_projects": reassigned_projects,
                    "reason": reason,
                },
            )
            conn.commit()
            return {
                "active_release": self.get_release(release_tag, conn),
                "previous_release_tag": previous_known_good,
                "reassigned_projects": reassigned_projects,
            }

    def rollback_previous_release(
        self,
        actor_type: str = "admin",
        actor_id: str = "admin",
        reason: str = "manual rollback",
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            rollback_result = self._rollback_previous_release_locked(
                conn,
                actor_type=actor_type,
                actor_id=actor_id,
                reason=reason,
            )
            conn.commit()
            return rollback_result

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, actor_type, actor_id, action, outcome, message, payload_json, created_at
                FROM audit_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "actor_type": row["actor_type"],
                "actor_id": row["actor_id"],
                "action": row["action"],
                "outcome": row["outcome"],
                "message": row["message"],
                "payload": _load_json(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_release": self.get_active_release(),
            "releases": self.list_releases(),
            "projects": self.list_project_assignments(),
            "pending_requests": self.list_pending_requests(),
            "audit_events": self.list_audit_events(),
            "release_history": self.list_release_history(),
        }

    def list_project_assignments(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT project_id, release_tag, requested_capability, requested_capabilities_json, requested_api_version, updated_at
                FROM project_dependencies
                ORDER BY project_id
                """
            ).fetchall()
        return [
            {
                "project_id": row["project_id"],
                "release_tag": row["release_tag"],
                "requested_capability": row["requested_capability"],
                "requested_capabilities": _load_json(row["requested_capabilities_json"], [row["requested_capability"]]),
                "requested_api_version": row["requested_api_version"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_release_history(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, action, from_release_tag, to_release_tag, reason, actor_type, actor_id, created_at
                FROM release_history
                ORDER BY id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _row_to_release(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "release_tag": row["release_tag"],
            "image_digest": row["image_digest"],
            "supported_api_versions": _load_json(row["supported_api_versions_json"], []),
            "capabilities": _load_json(row["capabilities_json"], []),
            "notes": row["notes"],
            "build_state": row["build_state"],
            "health_state": row["health_state"],
            "readiness_state": row["readiness_state"],
            "created_at": row["created_at"],
            "validated_at": row["validated_at"],
            "promoted_at": row["promoted_at"],
            "previous_release_tag": row["previous_release_tag"],
        }

    def _audit(
        self,
        conn: sqlite3.Connection,
        actor_type: str,
        actor_id: str,
        action: str,
        outcome: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO audit_events (actor_type, actor_id, action, outcome, message, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (actor_type, actor_id, action, outcome, message, _dump_json(payload), _utc_now()),
        )

    def _rollback_previous_release_locked(
        self,
        conn: sqlite3.Connection,
        actor_type: str,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        state = conn.execute(
            "SELECT active_release_tag, previous_release_tag FROM engine_state WHERE id = 1"
        ).fetchone()
        if not state or not state["previous_release_tag"]:
            raise ValueError("No previous known-good release is available.")

        current = state["active_release_tag"]
        previous = state["previous_release_tag"]
        now = _utc_now()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (id, active_release_tag, previous_release_tag, updated_at) VALUES (1, ?, ?, ?)",
            (previous, current, now),
        )
        conn.execute(
            "UPDATE project_dependencies SET release_tag = ?, updated_at = ? WHERE release_tag = ?",
            (previous, now, current),
        )
        conn.execute(
            "UPDATE engine_releases SET build_state = 'rolled_back' WHERE release_tag = ?",
            (current,),
        )
        conn.execute(
            """
            INSERT INTO release_history (action, from_release_tag, to_release_tag, reason, actor_type, actor_id, created_at)
            VALUES ('rollback', ?, ?, ?, ?, ?, ?)
            """,
            (current, previous, reason, actor_type, actor_id, now),
        )
        self._audit(
            conn,
            actor_type,
            actor_id,
            "release.rollback",
            "ok",
            "Rolled back to previous known-good release.",
            {"from_release_tag": current, "to_release_tag": previous, "reason": reason},
        )
        return {
            "active_release": self.get_release(previous, conn),
            "previous_release_tag": current,
        }
