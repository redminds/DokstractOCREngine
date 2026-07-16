from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.core.config import SETTINGS
from app.core.registry import EngineRegistry, ReleaseSpec


@dataclass(frozen=True)
class RequestOutcome:
    allowed: bool
    manual_intervention_required: bool
    message: str
    project_id: str
    requested_capability: str
    requested_capabilities: list[str]
    requested_api_version: str
    active_release: dict[str, Any] | None
    compatible_release: dict[str, Any] | None
    assigned_release: dict[str, Any] | None
    pending_request: dict[str, Any] | None
    blocking_projects: list[str]


class OCREngineService:
    def __init__(self, registry: EngineRegistry):
        self.registry = registry

    def initialize(self) -> None:
        default_release = ReleaseSpec(
            release_tag=SETTINGS.default_release_tag,
            image_digest=SETTINGS.default_image_digest,
            supported_api_versions=SETTINGS.default_supported_api_versions,
            capabilities=SETTINGS.default_capabilities,
            notes="Default immutable engine release baked into the container.",
            build_state="promoted",
            health_state="healthy",
            readiness_state="ready",
        )
        self.registry.initialize(default_release)

    def request_dependency(
        self,
        project_id: str,
        capability: str | None = None,
        api_version: str = "v1",
        capabilities: tuple[str, ...] | list[str] | None = None,
    ) -> RequestOutcome:
        requested_capabilities = tuple(str(item).strip() for item in (capabilities or (capability or "",)) if str(item).strip())
        primary_capability = requested_capabilities[0] if requested_capabilities else (capability or "").strip()
        active = self.registry.get_active_release()
        if not active:
            raise HTTPException(status_code=503, detail="No active engine release is configured.")

        if requested_capabilities and all(capability in active["capabilities"] for capability in requested_capabilities) and api_version in active["supported_api_versions"]:
            assignment = self.registry.assign_project(
                project_id=project_id,
                release_tag=active["release_tag"],
                requested_capability=primary_capability,
                requested_api_version=api_version,
                requested_capabilities=requested_capabilities,
            )
            pending = self.registry.record_dependency_request(
                project_id=project_id,
                requested_capability=primary_capability,
                requested_capabilities=requested_capabilities,
                requested_api_version=api_version,
                requested_release_tag=active["release_tag"],
                resolved_release_tag=active["release_tag"],
                status="assigned",
                message="Assigned to the currently active immutable release.",
            )
            return self._outcome(
                allowed=True,
                manual=False,
                message="Assigned to the currently active immutable release.",
                project_id=project_id,
                requested_capability=primary_capability,
                requested_capabilities=list(requested_capabilities),
                requested_api_version=api_version,
                active_release=active,
                compatible_release=active,
                assigned_release=active,
                pending_request=pending,
                blocking_projects=[],
            )

        compatible = self.registry.compatible_releases(requested_capabilities, api_version)
        compatible_release = compatible[0] if compatible else None
        blocking_projects = self.registry.list_projects_for_release(active["release_tag"], exclude_project=project_id)

        if compatible_release:
            pending = self.registry.record_dependency_request(
                project_id=project_id,
                requested_capability=primary_capability,
                requested_capabilities=requested_capabilities,
                requested_api_version=api_version,
                requested_release_tag=compatible_release["release_tag"],
                status="manual_required",
                resolved_release_tag=None,
                message="Compatible release exists, but it is not the active production image yet.",
            )
            return self._outcome(
                allowed=False,
                manual=True,
                message="Compatible release exists, but admin promotion is required before assignment.",
                project_id=project_id,
                requested_capability=primary_capability,
                requested_capabilities=list(requested_capabilities),
                requested_api_version=api_version,
                active_release=active,
                compatible_release=compatible_release,
                assigned_release=None,
                pending_request=pending,
                blocking_projects=blocking_projects,
            )

        pending = self.registry.record_dependency_request(
            project_id=project_id,
            requested_capability=primary_capability,
            requested_capabilities=requested_capabilities,
            requested_api_version=api_version,
            requested_release_tag=None,
            status="blocked",
            resolved_release_tag=None,
            message="No immutable release currently supports the requested capability and API version.",
        )
        return self._outcome(
            allowed=False,
            manual=False,
            message="No immutable release currently supports the requested capability and API version.",
            project_id=project_id,
            requested_capability=primary_capability,
            requested_capabilities=list(requested_capabilities),
            requested_api_version=api_version,
            active_release=active,
            compatible_release=None,
            assigned_release=None,
            pending_request=pending,
            blocking_projects=blocking_projects,
        )

    def register_release(
        self,
        release_tag: str,
        image_digest: str,
        supported_api_versions: tuple[str, ...],
        capabilities: tuple[str, ...],
        notes: str = "",
    ) -> dict[str, Any]:
        release = ReleaseSpec(
            release_tag=release_tag,
            image_digest=image_digest,
            supported_api_versions=supported_api_versions,
            capabilities=capabilities,
            notes=notes,
        )
        try:
            return self.registry.register_release(release)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def validate_release(
        self,
        release_tag: str,
        healthy: bool,
        ready: bool,
        validation_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self.registry.mark_release_validated(
                release_tag=release_tag,
                healthy=healthy,
                ready=ready,
                validation_report=validation_report,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def promote_release(self, release_tag: str, reason: str = "manual promotion") -> dict[str, Any]:
        try:
            return self.registry.promote_release(release_tag, reason=reason)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def rollback_release(self, reason: str = "manual rollback") -> dict[str, Any]:
        try:
            return self.registry.rollback_previous_release(reason=reason)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def snapshot(self) -> dict[str, Any]:
        return self.registry.snapshot()

    def audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.registry.list_audit_events(limit)

    def _outcome(
        self,
        *,
        allowed: bool,
        manual: bool,
        message: str,
        project_id: str,
        requested_capability: str,
        requested_capabilities: list[str],
        requested_api_version: str,
        active_release: dict[str, Any] | None,
        compatible_release: dict[str, Any] | None,
        assigned_release: dict[str, Any] | None,
        pending_request: dict[str, Any] | None,
        blocking_projects: list[str],
    ) -> RequestOutcome:
        return RequestOutcome(
            allowed=allowed,
            manual_intervention_required=manual,
            message=message,
            project_id=project_id,
            requested_capability=requested_capability,
            requested_capabilities=requested_capabilities,
            requested_api_version=requested_api_version,
            active_release=active_release,
            compatible_release=compatible_release,
            assigned_release=assigned_release,
            pending_request=pending_request,
            blocking_projects=blocking_projects,
        )
