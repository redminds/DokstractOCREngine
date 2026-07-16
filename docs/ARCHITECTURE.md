# Architecture

The service has two responsibilities:

1. maintain a registry of immutable OCR engine releases and project dependencies
2. validate, promote, and roll back versioned Docker image releases

## Core flow

1. A project submits a dependency request with a capability and supported engine API version.
2. The registry checks whether the active release already satisfies the request.
3. If the active release is compatible, the project is assigned to it and the request is recorded.
4. If a compatible future release exists, the request is marked manual and an audit event is recorded.
5. Admins validate a release, then promote it atomically.
6. Promotion reassigns projects only after the release becomes the active release.
7. Rollback returns the active release to the previous known-good image and restores project mappings.

## Manual override

Admin approval is required to promote a new image release into the active slot.
