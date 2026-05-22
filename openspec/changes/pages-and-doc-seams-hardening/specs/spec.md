# Spec: Pages and Docs Seams Hardening

## Requirement 1 - Canonical public metadata

Public-facing documentation and metadata MUST use one canonical GitHub repo URL and one canonical Pages URL.

### Scenario: reader opens public docs

- **Given** a reader opens README files, docs homepages, issue templates, or contributor docs
- **When** they follow repository or Pages links
- **Then** those links point to the same canonical repository and Pages site

## Requirement 2 - Stable docs root entrypoint

The docs root MUST be a stable landing page rather than a hardcoded client-side redirect shim.

### Scenario: reader lands on `/`

- **Given** a reader opens the docs root
- **When** the page renders
- **Then** it presents explicit locale choices and reading-path links
- **And** it does not rely on hardcoded `/cuda-kernel-academy/...` redirect logic

## Requirement 3 - Pages reading-path expansion

The documentation site MUST expose bilingual reading-path guidance for different reader goals.

### Scenario: reader wants a guided path

- **Given** a reader wants to study the repository by objective instead of by directory
- **When** they open the docs root or guides navigation
- **Then** they can reach a reading-path guide in both English and Chinese

## Requirement 4 - Inference docs/build alignment

The `04-inference-engine` README MUST describe artifacts that match the actual CMake targets.

### Scenario: reader follows inference README build instructions

- **Given** a reader opens `04-inference-engine/README.md`
- **When** they inspect documented artifacts and commands
- **Then** the binary names and targets match `04-inference-engine/CMakeLists.txt`
