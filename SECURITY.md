# Security Policy

CUDA Kernel Academy is an educational CUDA repository, but security issues still matter, especially in build scripts, dependency handling, and example code that may be reused elsewhere.

## Supported versions

| Version | Status |
| --- | --- |
| `1.x` | supported |
| `< 1.0` | unsupported |

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Please use GitHub private vulnerability reporting:

- <https://github.com/AICL-Lab/cuda-foundations/security/advisories/new>

Include:

- a clear description of the issue
- affected files or modules
- realistic impact
- reproduction steps
- a suggested fix if you have one

## Scope

The most relevant security areas for this repository are:

- dependency and workflow configuration
- shell scripts and developer tooling
- unsafe example patterns that could be copied into downstream code
- GPU memory handling in reusable components

## Expectations

- prefer responsible disclosure
- give maintainers time to assess and fix the issue
- avoid publishing exploit details before a fix or mitigation is available

## Notes for users

- treat example code as learning material, not automatically production-hardened code
- validate dimensions, pointers, and resource lifetimes in downstream integrations
- clear sensitive GPU buffers explicitly when your own application requires it
