# Security Policy

## Supported Versions

The following versions of CUDA Kernel Academy are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security vulnerabilities through:

1. **GitHub Security Advisories** (preferred): Use [GitHub's private vulnerability reporting](https://github.com/LessUp/cuda-kernel-academy/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: Potential impact and severity assessment
- **Reproduction**: Steps to reproduce the issue
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have a suggested fix or mitigation

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial assessment**: We will provide an initial assessment within 7 days
- **Resolution**: We aim to resolve critical issues within 30 days

### Disclosure Policy

- We follow responsible disclosure practices
- We will coordinate with you on disclosure timing
- We will credit reporters in security advisories (unless you prefer anonymity)

## Security Best Practices

When using CUDA Kernel Academy in your projects:

### Memory Safety

- Always use CUDA error checking macros (`TC_CUDA_CHECK`)
- Validate input dimensions before kernel launches
- Use bounds checking in debug builds

### Input Validation

```cpp
// Example: Validate matrix dimensions
if (M <= 0 || N <= 0 || K <= 0) {
    throw std::invalid_argument("Matrix dimensions must be positive");
}

// Example: Check for null pointers
if (A == nullptr || B == nullptr || C == nullptr) {
    throw std::invalid_argument("Matrix pointers cannot be null");
}
```

### Resource Management

- Use RAII wrappers for GPU memory (`DeviceMemory<T>`)
- Always free allocated resources
- Handle CUDA errors gracefully

## Known Security Considerations

### GPU Memory

- GPU memory is not automatically cleared on allocation
- Sensitive data should be explicitly zeroed before deallocation
- Consider using `cudaMemset` for sensitive buffers

### Shared Memory

- Shared memory persists between kernel launches on the same SM
- Be cautious with sensitive data in shared memory

### Error Messages

- Error messages may contain system information
- Consider sanitizing error output in production environments

## Security Updates

Security updates will be released as:

- **Patch releases** for critical vulnerabilities
- **Minor releases** for non-critical security improvements

Subscribe to releases to stay informed about security updates.

## Contact

For security-related questions that are not vulnerabilities, you can:

- Open a [Discussion](https://github.com/LessUp/cuda-kernel-academy/discussions)
- Contact maintainers through GitHub

---

Thank you for helping keep CUDA Kernel Academy secure!
