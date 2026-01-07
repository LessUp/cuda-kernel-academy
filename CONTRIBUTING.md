# 🤝 Contributing to CUDA Kernel Academy

感谢你对 CUDA Kernel Academy 的关注！我们欢迎各种形式的贡献。

Thank you for your interest in contributing to CUDA Kernel Academy! We welcome contributions of all kinds.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Types of Contributions

We welcome the following types of contributions:

- 🐛 **Bug fixes** - Fix issues in existing code
- ✨ **New features** - Add new kernels, optimizations, or modules
- 📝 **Documentation** - Improve or translate documentation
- 🧪 **Tests** - Add or improve test coverage
- ⚡ **Performance** - Optimize existing implementations
- 🎨 **Code quality** - Refactoring and code cleanup

### First-time Contributors

If you're new to the project, look for issues labeled:
- `good first issue` - Simple issues suitable for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements

## Development Setup

### Prerequisites

| Requirement | Minimum Version | Recommended |
|-------------|-----------------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.20 | 3.24+ |
| GCC/Clang | GCC 9 / Clang 10 | GCC 11+ |
| GPU | Volta (sm_70) | Ampere (sm_80+) |

### Clone and Build

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/cuda-kernel-academy.git
cd cuda-kernel-academy

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/cuda-kernel-academy.git

# Create a build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

### Setting Up Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files
```

## Making Changes

### Workflow

1. **Sync with upstream**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Make your changes**
   - Write code following our [Code Style](#code-style)
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   cd build
   make -j$(nproc)
   ctest --output-on-failure
   ```

5. **Commit your changes**
   - Follow our [Commit Message](#commit-messages) guidelines
   - Keep commits atomic and focused

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Code Style

### C++/CUDA Style

We follow the style defined in [docs/CODING_STYLE.md](docs/CODING_STYLE.md). Key points:

```cpp
// Namespaces: snake_case
namespace tensorcraft::kernels { }

// Classes/Structs: PascalCase
class GemmConfig { };

// Functions/Variables: snake_case
void launch_gemm(...);
int block_size;

// Constants: kPascalCase or UPPER_SNAKE_CASE
constexpr int kBlockSize = 32;
constexpr int TILE_SIZE = 16;

// Private members: trailing underscore
class MyClass {
private:
    int value_;
};
```

### Formatting

Use `clang-format` with the project's `.clang-format` configuration:

```bash
# Format a single file
clang-format -i src/my_file.cu

# Format all files
find . -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" -o -name "*.cuh" | xargs clang-format -i
```

### CUDA Best Practices

- Always use error checking macros (`TC_CUDA_CHECK`, `TC_CUDA_CHECK_LAST`)
- Use `__restrict__` for kernel pointer parameters
- Document kernel launch configurations
- Include performance benchmarks for new kernels

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or dependencies |
| `ci` | CI configuration |
| `chore` | Other changes |

### Examples

```
feat(gemm): add register tiling optimization

Implement 8x8 register tiling for improved performance.
Achieves 2x speedup on Ampere GPUs.

Closes #123
```

```
fix(attention): correct softmax numerical stability

Add max subtraction before exp() to prevent overflow
in flash attention implementation.
```

```
docs(readme): update installation instructions for CUDA 12
```

## Pull Request Process

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted with clang-format
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

### PR Template

When creating a PR, please fill out the template with:
- Description of changes
- Type of change (bug fix, feature, etc.)
- Related issues
- Testing performed

### Review Process

1. **Automated checks** - CI must pass
2. **Code review** - At least one maintainer approval
3. **Testing** - Verify on target GPU architectures
4. **Merge** - Squash and merge to main

### After Merge

- Delete your feature branch
- Sync your fork with upstream

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Environment**: CUDA version, GPU model, OS
- **Description**: Clear description of the bug
- **Reproduction**: Steps to reproduce
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages or logs

### Feature Requests

For feature requests, please describe:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered

### Security Issues

For security vulnerabilities, please see [SECURITY.md](SECURITY.md). Do NOT create public issues for security problems.

---

## 🙏 Thank You!

Your contributions make CUDA Kernel Academy better for everyone. We appreciate your time and effort!

If you have questions, feel free to:
- Open a [Discussion](https://github.com/OWNER/cuda-kernel-academy/discussions)
- Ask in an Issue
- Reach out to maintainers

Happy CUDA Hacking! 🚀
