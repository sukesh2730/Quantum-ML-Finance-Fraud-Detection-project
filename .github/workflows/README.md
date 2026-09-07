# GitHub Actions CI/CD Workflows

This directory contains automated workflows for continuous integration and testing.

## 📋 Available Workflows

### 1. `pytest.yml` - Basic Python Tests

**Triggers:** Push and Pull Requests to `main` branch

**What it does:**
- Sets up Python 3.10 environment
- Installs project dependencies
- Runs pytest test suite
- Reports test results

**Status:** ![Python Tests](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/pytest.yml/badge.svg)

**Use case:** Quick validation for commits and PRs

---

### 2. `ci.yml` - Comprehensive CI Pipeline

**Triggers:** Push to `main`/`develop` branches, Pull Requests to `main`

**Jobs:**

#### a. Test Matrix
- **Python Versions:** 3.8, 3.9, 3.10, 3.11
- **OS:** Ubuntu Latest
- **Runs:** Full test suite on all Python versions
- **Linting:** Optional flake8 checks
- **Caching:** pip dependencies cached for faster builds

#### b. Code Coverage
- **Python Version:** 3.10
- **Coverage Tool:** pytest-cov
- **Reports:** XML format uploaded to Codecov
- **Displays:** Terminal coverage summary

#### c. Build Check
- **Validates:** Package can be built successfully
- **Uses:** Python build tools
- **Artifacts:** Creates wheel and source distributions

**Status:** ![CI Pipeline](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/ci.yml/badge.svg)

**Use case:** Comprehensive validation before merging

---

## 🚀 Workflow Features

### Caching
- Pip packages are cached to speed up subsequent runs
- Cache key includes Python version and requirements.txt hash
- Significantly reduces build time

### Fail-Fast Strategy
- Disabled (`fail-fast: false`) to see all test results
- All Python versions tested even if one fails

### Error Handling
- Linting errors don't fail the build (continue-on-error)
- Build artifacts are always checked
- Summary messages always displayed

---

## 📊 Status Badges

Add these badges to your README:

```markdown
[![Python Tests](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/pytest.yml/badge.svg)](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/pytest.yml)

[![CI Pipeline](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/ci.yml/badge.svg)](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project/actions/workflows/ci.yml)
```

---

## 🔧 Local Testing

Before pushing, test locally:

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/quantum_fraud_detector --cov-report=term

# Lint code
flake8 src/ tests/ --max-line-length=127
```

---

## 🛠️ Customization

### Change Python Versions

Edit `ci.yml`:
```yaml
matrix:
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Add More OS Support

```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest]
  python-version: ['3.10']
```

### Adjust Test Verbosity

```yaml
- name: Run tests
  run: pytest tests/ -v --tb=short --maxfail=5
```

### Enable Strict Linting

Change `continue-on-error: true` to `continue-on-error: false`

---

## 📈 CI/CD Best Practices

### ✅ What's Implemented

1. **Automated Testing**: Every push/PR triggers tests
2. **Multi-Version Support**: Tests on Python 3.8-3.11
3. **Code Coverage**: Tracks test coverage metrics
4. **Build Validation**: Ensures package builds correctly
5. **Fast Feedback**: Caching speeds up builds
6. **Clear Status**: Badges show build health

### 🎯 Recommended Additions

1. **Dependency Scanning**: Add Dependabot
2. **Security Scanning**: Add CodeQL analysis
3. **Performance Testing**: Benchmark quantum operations
4. **Deploy Previews**: Auto-deploy to staging
5. **Release Automation**: Publish to PyPI on tags

---

## 🐛 Troubleshooting

### Tests Fail in CI but Pass Locally

**Possible causes:**
- Environment differences
- Missing dependencies
- Race conditions

**Solutions:**
1. Check GitHub Actions logs
2. Replicate CI environment locally:
   ```bash
   docker run -it python:3.10 /bin/bash
   # Install and test
   ```
3. Add debug output to tests

### Slow Build Times

**Solutions:**
1. Verify caching is working
2. Reduce test dataset sizes
3. Use faster quantum simulator
4. Split tests into parallel jobs

### Coverage Upload Fails

**Check:**
1. Codecov token configured (if private repo)
2. XML coverage report generated
3. Network connectivity in Actions

---

## 📝 Workflow File Structure

```yaml
name: Workflow Name
on: [triggers]
jobs:
  job-name:
    runs-on: runner
    strategy: matrix
    steps:
      - checkout
      - setup
      - install
      - test
      - report
```

---

## 🔗 Useful Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
- [Python Packaging Guide](https://packaging.python.org/)

---

## 📞 Support

If CI/CD workflows fail:
1. Check GitHub Actions logs
2. Review recent changes
3. Test locally first
4. Open an issue with workflow logs

---

**Last Updated:** 2025-01-XX
**Maintainer:** Project Team
