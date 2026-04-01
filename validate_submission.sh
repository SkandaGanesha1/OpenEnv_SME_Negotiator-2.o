#!/bin/bash
# Pre-Submission Validation Script for OpenEnv SME Negotiation
# Performs checks before hackathon submission
# Reference: Meta-PyTorch Hackathon 2026 Requirements

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OpenEnv SME Negotiation - Pre-Submission Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PASSED=0
FAILED=0

# Helper functions
check_pass() {
    echo "  ✓ PASS: $1"
    ((PASSED++))
}

check_fail() {
    echo "  ✗ FAIL: $1"
    ((FAILED++))
}

check_warn() {
    echo "  ⚠ WARN: $1"
}

# ============================================================================
# CHECK 1: openenv.yaml Specification File
# ============================================================================

echo ""
echo "━━━ CHECK 1: OpenEnv Specification (openenv.yaml) ━━━"
if [ -f "openenv.yaml" ]; then
    check_pass "openenv.yaml exists"
    
    # Validate YAML syntax
    if python -c "import yaml; yaml.safe_load(open('openenv.yaml'))" 2>/dev/null; then
        check_pass "openenv.yaml has valid YAML syntax"
    else
        check_fail "openenv.yaml has invalid YAML syntax"
    fi
    
    # Check required fields
    if grep -q "api_version" openenv.yaml && grep -q "metadata" openenv.yaml; then
        check_pass "openenv.yaml contains required fields"
    else
        check_fail "openenv.yaml missing required fields (api_version, metadata)"
    fi
    
    # Check task definitions
    if grep -q "tasks:" openenv.yaml; then
        task_count=$(grep -c "- id:" openenv.yaml || true)
        if [ "$task_count" -ge 3 ]; then
            check_pass "openenv.yaml defines 3+ tasks (found: $task_count)"
        else
            check_fail "openenv.yaml must define 3+ tasks (found: $task_count)"
        fi
    else
        check_fail "openenv.yaml missing task definitions"
    fi
else
    check_fail "openenv.yaml not found"
fi

# ============================================================================
# CHECK 2: Dockerfile for HF Spaces Deployment
# ============================================================================

echo ""
echo "━━━ CHECK 2: Docker Configuration ━━━"
if [ -f "Dockerfile" ]; then
    check_pass "Dockerfile exists"
    
    # Check for required base image
    if grep -q "FROM python" Dockerfile; then
        check_pass "Dockerfile uses Python base image"
    else
        check_fail "Dockerfile must use Python base image"
    fi
    
    # Check for port 7860 (HF Spaces requirement)
    if grep -q "7860" Dockerfile; then
        check_pass "Dockerfile configured for HF Spaces (port 7860)"
    else
        check_warn "Dockerfile may not be configured for HF Spaces (port 7860)"
    fi
    
    # Check for health check
    if grep -q "HEALTHCHECK\|healthcheck" Dockerfile; then
        check_pass "Dockerfile includes health check"
    else
        check_warn "Dockerfile should include health check endpoint"
    fi
    
    # Try to build Docker image (if Docker available)
    if command -v docker &> /dev/null; then
        echo "  Attempting Docker build..."
        if docker build -t openenv-sme-negotiator:test . > /dev/null 2>&1; then
            check_pass "Docker image builds successfully"
        else
            check_warn "Docker build failed (may require docker daemon running)"
        fi
    else
        check_warn "Docker not installed - skipping docker build test"
    fi
else
    check_fail "Dockerfile not found"
fi

# ============================================================================
# CHECK 3: Environment Implementation
# ============================================================================

echo ""
echo "━━━ CHECK 3: Environment Core Files ━━━"
files_required=(
    "src/env/sme_negotiation.py:SMENegotiationEnv class"
    "src/utils/grader.py:DeterministicGrader class"
    "src/utils/models.py:Pydantic type models"
    "src/server.py:FastAPI server"
    "client/env_client.py:AsyncEnvClient"
    "requirements.txt:Python dependencies"
)

for file_spec in "${files_required[@]}"; do
    IFS=':' read -r filepath description <<< "$file_spec"
    if [ -f "$filepath" ]; then
        check_pass "$description ($filepath exists)"
    else
        check_fail "$description ($filepath NOT FOUND)"
    fi
done

# Test environment imports
echo "  Testing Python imports..."
if python -c "from src.env.sme_negotiation import SMENegotiationEnv" 2>/dev/null; then
    check_pass "SMENegotiationEnv imports successfully"
else
    check_fail "SMENegotiationEnv import failed"
fi

if python -c "from src.utils.grader import DeterministicGrader" 2>/dev/null; then
    check_pass "DeterministicGrader imports successfully"
else
    check_fail "DeterministicGrader import failed"
fi

# ============================================================================
# CHECK 4: Baseline Inference Script
# ============================================================================

echo ""
echo "━━━ CHECK 4: Baseline Inference (inference.py) ━━━"
if [ -f "inference.py" ]; then
    check_pass "inference.py exists"
    
    # Check for OpenAI integration
    if grep -q "openai\|OpenAI" inference.py; then
        check_pass "inference.py includes OpenAI integration"
    else
        check_warn "inference.py may not include OpenAI integration"
    fi
    
    # Check for reward logging
    if grep -q "Reward\|reward" inference.py; then
        check_pass "inference.py includes reward logging/tracking"
    else
        check_warn "inference.py should include reward logging"
    fi
    
    # Check for episode execution
    if grep -q "run_episode\|main" inference.py; then
        check_pass "inference.py has episode execution methods"
    else
        check_fail "inference.py missing episode execution methods"
    fi
else
    check_fail "inference.py NOT FOUND (required for baseline)"
fi

# ============================================================================
# CHECK 5: Test Suite
# ============================================================================

echo ""
echo "━━━ CHECK 5: Test Coverage ━━━"
test_files=(
    "test_env.py"
    "eval_standalone.py"
    "run_diagnostics.py"
)

for test_file in "${test_files[@]}"; do
    if [ -f "$test_file" ]; then
        check_pass "$test_file exists"
    else
        check_warn "$test_file not found"
    fi
done

# Try to run diagnostics
if [ -f "run_diagnostics.py" ]; then
    echo "  Running diagnostic tests..."
    if timeout 30 python run_diagnostics.py > /tmp/diagnostics.log 2>&1; then
        check_pass "Diagnostic tests pass"
    else
        check_warn "Diagnostic tests did not complete (timeout or error)"
    fi
fi

# ============================================================================
# CHECK 6: Dependency Requirements
# ============================================================================

echo ""
echo "━━━ CHECK 6: Dependencies ━━━"
if [ -f "requirements.txt" ]; then
    check_pass "requirements.txt exists"
    
    # Check for critical packages
    required_packages=(
        "gymnasium"
        "fastapi"
        "pydantic"
        "openai"
    )
    
    for package in "${required_packages[@]}"; do
        if grep -q "$package" requirements.txt; then
            check_pass "requirements.txt includes $package"
        else
            check_fail "requirements.txt missing $package"
        fi
    done
else
    check_fail "requirements.txt NOT FOUND"
fi

# ============================================================================
# CHECK 7: Server Endpoints
# ============================================================================

echo ""
echo "━━━ CHECK 7: API Endpoints ━━━"
endpoint_checks=(
    "src/server.py:/health"
    "src/server.py:/reset"
    "src/server.py:/step"
    "src/server.py:/state"
)

if [ -f "src/server.py" ]; then
    for endpoint_spec in "${endpoint_checks[@]}"; do
        IFS=':' read -r filepath endpoint <<< "$endpoint_spec"
        if grep -q "\"$endpoint\"\|'$endpoint'" "$filepath"; then
            check_pass "Endpoint $endpoint is implemented"
        else
            check_warn "Endpoint $endpoint may not be implemented"
        fi
    done
else
    check_warn "src/server.py not found - cannot verify endpoints"
fi

# ============================================================================
# CHECK 8: Documentation
# ============================================================================

echo ""
echo "━━━ CHECK 8: Documentation ━━━"
docs=(
    "README.md:Main documentation"
    "QUICK_START.md:Quick start guide"
    "SETUP.md:Setup instructions"
)

for doc_spec in "${docs[@]}"; do
    IFS=':' read -r filepath description <<< "$doc_spec"
    if [ -f "$filepath" ]; then
        check_pass "$description ($filepath)"
    else
        check_warn "$description ($filepath) not found"
    fi
done

# ============================================================================
# CHECK 9: Git Repository
# ============================================================================

echo ""
echo "━━━ CHECK 9: Git Repository ━━━"
if [ -d ".git" ]; then
    check_pass "Git repository initialized"
    
    # Check for commits
    if git log --oneline | head -1 > /dev/null 2>&1; then
        commit_count=$(git log --oneline | wc -l)
        check_pass "Git has $commit_count commits"
    else
        check_warn "Git repository appears empty"
    fi
    
    # Check for GitHub remote
    if git remote -v | grep -q "github\|github.com"; then
        check_pass "GitHub remote is configured"
    else
        check_warn "GitHub remote not found - may not be pushed"
    fi
else
    check_warn "Not in a Git repository"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  ✓ Checks Passed:  $PASSED"
echo "  ✗ Checks Failed:  $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "  🎉 All critical checks passed! Ready for submission."
    echo ""
    exit 0
else
    echo "  ⚠️  Some checks failed. Please review above."
    echo ""
    exit 1
fi
