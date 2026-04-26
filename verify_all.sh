#!/usr/bin/env bash
# Project compilation and verification script

echo "================================="
echo "OpenEnv SME Negotiation - Build & Verify"
echo "================================="
echo ""

# Check Python version
echo "[1] Checking Python environment..."
python --version
echo ""

# Check dependencies
echo "[2] Verifying dependencies..."
python -c "import numpy, pydantic, fastapi, gymnasium; print('All core dependencies OK')"
echo ""

# Run syntax check
echo "[3] Checking syntax of all Python files..."
python -m py_compile eval/run_eval.py client/env_client.py server/exploit_guard.py src/app.py src/env/sme_negotiation.py src/utils/models.py 2>&1 && echo "Syntax check: PASS" || echo "Syntax check: FAIL"
echo ""

# Run basic environment test
echo "[4] Running basic environment test..."
python test_env.py
echo ""

# Run evaluation
echo "[5] Running standalone evaluation..."
python eval_standalone.py
echo ""

# Run diagnostics
echo "[6] Running diagnostics..."
python run_diagnostics.py
echo ""

echo "================================="
echo "Verification complete!"
echo "================================="
