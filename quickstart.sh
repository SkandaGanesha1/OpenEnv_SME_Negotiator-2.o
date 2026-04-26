#!/bin/bash
# Quick Start Script for OpenEnv SME Negotiator
# Run: bash quickstart.sh

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  OpenEnv SME Negotiator - Hackathon Quick Start Script     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "📍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check if pip is available
echo "📍 Checking pip..."
python3 -m pip --version > /dev/null 2>&1 || {
    echo "❌ pip is not available"
    exit 1
}
echo "   ✓ pip is available"

# Install dependencies
echo ""
echo "📦 Installing dependencies (this may take a few minutes)..."
python3 -m pip install -e . -q || {
    echo "❌ Failed to install dependencies"
    exit 1
}
echo "   ✓ Dependencies installed"

# Check for HF_TOKEN (Hugging Face Inference API key)
echo ""
echo "🔑 Checking HF_TOKEN..."
if [ -z "$HF_TOKEN" ]; then
    echo "   ⚠️  HF_TOKEN not set"
    echo ""
    echo "   To use baseline inference, set:"
    echo "   export HF_TOKEN='hf_...'"
    echo ""
    read -p "   Enter your Hugging Face token (or press Enter to skip): " HF_KEY
    
    if [ ! -z "$HF_KEY" ]; then
        export HF_TOKEN="$HF_KEY"
        
        # Save to .env file
        if [ ! -f .env ]; then
            cat > .env << EOF
HF_TOKEN=$HF_TOKEN
API_BASE_URL=https://router.huggingface.co/v1
OPENENV_BASE_URL=http://127.0.0.1:7860
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
OPENENV_IN_PROCESS=0
EOF
            echo "   ✓ Saved to .env file"
        fi
    fi
else
    echo "   ✓ HF_TOKEN is set"
fi

# Run diagnostics
echo ""
echo "🔍 Running diagnostics..."
python3 -c "from server.app import app; print('app OK:', app.title)" || {
    echo "⚠️  Import check failed — run from repo root with: pip install -e ."
}

# Print next steps
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete! Next Steps:                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  Start the server (in terminal 1):"
echo "   make server"
echo ""
echo "2️⃣  In another terminal, run baseline inference:"
echo "   make baseline"
echo ""
echo "3️⃣  Or run tests:"
echo "   make test"
echo ""
echo "📚 For more info, see:"
echo "   - README.md      (full documentation)"
echo "   - SETUP.md       (detailed setup guide)"
echo "   - inference.py   (baseline agent code)"
echo ""
echo "🚀 Happy hacking!"
echo ""
