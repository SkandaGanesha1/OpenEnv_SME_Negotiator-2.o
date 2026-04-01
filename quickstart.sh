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

# Check for OpenAI API key
echo ""
echo "🔑 Checking OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠️  OPENAI_API_KEY not set"
    echo ""
    echo "   To use baseline inference, set:"
    echo "   export OPENAI_API_KEY='sk-...'"
    echo ""
    read -p "   Enter your OpenAI API key (or press Enter to skip): " API_KEY
    
    if [ ! -z "$API_KEY" ]; then
        export OPENAI_API_KEY="$API_KEY"
        
        # Save to .env file
        if [ ! -f .env ]; then
            cat > .env << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
API_BASE_URL=http://localhost:8000
MODEL_NAME=gpt-4o
EOF
            echo "   ✓ Saved to .env file"
        fi
    fi
else
    echo "   ✓ OPENAI_API_KEY is set"
fi

# Run diagnostics
echo ""
echo "🔍 Running diagnostics..."
python3 run_diagnostics.py || {
    echo "⚠️  Some diagnostics failed (this may be normal if server isn't running)"
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
