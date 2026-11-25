#!/usr/bin/env bash
# ==============================================================================
# OPV-PredictNet — Local Demonstration Launcher (Bash)
# Starts FastAPI backend (port 8000) and Vite React frontend (port 5173).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "     OPV-PredictNet — Academic Demonstration Launcher    "
echo "=========================================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# 1. Determine Python executable
if [ -f "$PROJECT_ROOT/backend/.venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_ROOT/backend/.venv/bin/python"
elif [ -f "$PROJECT_ROOT/backend/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$PROJECT_ROOT/backend/.venv/Scripts/python.exe"
else
    PYTHON_EXE="python3"
fi

# 2. Check model artifacts
MODELS_DIR="$PROJECT_ROOT/ml/models/saved_models/phase9_v1"
if [ ! -f "$MODELS_DIR/phase9_M3_model.joblib" ]; then
    echo "[INFO] Packaging model artifacts..."
    $PYTHON_EXE "$PROJECT_ROOT/ml/scripts/package_models_phase9.py"
fi

# 3. Check frontend dependencies
FRONTEND_DIR="$PROJECT_ROOT/frontend"
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    cd "$FRONTEND_DIR" && npm install && cd "$PROJECT_ROOT"
fi

# 4. Cleanup on exit
cleanup() {
    echo ""
    echo "[INFO] Shutting down demonstration processes..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    echo "[OK] Services stopped cleanly."
}
trap cleanup EXIT INT TERM

# 5. Start Backend
echo "[INFO] Starting Backend FastAPI Service (http://127.0.0.1:8000)..."
cd "$PROJECT_ROOT"
$PYTHON_EXE -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# 6. Start Frontend
echo "[INFO] Starting Frontend Vite Server (http://localhost:5173)..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "----------------------------------------------------------"
echo " Demonstration Services Running:"
echo "   Frontend Application: http://localhost:5173"
echo "   Backend API Docs:     http://127.0.0.1:8000/docs"
echo "   Backend Health Check: http://127.0.0.1:8000/api/health"
echo "----------------------------------------------------------"
echo "Press Ctrl+C to terminate demonstration servers."
echo ""

wait
