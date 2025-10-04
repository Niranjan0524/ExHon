#!/bin/bash

# ExHon Proctored Exam Platform - Easy Launcher for Linux/macOS
# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Print banner
echo -e "${GREEN}"
echo "    ███████╗██╗  ██╗██╗  ██╗ ██████╗ ███╗   ██╗"
echo "    ██╔════╝╚██╗██╔╝██║  ██║██╔═══██╗████╗  ██║"
echo "    █████╗   ╚███╔╝ ███████║██║   ██║██╔██╗ ██║"
echo "    ██╔══╝   ██╔██╗ ██╔══██║██║   ██║██║╚██╗██║"
echo "    ███████╗██╔╝ ██╗██║  ██║╚██████╔╝██║ ╚████║"
echo "    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝"
echo -e "${NC}"
echo -e "${BLUE}    Proctored Exam Platform - Easy Launcher${NC}"
echo "    ========================================"
echo ""

# Change to the script directory
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

print_info "Working directory: $SCRIPT_DIR"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_info "Detected OS: $MACHINE"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        echo "Please install Python 3.8+ from https://python.org"
        if [[ "$MACHINE" == "Mac" ]]; then
            echo "Or install via Homebrew: brew install python3"
        elif [[ "$MACHINE" == "Linux" ]]; then
            echo "Or install via package manager:"
            echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
            echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        fi
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

print_success "Python found: $($PYTHON_CMD --version)"

# Check if required files exist
if [[ ! -f "app.py" ]]; then
    print_error "app.py not found. Make sure you're in the ExHon directory."
    exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
    print_error "requirements.txt not found."
    exit 1
fi

print_info "Checking dependencies..."

# Check/install dependencies
$PYTHON_CMD -c "import flask, flask_socketio, psutil, requests" 2>/dev/null
if [[ $? -ne 0 ]]; then
    print_info "Installing missing dependencies..."
    $PYTHON_CMD -m pip install --user -r requirements.txt
    if [[ $? -ne 0 ]]; then
        print_error "Failed to install dependencies"
        echo "Try running manually: $PYTHON_CMD -m pip install -r requirements.txt"
        exit 1
    fi
else
    print_success "All dependencies are installed"
fi

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Create Chrome debugging script if it doesn't exist
CHROME_SCRIPT="scripts/start_chrome_debug.sh"
if [[ ! -f "$CHROME_SCRIPT" ]]; then
    print_info "Creating Chrome debugging script..."
    cat > "$CHROME_SCRIPT" << 'EOF'
#!/bin/bash

echo "Starting Chrome with debugging enabled..."
echo "Chrome will open with monitoring capabilities."
echo "You can use this browser normally for your exam."
echo ""

# Detect OS for Chrome path
OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # macOS
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if [[ -f "$CHROME_PATH" ]]; then
        echo "Starting Chrome on macOS..."
        "$CHROME_PATH" --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows &
    else
        echo "[ERROR] Chrome not found at $CHROME_PATH"
        echo "Please install Google Chrome from https://www.google.com/chrome/"
        exit 1
    fi
elif [[ "$OS" == "Linux" ]]; then
    # Linux - try common locations
    CHROME_PATHS=(
        "/usr/bin/google-chrome"
        "/usr/bin/google-chrome-stable"
        "/usr/bin/chromium-browser"
        "/snap/bin/chromium"
        "/usr/bin/chromium"
    )

    CHROME_FOUND=false
    for CHROME_PATH in "${CHROME_PATHS[@]}"; do
        if [[ -x "$CHROME_PATH" ]]; then
            echo "Starting Chrome on Linux: $CHROME_PATH"
            "$CHROME_PATH" --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows &
            CHROME_FOUND=true
            break
        fi
    done

    if [[ "$CHROME_FOUND" == false ]]; then
        echo "[ERROR] Chrome/Chromium not found"
        echo "Please install Google Chrome or Chromium:"
        echo "  Ubuntu/Debian: sudo apt-get install google-chrome-stable"
        echo "  Or download from: https://www.google.com/chrome/"
        exit 1
    fi
else
    echo "[ERROR] Unsupported OS: $OS"
    exit 1
fi

echo "Chrome started with debugging on port 9222"
EOF

    chmod +x "$CHROME_SCRIPT"
fi

# Test system functionality
print_info "Running system test..."
if $PYTHON_CMD test_system.py >/dev/null 2>&1; then
    print_success "System test passed"
else
    print_warning "System test reported issues. Continuing anyway..."
fi

echo ""
print_info "Starting Chrome with debugging support..."
print_info "A new Chrome window will open - use this for your exam"
echo ""

# Start Chrome in background
if [[ -x "$CHROME_SCRIPT" ]]; then
    bash "$CHROME_SCRIPT" &
    CHROME_PID=$!
    print_success "Chrome debugging started (PID: $CHROME_PID)"
else
    print_warning "Chrome debug script not executable"
fi

# Wait a moment for Chrome to start
sleep 3

print_info "Starting ExHon monitoring server..."
print_info "Dashboard will be available at: http://localhost:5000"
echo ""
echo "======================================="
echo " IMPORTANT INSTRUCTIONS:"
echo "======================================="
echo " 1. Use the Chrome window that just opened for your exam"
echo " 2. The dashboard will open automatically in a few seconds"
echo " 3. Click 'Close All' to terminate AI apps and other tabs"
echo " 4. Your exam tab will be protected from closure"
echo " 5. Press Ctrl+C to stop the server when exam is complete"
echo ""

print_info "Server starting in 2 seconds..."
sleep 2

# Open the dashboard
print_info "Opening ExHon dashboard..."
sleep 2

# Try to open browser (different commands for different systems)
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:5000" &
elif command -v open &> /dev/null; then
    open "http://localhost:5000" &
else
    print_info "Please open http://localhost:5000 in your browser manually"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down ExHon..."

    # Kill Chrome debug process if we started it
    if [[ ! -z "$CHROME_PID" ]]; then
        kill $CHROME_PID 2>/dev/null
    fi

    # Kill any remaining processes
    pkill -f "chrome.*remote-debugging-port=9222" 2>/dev/null

    print_success "Thank you for using ExHon Proctored Exam Platform!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the Python application
print_success "Launching ExHon dashboard..."
echo ""

if [[ -f "start.py" ]]; then
    $PYTHON_CMD start.py
else
    $PYTHON_CMD app.py
fi

# Cleanup on normal exit
cleanup