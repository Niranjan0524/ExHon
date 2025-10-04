#!/usr/bin/env python3
"""
ExHon Quick Start Launcher
Simplified startup script for the proctored exam platform
"""

import os
import sys
import time
import subprocess
import platform
import webbrowser
from pathlib import Path

def print_banner():
    """Display ExHon banner"""
    banner = """
    ███████╗██╗  ██╗██╗  ██╗ ██████╗ ███╗   ██╗
    ██╔════╝╚██╗██╔╝██║  ██║██╔═══██╗████╗  ██║
    █████╗   ╚███╔╝ ███████║██║   ██║██╔██╗ ██║
    ██╔══╝   ██╔██╗ ██╔══██║██║   ██║██║╚██╗██║
    ███████╗██╔╝ ██╗██║  ██║╚██████╔╝██║ ╚████║
    ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

    Proctored Exam Platform - Professional Monitoring Solution
    """
    print(banner)
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    required_modules = [
        'flask', 'flask_socketio', 'psutil',
        'requests', 'websocket'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module.replace('_', '-'))
        except ImportError:
            missing.append(module)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    else:
        print("✅ All dependencies available")

    return True

def setup_browser_debugging():
    """Setup browser debugging if not already configured"""
    print("Setting up browser debugging...")

    scripts_dir = Path("scripts")
    system = platform.system()

    if not scripts_dir.exists():
        print("Running initial setup...")
        try:
            subprocess.run([sys.executable, "setup.py"], check=True)
            print("✅ Setup completed")
        except subprocess.CalledProcessError:
            print("⚠️  Setup completed with warnings")

    # Start Chrome debugging
    if system == "Windows":
        debug_script = scripts_dir / "start_chrome_debug.bat"
    else:
        debug_script = scripts_dir / "start_chrome_debug.sh"

    if debug_script.exists():
        print("Starting Chrome with debugging enabled...")
        try:
            if system == "Windows":
                subprocess.Popen(str(debug_script), shell=True)
            else:
                subprocess.Popen(["bash", str(debug_script)])
            print("✅ Chrome debugging started")
        except Exception as e:
            print(f"⚠️  Could not start Chrome debugging: {e}")
    else:
        print("⚠️  Chrome debug script not found")

def start_flask_app():
    """Start the Flask application"""
    print("Starting ExHon monitoring server...")

    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'production'

    try:
        # Import and run the app
        from app import app, socketio

        print("✅ Server starting on http://localhost:5000")
        print("\nDashboard Features:")
        print("• Real-time process monitoring")
        print("• AI application detection")
        print("• Browser tab management")
        print("• One-click termination controls")
        print("• Activity logging")
        print("\n" + "=" * 60)

        # Open browser automatically
        def open_browser():
            time.sleep(2)  # Wait for server to start
            webbrowser.open('http://localhost:5000')

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

        # Start the server
        socketio.run(app, host='127.0.0.1', port=5000, debug=False)

    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True

def main():
    """Main launcher function"""
    print_banner()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    print()

    try:
        # Step 1: Check dependencies
        if not check_dependencies():
            print("\n❌ Dependency check failed. Please install requirements manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)

        print()

        # Step 2: Setup browser debugging
        setup_browser_debugging()

        print()

        # Step 3: Start the application
        print("🚀 Launching ExHon Proctoring Platform...")
        print()

        # Give user a moment to read the info
        time.sleep(1)

        # Start Flask app
        start_flask_app()

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down ExHon...")
        print("Thank you for using ExHon Proctoring Platform!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nFor support, please check:")
        print("• README.md for troubleshooting")
        print("• setup.log for detailed logs")
        print("• GitHub issues for known problems")
        sys.exit(1)

if __name__ == "__main__":
    main()
