#!/usr/bin/env python3
"""
ExHon Setup Script
Prepares the system for proctored exam monitoring by enabling browser debugging protocols
and checking system compatibility.
"""

import os
import sys
import platform
import subprocess
import json
import psutil
import time
from pathlib import Path
import argparse

class ExHonSetup:
    """Setup and configuration handler for ExHon proctoring system"""

    def __init__(self):
        self.system = platform.system()
        self.setup_log = []

    def log(self, message, level="INFO"):
        """Log setup messages"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.setup_log.append(log_entry)

    def check_python_version(self):
        """Check if Python version is compatible"""
        self.log("Checking Python version...")
        version = sys.version_info

        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.log("Python 3.8+ is required. Current version: " +
                    f"{version.major}.{version.minor}.{version.micro}", "ERROR")
            return False

        self.log(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

    def install_dependencies(self):
        """Install required Python packages"""
        self.log("Installing Python dependencies...")

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            self.log("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install dependencies: {e}", "ERROR")
            return False

    def setup_chrome_debugging(self):
        """Setup Chrome for remote debugging"""
        self.log("Setting up Chrome debugging...")

        chrome_paths = self._get_chrome_paths()

        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                self.log(f"Found Chrome at: {chrome_path}")

                # Create startup script
                self._create_chrome_debug_script(chrome_path)
                self.log("Chrome debugging setup completed")
                return True

        self.log("Chrome not found. Please install Google Chrome.", "WARNING")
        return False

    def setup_firefox_debugging(self):
        """Setup Firefox for remote debugging"""
        self.log("Setting up Firefox debugging...")

        firefox_paths = self._get_firefox_paths()

        for firefox_path in firefox_paths:
            if os.path.exists(firefox_path):
                self.log(f"Found Firefox at: {firefox_path}")

                # Create Firefox profile for debugging
                self._create_firefox_debug_profile()
                self.log("Firefox debugging setup completed")
                return True

        self.log("Firefox not found. Install optional.", "WARNING")
        return False

    def setup_edge_debugging(self):
        """Setup Microsoft Edge for remote debugging"""
        if self.system != "Windows":
            self.log("Edge debugging only available on Windows", "INFO")
            return False

        self.log("Setting up Edge debugging...")

        edge_paths = self._get_edge_paths()

        for edge_path in edge_paths:
            if os.path.exists(edge_path):
                self.log(f"Found Edge at: {edge_path}")

                # Create startup script
                self._create_edge_debug_script(edge_path)
                self.log("Edge debugging setup completed")
                return True

        self.log("Microsoft Edge not found. Install optional.", "WARNING")
        return False

    def _get_chrome_paths(self):
        """Get possible Chrome installation paths"""
        if self.system == "Windows":
            return [
                os.path.expanduser("~/AppData/Local/Google/Chrome/Application/chrome.exe"),
                "C:/Program Files/Google/Chrome/Application/chrome.exe",
                "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
            ]
        elif self.system == "Darwin":  # macOS
            return [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            ]
        else:  # Linux
            return [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/chromium-browser",
                "/snap/bin/chromium"
            ]

    def _get_firefox_paths(self):
        """Get possible Firefox installation paths"""
        if self.system == "Windows":
            return [
                "C:/Program Files/Mozilla Firefox/firefox.exe",
                "C:/Program Files (x86)/Mozilla Firefox/firefox.exe"
            ]
        elif self.system == "Darwin":  # macOS
            return [
                "/Applications/Firefox.app/Contents/MacOS/firefox"
            ]
        else:  # Linux
            return [
                "/usr/bin/firefox",
                "/usr/bin/firefox-esr",
                "/snap/bin/firefox"
            ]

    def _get_edge_paths(self):
        """Get possible Edge installation paths"""
        if self.system == "Windows":
            return [
                os.path.expanduser("~/AppData/Local/Microsoft/Edge/Application/msedge.exe"),
                "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
            ]
        elif self.system == "Darwin":  # macOS
            return [
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
            ]
        else:  # Linux
            return [
                "/usr/bin/microsoft-edge",
                "/usr/bin/microsoft-edge-stable"
            ]

    def _create_chrome_debug_script(self, chrome_path):
        """Create Chrome debugging startup script"""
        script_dir = Path("scripts")
        script_dir.mkdir(exist_ok=True)

        if self.system == "Windows":
            script_path = script_dir / "start_chrome_debug.bat"
            script_content = f'''@echo off
echo Starting Chrome with debugging enabled...
"{chrome_path}" --remote-debugging-port=9222 --user-data-dir="%TEMP%\\chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows
'''
        else:
            script_path = script_dir / "start_chrome_debug.sh"
            script_content = f'''#!/bin/bash
echo "Starting Chrome with debugging enabled..."
"{chrome_path}" --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug" --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows &
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        if self.system != "Windows":
            os.chmod(script_path, 0o755)

        self.log(f"Chrome debug script created: {script_path}")

    def _create_firefox_debug_profile(self):
        """Create Firefox debugging profile"""
        script_dir = Path("scripts")
        script_dir.mkdir(exist_ok=True)

        profile_dir = script_dir / "firefox_debug_profile"
        profile_dir.mkdir(exist_ok=True)

        # Create user.js with debugging settings
        user_js = profile_dir / "user.js"
        with open(user_js, 'w') as f:
            f.write('''
// Firefox debugging configuration
user_pref("devtools.debugger.remote-enabled", true);
user_pref("devtools.chrome.enabled", true);
user_pref("devtools.debugger.remote-port", 9229);
user_pref("devtools.debugger.prompt-connection", false);
user_pref("devtools.debugger.force-local", false);
''')

        if self.system == "Windows":
            script_path = script_dir / "start_firefox_debug.bat"
            script_content = '''@echo off
echo Starting Firefox with debugging enabled...
firefox -profile "scripts/firefox_debug_profile" --start-debugger-server=9229
'''
        else:
            script_path = script_dir / "start_firefox_debug.sh"
            script_content = '''#!/bin/bash
echo "Starting Firefox with debugging enabled..."
firefox -profile "scripts/firefox_debug_profile" --start-debugger-server=9229 &
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        if self.system != "Windows":
            os.chmod(script_path, 0o755)

        self.log(f"Firefox debug script created: {script_path}")

    def _create_edge_debug_script(self, edge_path):
        """Create Edge debugging startup script"""
        script_dir = Path("scripts")
        script_dir.mkdir(exist_ok=True)

        if self.system == "Windows":
            script_path = script_dir / "start_edge_debug.bat"
            script_content = f'''@echo off
echo Starting Edge with debugging enabled...
"{edge_path}" --remote-debugging-port=9225 --user-data-dir="%TEMP%\\edge_debug" --disable-background-timer-throttling
'''
        else:
            script_path = script_dir / "start_edge_debug.sh"
            script_content = f'''#!/bin/bash
echo "Starting Edge with debugging enabled..."
"{edge_path}" --remote-debugging-port=9225 --user-data-dir="/tmp/edge_debug" --disable-background-timer-throttling &
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        if self.system != "Windows":
            os.chmod(script_path, 0o755)

        self.log(f"Edge debug script created: {script_path}")

    def create_config_file(self):
        """Create configuration file"""
        self.log("Creating configuration file...")

        config = {
            "system_info": {
                "platform": self.system,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "setup_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "browser_debugging": {
                "chrome_port": 9222,
                "firefox_port": 9229,
                "edge_port": 9225
            },
            "monitoring": {
                "update_interval": 5,
                "max_log_entries": 50,
                "auto_start_monitoring": True
            },
            "security": {
                "preserve_exam_tab": True,
                "require_confirmation": True,
                "log_all_actions": True
            }
        }

        with open("config.json", 'w') as f:
            json.dump(config, f, indent=4)

        self.log("Configuration file created: config.json")

    def test_system_access(self):
        """Test system access for monitoring"""
        self.log("Testing system access...")

        try:
            # Test process enumeration
            processes = list(psutil.process_iter(['pid', 'name']))
            self.log(f"Process access: OK ({len(processes)} processes detected)")

            # Test system stats access
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.log(f"System stats access: OK (CPU: {cpu_percent}%, Memory: {memory.percent}%)")

            return True

        except Exception as e:
            self.log(f"System access test failed: {e}", "ERROR")
            return False

    def create_desktop_shortcut(self):
        """Create desktop shortcut for easy access"""
        self.log("Creating desktop shortcut...")

        if self.system == "Windows":
            # Create .bat file for Windows
            shortcut_content = f'''@echo off
cd /d "{os.getcwd()}"
python app.py
pause
'''
            shortcut_path = Path.home() / "Desktop" / "ExHon_Proctoring.bat"

        elif self.system == "Darwin":  # macOS
            # Create .command file for macOS
            shortcut_content = f'''#!/bin/bash
cd "{os.getcwd()}"
python3 app.py
'''
            shortcut_path = Path.home() / "Desktop" / "ExHon_Proctoring.command"

        else:  # Linux
            # Create .desktop file for Linux
            shortcut_content = f'''[Desktop Entry]
Version=1.0
Type=Application
Name=ExHon Proctoring
Comment=ExHon Proctored Exam Platform
Exec=python3 {os.getcwd()}/app.py
Icon=applications-system
Terminal=true
Categories=Education;
'''
            shortcut_path = Path.home() / "Desktop" / "ExHon_Proctoring.desktop"

        try:
            with open(shortcut_path, 'w') as f:
                f.write(shortcut_content)

            if self.system != "Windows":
                os.chmod(shortcut_path, 0o755)

            self.log(f"Desktop shortcut created: {shortcut_path}")
            return True

        except Exception as e:
            self.log(f"Failed to create desktop shortcut: {e}", "WARNING")
            return False

    def run_setup(self, install_deps=True, create_shortcuts=True):
        """Run complete setup process"""
        self.log("Starting ExHon setup process...")
        self.log(f"System: {self.system}")

        success = True

        # Check Python version
        if not self.check_python_version():
            return False

        # Install dependencies
        if install_deps and not self.install_dependencies():
            success = False

        # Setup browser debugging
        self.setup_chrome_debugging()
        self.setup_firefox_debugging()
        self.setup_edge_debugging()

        # Create configuration
        self.create_config_file()

        # Test system access
        if not self.test_system_access():
            success = False

        # Create shortcuts
        if create_shortcuts:
            self.create_desktop_shortcut()

        # Save setup log
        log_path = Path("setup.log")
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.setup_log))
        self.log(f"Setup log saved to: {log_path}")

        if success:
            self.log("Setup completed successfully!")
            self.log("To start the system:")
            self.log("1. Run browser debugging scripts in 'scripts/' folder")
            self.log("2. Execute: python app.py")
            self.log("3. Open http://localhost:5000 in your browser")
        else:
            self.log("Setup completed with warnings. Check the log for details.", "WARNING")

        return success

def main():
    parser = argparse.ArgumentParser(description="ExHon Setup Script")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--no-shortcuts", action="store_true", help="Skip desktop shortcut creation")

    args = parser.parse_args()

    setup = ExHonSetup()
    setup.run_setup(
        install_deps=not args.no_deps,
        create_shortcuts=not args.no_shortcuts
    )

if __name__ == "__main__":
    main()
