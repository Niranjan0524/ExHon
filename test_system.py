#!/usr/bin/env python3
"""
ExHon System Test Script
Verifies that all components are working correctly before deployment
"""

import sys
import os
import time
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print formatted test section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")
    version = sys.version_info

    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.8+")
        return False

def test_imports():
    """Test critical module imports"""
    print("Testing module imports...")

    modules_to_test = [
        ('psutil', 'System process monitoring'),
        ('flask', 'Web framework'),
        ('flask_socketio', 'Real-time communication'),
        ('requests', 'HTTP client'),
        ('platform', 'System information'),
        ('json', 'JSON handling'),
        ('threading', 'Threading support'),
        ('logging', 'Logging system')
    ]

    failed_imports = []

    for module_name, description in modules_to_test:
        try:
            if module_name == 'flask_socketio':
                import flask_socketio
            else:
                __import__(module_name)
            print(f"✅ {module_name:15} - {description}")
        except ImportError as e:
            print(f"❌ {module_name:15} - Failed: {e}")
            failed_imports.append(module_name)

    if failed_imports:
        print(f"\n⚠️  Missing modules: {', '.join(failed_imports)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True

def test_custom_modules():
    """Test our custom module imports"""
    print("Testing custom modules...")

    try:
        from monitor import SystemMonitor, get_monitor_instance
        print("✅ monitor.py - System monitoring module")
    except ImportError as e:
        print(f"❌ monitor.py - Failed: {e}")
        return False

    try:
        from killer import ProcessKiller, get_killer_instance
        print("✅ killer.py - Process termination module")
    except ImportError as e:
        print(f"❌ killer.py - Failed: {e}")
        return False

    return True

def test_system_monitoring():
    """Test system monitoring functionality"""
    print("Testing system monitoring...")

    try:
        from monitor import get_monitor_instance
        monitor = get_monitor_instance()

        # Test process enumeration
        processes = monitor.get_running_processes()
        if len(processes) > 0:
            print(f"✅ Process enumeration - {len(processes)} processes detected")
        else:
            print("⚠️  Process enumeration - No processes detected")

        # Test AI process detection
        ai_processes = monitor.get_ai_processes()
        print(f"✅ AI process detection - {len(ai_processes)} AI processes found")

        # Test system info
        system_info = monitor.get_system_info()
        if system_info:
            print("✅ System information - Retrieved successfully")
        else:
            print("⚠️  System information - Failed to retrieve")

        # Test browser tab detection
        tabs = monitor.get_browser_tabs()
        total_tabs = sum(len(browser_tabs) for browser_tabs in tabs.values())
        print(f"✅ Browser tab detection - {total_tabs} tabs found across {len(tabs)} browsers")

        return True

    except Exception as e:
        print(f"❌ System monitoring test failed: {e}")
        return False

def test_process_killer():
    """Test process killer functionality (dry run)"""
    print("Testing process killer...")

    try:
        from killer import get_killer_instance
        killer = get_killer_instance()

        # Test preview functionality (safe)
        preview = killer.get_termination_preview()

        ai_count = len(preview.get('ai_applications', []))
        print(f"✅ Termination preview - {ai_count} AI applications would be terminated")

        # Count browser tabs that would be affected
        tab_counts = {}
        for browser, tabs in preview.get('browser_tabs', {}).items():
            tab_counts[browser] = len(tabs)

        total_tabs = sum(tab_counts.values())
        print(f"✅ Browser tab preview - {total_tabs} tabs detected across browsers")

        return True

    except Exception as e:
        print(f"❌ Process killer test failed: {e}")
        return False

def test_flask_app():
    """Test Flask application startup"""
    print("Testing Flask application...")

    try:
        # Try importing the app
        from app import app
        print("✅ Flask app import - Success")

        # Test if app is configured
        if app.config.get('SECRET_KEY'):
            print("✅ Flask configuration - App configured")
        else:
            print("⚠️  Flask configuration - Default secret key")

        return True

    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_file_structure():
    """Test required file structure"""
    print("Testing file structure...")

    required_files = [
        'app.py',
        'monitor.py',
        'killer.py',
        'requirements.txt',
        'templates/dashboard.html',
        'static/css/dashboard.css',
        'static/js/dashboard.js'
    ]

    missing_files = []

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False

    return True

def test_permissions():
    """Test system permissions"""
    print("Testing system permissions...")

    try:
        import psutil

        # Test process access
        processes = list(psutil.process_iter(['pid', 'name']))
        print(f"✅ Process access - Can enumerate {len(processes)} processes")

        # Test system stats access
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"✅ System stats - CPU: {cpu_percent}%, Memory: {memory.percent}%")

        return True

    except Exception as e:
        print(f"❌ Permission test failed: {e}")
        print("   Try running as administrator/root if needed")
        return False

def test_network_access():
    """Test network access for browser debugging"""
    print("Testing network access...")

    try:
        import requests

        # Test if debugging ports are accessible (won't work unless browsers are running with debug)
        test_ports = [9222, 9229, 9225]  # Chrome, Firefox, Edge
        accessible_ports = []

        for port in test_ports:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=1)
                if response.status_code == 200:
                    accessible_ports.append(port)
            except:
                pass

        if accessible_ports:
            print(f"✅ Browser debugging - Ports accessible: {accessible_ports}")
        else:
            print("⚠️  Browser debugging - No debug ports accessible")
            print("   This is normal if browsers aren't running with debug flags")

        return True

    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def generate_test_report(results):
    """Generate final test report"""
    print_header("TEST REPORT")

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    print("\nTest Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")

    if failed_tests == 0:
        print(f"\n🎉 All tests passed! ExHon is ready to run.")
        print("\nTo start the application:")
        print("  python start.py")
        print("  or")
        print("  python app.py")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please resolve issues before running.")

    return failed_tests == 0

def main():
    """Run all tests"""
    print("ExHon System Test Suite")
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")

    # Run all tests
    test_results = {}

    test_results['Python Version'] = test_python_version()
    print_header("MODULE IMPORTS")
    test_results['Standard Imports'] = test_imports()
    test_results['Custom Modules'] = test_custom_modules()

    print_header("FILE STRUCTURE")
    test_results['File Structure'] = test_file_structure()

    print_header("FUNCTIONALITY TESTS")
    test_results['System Monitoring'] = test_system_monitoring()
    test_results['Process Killer'] = test_process_killer()
    test_results['Flask App'] = test_flask_app()

    print_header("SYSTEM ACCESS")
    test_results['Permissions'] = test_permissions()
    test_results['Network Access'] = test_network_access()

    # Generate report
    success = generate_test_report(test_results)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
