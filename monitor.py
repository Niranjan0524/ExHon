"""
System Monitoring Module for ExHon Proctored Exam Platform
Handles detection and monitoring of running processes and browser tabs
"""

import psutil

import platform

import requests
import time
from typing import List, Dict
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Main class for monitoring system processes and browser tabs"""

    def __init__(self):
        self.system = platform.system()
        self.browser_ports = {
            'chrome': [9222, 9223, 9224],
            'edge': [9225, 9226, 9227],
            'firefox': [9228, 9229, 9230]
        }

    def get_running_processes(self) -> List[Dict]:
        """Get list of all running processes with details"""
        processes = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:
                        processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'exe': proc_info['exe'] or 'N/A',
                            'memory_percent': round(proc_info['memory_percent'], 2),
                            'cpu_percent': round(proc_info['cpu_percent'], 2),
                            'status': proc.status(),
                            'is_ai_app': self._is_ai_application(proc_info['name'])
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error getting processes: {e}")

        return sorted(processes, key=lambda x: x['memory_percent'], reverse=True)

    def _is_ai_application(self, process_name: str) -> bool:
        """Check if a process is an AI application"""
        ai_keywords = [
            'chatgpt', 'claude', 'gemini', 'copilot', 'github copilot',
            'openai', 'anthropic', 'bard', 'bing chat', 'perplexity',
            'tabnine', 'codewhisperer', 'codex', 'gpt', 'llm'
        ]

        name_lower = process_name.lower()
        return any(keyword in name_lower for keyword in ai_keywords)

    def get_browser_tabs(self) -> Dict[str, List[Dict]]:
        """Get all open browser tabs across different browsers"""
        all_tabs = {}

        # Chrome and Chromium-based browsers
        chrome_tabs = self._get_chrome_tabs()
        if chrome_tabs:
            all_tabs['chrome'] = chrome_tabs

        # Firefox tabs
        firefox_tabs = self._get_firefox_tabs()
        if firefox_tabs:
            all_tabs['firefox'] = firefox_tabs

        # Edge tabs
        edge_tabs = self._get_edge_tabs()
        if edge_tabs:
            all_tabs['edge'] = edge_tabs

        return all_tabs

    def _get_chrome_tabs(self) -> List[Dict]:
        """Get Chrome browser tabs using Chrome DevTools Protocol"""
        tabs = []

        for port in self.browser_ports['chrome']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tab_data = response.json()
                    for tab in tab_data:
                        if tab.get('type') == 'page':
                            tabs.append({
                                'id': tab.get('id'),
                                'title': tab.get('title', 'Untitled'),
                                'url': tab.get('url', ''),
                                'active': tab.get('active', False),
                                'browser': 'chrome',
                                'port': port,
                                'webSocketDebuggerUrl': tab.get('webSocketDebuggerUrl')
                            })
                    break  # Found active Chrome instance
            except Exception:
                continue

        return tabs

    def _get_firefox_tabs(self) -> List[Dict]:
        """Get Firefox browser tabs using Remote Debugging Protocol"""
        tabs = []

        for port in self.browser_ports['firefox']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tab_data = response.json()
                    for tab in tab_data:
                        tabs.append({
                            'id': tab.get('actor'),
                            'title': tab.get('title', 'Untitled'),
                            'url': tab.get('url', ''),
                            'active': False,  # Firefox API doesn't easily expose active tab
                            'browser': 'firefox',
                            'port': port
                        })
                    break
            except Exception:
                continue

        return tabs

    def _get_edge_tabs(self) -> List[Dict]:
        """Get Microsoft Edge tabs (Chromium-based)"""
        tabs = []

        for port in self.browser_ports['edge']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tab_data = response.json()
                    for tab in tab_data:
                        if tab.get('type') == 'page':
                            tabs.append({
                                'id': tab.get('id'),
                                'title': tab.get('title', 'Untitled'),
                                'url': tab.get('url', ''),
                                'active': tab.get('active', False),
                                'browser': 'edge',
                                'port': port,
                                'webSocketDebuggerUrl': tab.get('webSocketDebuggerUrl')
                            })
                    break
            except Exception:
                continue

        return tabs

    def get_system_info(self) -> Dict:
        """Get general system information"""
        try:
            cpu_info = {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }

            memory = psutil.virtual_memory()
            memory_info = {
                'total': self._bytes_to_gb(memory.total),
                'available': self._bytes_to_gb(memory.available),
                'percent': memory.percent,
                'used': self._bytes_to_gb(memory.used)
            }

            disk = psutil.disk_usage('/')
            disk_info = {
                'total': self._bytes_to_gb(disk.total),
                'used': self._bytes_to_gb(disk.used),
                'free': self._bytes_to_gb(disk.free),
                'percent': (disk.used / disk.total) * 100
            }

            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'boot_time': psutil.boot_time()
            }

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

    def _bytes_to_gb(self, bytes_value: int) -> float:
        """Convert bytes to gigabytes"""
        return round(bytes_value / (1024**3), 2)

    def is_exam_tab(self, tab: Dict, current_url: str = None) -> bool:
        """Check if a tab is the current exam tab"""
        if not current_url:
            return False

        tab_url = tab.get('url', '')
        if not tab_url:
            return False

        current_domain = urlparse(current_url).netloc
        tab_domain = urlparse(tab_url).netloc

        # Consider it an exam tab if it's on the same domain
        return current_domain == tab_domain

    def get_ai_processes(self) -> List[Dict]:
        """Get only AI-related processes"""
        all_processes = self.get_running_processes()
        return [proc for proc in all_processes if proc['is_ai_app']]

    def get_tab_summary(self) -> Dict:
        """Get summary of all browser tabs"""
        tabs = self.get_browser_tabs()
        summary = {
            'total_tabs': 0,
            'browsers': {},
            'tabs_by_browser': tabs
        }

        for browser, browser_tabs in tabs.items():
            summary['browsers'][browser] = len(browser_tabs)
            summary['total_tabs'] += len(browser_tabs)

        return summary

    def start_browser_debugging(self) -> bool:
        """Attempt to start browser debugging protocols"""
        success = False

        try:
            # For Chrome - check if already running with debugging
            chrome_processes = [p for p in psutil.process_iter(['name', 'cmdline'])
                             if 'chrome' in p.info['name'].lower()]

            if chrome_processes:
                # Chrome is running, check if debugging is enabled
                for proc in chrome_processes:
                    cmdline = proc.info.get('cmdline', [])
                    if any('remote-debugging-port' in arg for arg in cmdline):
                        success = True
                        break

        except Exception as e:
            logger.warning(f"Could not start browser debugging: {e}")

        return success

# Utility functions for external use
_monitor_instance = None

def get_monitor_instance() -> SystemMonitor:
    """Get a singleton instance of SystemMonitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance

def quick_process_scan() -> Dict:
    """Quick scan for processes and basic info"""
    monitor = get_monitor_instance()
    return {
        'processes': len(monitor.get_running_processes()),
        'ai_processes': len(monitor.get_ai_processes()),
        'tabs': monitor.get_tab_summary(),
        'timestamp': time.time()
    }
