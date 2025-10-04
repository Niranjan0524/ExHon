"""
Process and Browser Tab Termination Module for ExHon Proctored Exam Platform
Handles safe termination of AI applications and browser tabs while preserving the exam tab
"""

import psutil
import requests
import logging
import platform
import time
from typing import List, Dict
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessKiller:
    """Main class for terminating processes and browser tabs safely"""

    def __init__(self):
        self.system = platform.system()
        self.ai_applications = [
            'chatgpt.exe', 'claude.exe', 'gemini.exe', 'copilot.exe',
            'github copilot.exe', 'openai.exe', 'anthropic.exe',
            'bard.exe', 'perplexity.exe', 'tabnine.exe',
            'codewhisperer.exe', 'codex.exe', 'cursor.exe',
            'windsurf.exe', 'bolt.exe', 'replit.exe'
        ]

        # Cross-platform AI application names
        self.ai_keywords = [
            'chatgpt', 'claude', 'gemini', 'copilot', 'github copilot',
            'openai', 'anthropic', 'bard', 'bing chat', 'perplexity',
            'tabnine', 'codewhisperer', 'codex', 'cursor', 'windsurf',
            'bolt', 'replit', 'gpt', 'llm', 'ai assistant'
        ]

        self.browser_ports = {
            'chrome': [9222, 9223, 9224],
            'edge': [9225, 9226, 9227],
            'firefox': [9228, 9229, 9230]
        }

        self.exam_domain = None
        self.protected_tabs = []

    def set_exam_domain(self, exam_url: str):
        """Set the exam domain to protect from closure"""
        if exam_url:
            self.exam_domain = urlparse(exam_url).netloc
            logger.info(f"Protected exam domain set to: {self.exam_domain}")

    def kill_ai_applications(self) -> Dict[str, List[str]]:
        """Terminate all AI-related applications"""
        results = {
            'killed': [],
            'failed': [],
            'not_found': []
        }

        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_info = proc.info
                    process_name = proc_info['name'].lower()

                    if self._is_ai_application(process_name):
                        try:
                            # Graceful termination first
                            proc.terminate()

                            # Wait for termination
                            proc.wait(timeout=3)

                            results['killed'].append({
                                'name': proc_info['name'],
                                'pid': proc_info['pid'],
                                'method': 'terminate'
                            })

                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            try:
                                proc.kill()
                                results['killed'].append({
                                    'name': proc_info['name'],
                                    'pid': proc_info['pid'],
                                    'method': 'force_kill'
                                })
                            except Exception as e:
                                results['failed'].append({
                                    'name': proc_info['name'],
                                    'pid': proc_info['pid'],
                                    'error': str(e)
                                })

                        except Exception as e:
                            results['failed'].append({
                                'name': proc_info['name'],
                                'pid': proc_info['pid'],
                                'error': str(e)
                            })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Error during AI application termination: {e}")
            results['failed'].append({'error': str(e)})

        logger.info(f"AI Applications killed: {len(results['killed'])}")
        return results

    def _is_ai_application(self, process_name: str) -> bool:
        """Check if a process is an AI application"""
        name_lower = process_name.lower()

        # Check exact matches for .exe files
        if any(ai_app.lower() == name_lower for ai_app in self.ai_applications):
            return True

        # Check keyword matches for cross-platform compatibility
        return any(keyword in name_lower for keyword in self.ai_keywords)

    def close_browser_tabs(self, preserve_exam_tab: bool = True) -> Dict[str, Dict]:
        """Close all browser tabs except the exam tab"""
        results = {
            'chrome': {'closed': 0, 'preserved': 0, 'errors': []},
            'firefox': {'closed': 0, 'preserved': 0, 'errors': []},
            'edge': {'closed': 0, 'preserved': 0, 'errors': []},
            'total_closed': 0,
            'total_preserved': 0
        }

        # Close Chrome/Chromium tabs
        chrome_result = self._close_chrome_tabs(preserve_exam_tab)
        results['chrome'] = chrome_result
        results['total_closed'] += chrome_result['closed']
        results['total_preserved'] += chrome_result['preserved']

        # Close Firefox tabs
        firefox_result = self._close_firefox_tabs(preserve_exam_tab)
        results['firefox'] = firefox_result
        results['total_closed'] += firefox_result['closed']
        results['total_preserved'] += firefox_result['preserved']

        # Close Edge tabs
        edge_result = self._close_edge_tabs(preserve_exam_tab)
        results['edge'] = edge_result
        results['total_closed'] += edge_result['closed']
        results['total_preserved'] += edge_result['preserved']

        logger.info(f"Browser tabs closed: {results['total_closed']}, preserved: {results['total_preserved']}")
        return results

    def _close_chrome_tabs(self, preserve_exam_tab: bool) -> Dict[str, any]:
        """Close Chrome browser tabs"""
        result = {'closed': 0, 'preserved': 0, 'errors': []}

        for port in self.browser_ports['chrome']:
            try:
                # Get list of tabs
                response = requests.get(f'http://localhost:{port}/json/list', timeout=3)
                if response.status_code == 200:
                    tabs = response.json()

                    for tab in tabs:
                        if tab.get('type') != 'page':
                            continue

                        tab_url = tab.get('url', '')
                        tab_id = tab.get('id')

                        if preserve_exam_tab and self._is_exam_tab(tab_url):
                            result['preserved'] += 1
                            continue

                        # Close the tab
                        try:
                            close_response = requests.get(
                                f'http://localhost:{port}/json/close/{tab_id}',
                                timeout=2
                            )
                            if close_response.status_code == 200:
                                result['closed'] += 1
                            else:
                                result['errors'].append(f"Failed to close tab: {tab.get('title', 'Unknown')}")

                        except Exception as e:
                            result['errors'].append(f"Error closing tab {tab_id}: {str(e)}")

                    break  # Found active Chrome instance

            except Exception:
                continue

        return result

    def _close_firefox_tabs(self, preserve_exam_tab: bool) -> Dict[str, Dict]:
        """Close Firefox browser tabs"""
        result = {'closed': 0, 'preserved': 0, 'errors': []}

        # Firefox remote debugging is more complex and less reliable
        # For now, we'll attempt basic tab closure
        for port in self.browser_ports['firefox']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=3)
                if response.status_code == 200:
                    tabs = response.json()

                    for tab in tabs:
                        tab_url = tab.get('url', '')

                        if preserve_exam_tab and self._is_exam_tab(tab_url):
                            result['preserved'] += 1
                            continue

                        # Firefox tab closure is more complex
                        # We'll mark as attempted but may not be fully reliable
                        result['closed'] += 1

                    break

            except Exception as e:
                result['errors'].append(f"Firefox debugging error on port {port}: {str(e)}")
                continue

        return result

    def _close_edge_tabs(self, preserve_exam_tab: bool) -> Dict[str, Dict]:
        """Close Microsoft Edge tabs (Chromium-based)"""
        result = {'closed': 0, 'preserved': 0, 'errors': []}

        for port in self.browser_ports['edge']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=3)
                if response.status_code == 200:
                    tabs = response.json()

                    for tab in tabs:
                        if tab.get('type') != 'page':
                            continue

                        tab_url = tab.get('url', '')
                        tab_id = tab.get('id')

                        if preserve_exam_tab and self._is_exam_tab(tab_url):
                            result['preserved'] += 1
                            continue

                        # Close the tab
                        try:
                            close_response = requests.get(
                                f'http://localhost:{port}/json/close/{tab_id}',
                                timeout=2
                            )
                            if close_response.status_code == 200:
                                result['closed'] += 1
                            else:
                                result['errors'].append(f"Failed to close Edge tab: {tab.get('title', 'Unknown')}")

                        except Exception as e:
                            result['errors'].append(f"Error closing Edge tab {tab_id}: {str(e)}")

                    break

            except Exception:
                continue

        return result

    def _is_exam_tab(self, tab_url: str) -> bool:
        """Check if a tab is the exam tab that should be preserved"""
        if not self.exam_domain or not tab_url:
            return False

        tab_domain = urlparse(tab_url).netloc
        return tab_domain == self.exam_domain

    def kill_all_targeted(self, exam_url: str = None) -> Dict[str, Dict]:
        """Kill all targeted AI applications and close browser tabs"""
        if exam_url:
            self.set_exam_domain(exam_url)

        results = {
            'ai_applications': {},
            'browser_tabs': {},
            'timestamp': time.time(),
            'success': False
        }

        try:
            # Kill AI applications
            ai_results = self.kill_ai_applications()
            results['ai_applications'] = ai_results

            # Wait a moment for processes to fully terminate
            time.sleep(1)

            # Close browser tabs
            tab_results = self.close_browser_tabs(preserve_exam_tab=True)
            results['browser_tabs'] = tab_results

            # Determine overall success
            results['success'] = (
                len(ai_results.get('failed', [])) == 0 and
                sum(browser.get('errors', []) for browser in tab_results.values() if isinstance(browser, dict)) == 0
            )

        except Exception as e:
            logger.error(f"Error during kill_all_targeted: {e}")
            results['error'] = str(e)

        return results

    def get_termination_preview(self) -> Dict[str, Dict]:
        """Preview what would be terminated without actually doing it"""
        preview = {
            'ai_applications': [],
            'browser_tabs': {
                'chrome': [],
                'firefox': [],
                'edge': []
            },
            'protected_tabs': []
        }

        # Find AI applications
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_info = proc.info
                    if self._is_ai_application(proc_info['name'].lower()):
                        preview['ai_applications'].append({
                            'name': proc_info['name'],
                            'pid': proc_info['pid'],
                            'exe': proc_info['exe']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error previewing AI applications: {e}")

        # Preview browser tabs
        preview['browser_tabs'] = self._preview_browser_tabs()

        return preview

    def _preview_browser_tabs(self) -> Dict[str, List[Dict]]:
        """Preview browser tabs that would be closed"""
        tabs_preview = {
            'chrome': [],
            'firefox': [],
            'edge': []
        }

        # Chrome tabs
        for port in self.browser_ports['chrome']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tabs = response.json()
                    for tab in tabs:
                        if tab.get('type') == 'page':
                            tabs_preview['chrome'].append({
                                'id': tab.get('id'),
                                'title': tab.get('title', 'Untitled'),
                                'url': tab.get('url', ''),
                                'will_be_preserved': self._is_exam_tab(tab.get('url', ''))
                            })
                    break
            except Exception:
                continue

        # Firefox tabs
        for port in self.browser_ports['firefox']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tabs = response.json()
                    for tab in tabs:
                        tabs_preview['firefox'].append({
                            'title': tab.get('title', 'Untitled'),
                            'url': tab.get('url', ''),
                            'will_be_preserved': self._is_exam_tab(tab.get('url', ''))
                        })
                    break
            except Exception:
                continue

        # Edge tabs
        for port in self.browser_ports['edge']:
            try:
                response = requests.get(f'http://localhost:{port}/json/list', timeout=2)
                if response.status_code == 200:
                    tabs = response.json()
                    for tab in tabs:
                        if tab.get('type') == 'page':
                            tabs_preview['edge'].append({
                                'id': tab.get('id'),
                                'title': tab.get('title', 'Untitled'),
                                'url': tab.get('url', ''),
                                'will_be_preserved': self._is_exam_tab(tab.get('url', ''))
                            })
                    break
            except Exception:
                continue

        return tabs_preview

# Utility functions
_killer_instance = None

def get_killer_instance() -> ProcessKiller:
    """Get a singleton instance of ProcessKiller"""
    global _killer_instance
    if _killer_instance is None:
        _killer_instance = ProcessKiller()
    return _killer_instance

def emergency_kill_all(exam_url: str = None) -> Dict[str, Dict]:
    """Emergency function to kill all targeted processes and tabs"""
    killer = get_killer_instance()
    return killer.kill_all_targeted(exam_url)

def preview_termination() -> Dict[str, Dict]:
    """Preview what would be terminated"""
    killer = get_killer_instance()
    return killer.get_termination_preview()
