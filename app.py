"""
Main Flask Application for ExHon Proctored Exam Platform
Provides monitoring dashboard with real-time updates via WebSocket
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import threading
from datetime import datetime
import logging

# Import our custom modules
from monitor import get_monitor_instance
from killer import get_killer_instance, preview_termination

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global variables
monitor = get_monitor_instance()
killer = get_killer_instance()
monitoring_active = False
update_thread = None
demo_mode_enabled = False
demo_ai_processes = []
demo_browser_tabs = {
    'chrome': [],
    'firefox': [],
    'edge': []
}
demo_summary = {'total_tabs': 0, 'browsers': {'chrome': 0, 'firefox': 0, 'edge': 0}}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/system-info')
def get_system_info():
    """Get general system information"""
    try:
        system_info = monitor.get_system_info()
        return jsonify({
            'success': True,
            'data': system_info,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/processes')
def get_processes():
    """Get all running processes"""
    try:
        if demo_mode_enabled:
            # Return demo data
            real_processes = monitor.get_running_processes()[:20]
            all_processes = real_processes + demo_ai_processes
        else:
            processes = monitor.get_running_processes()
            ai_processes = monitor.get_ai_processes()
            all_processes = processes

        return jsonify({
            'success': True,
            'data': {
                'all_processes': all_processes[:50],  # Limit to top 50 by memory
                'ai_processes': demo_ai_processes if demo_mode_enabled else monitor.get_ai_processes(),
                'total_count': len(all_processes),
                'ai_count': len(demo_ai_processes) if demo_mode_enabled else len(monitor.get_ai_processes())
            },
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/browser-tabs')
def get_browser_tabs():
    """Get all browser tabs"""
    try:
        if demo_mode_enabled:
            # Return current demo tab data
            return jsonify({
                'success': True,
                'data': {
                    'tabs': demo_browser_tabs,
                    'summary': demo_summary
                },
                'timestamp': time.time()
            })
        else:
            tabs = monitor.get_browser_tabs()
            tab_summary = monitor.get_tab_summary()

            return jsonify({
                'success': True,
                'data': {
                    'tabs': tabs,
                    'summary': tab_summary
                },
                'timestamp': time.time()
            })
    except Exception as e:
        logger.error(f"Error getting browser tabs: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/preview-termination')
def get_termination_preview():
    """Preview what would be terminated"""
    try:
        exam_url = request.args.get('exam_url', '')
        if exam_url:
            killer.set_exam_domain(exam_url)

        preview = preview_termination()

        return jsonify({
            'success': True,
            'data': preview,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error getting termination preview: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/kill-all', methods=['POST'])
def kill_all():
    """Kill all targeted AI applications and browser tabs"""
    global demo_ai_processes, demo_browser_tabs, demo_summary
    try:
        data = request.get_json() or {}
        exam_url = data.get('exam_url', '')

        if demo_mode_enabled:
            # In demo mode, simulate killing all
            killed_ai_count = len(demo_ai_processes)
            closed_tabs_count = demo_summary['total_tabs'] - 1  # Keep exam tab

            # Clear demo AI processes
            demo_ai_processes = []

            # Clear demo browser tabs (keep exam tab)
            exam_tabs = []
            for browser, tabs in demo_browser_tabs.items():
                browser_exam_tabs = [tab for tab in tabs if 'localhost:5000' in tab.get('url', '')]
                demo_browser_tabs[browser] = browser_exam_tabs
                exam_tabs.extend(browser_exam_tabs)

            # Update summary
            demo_summary = {
                'total_tabs': len(exam_tabs),
                'browsers': {
                    'chrome': len([tab for tab in demo_browser_tabs['chrome']]),
                    'firefox': len([tab for tab in demo_browser_tabs['firefox']]),
                    'edge': len([tab for tab in demo_browser_tabs['edge']])
                }
            }

            results = {
                'ai_applications': {'killed': [{'name': f'Demo AI Process {i}'} for i in range(killed_ai_count)]},
                'browser_tabs': {'total_closed': closed_tabs_count},
                'success': True
            }
        else:
            # Set the exam URL to protect
            if exam_url:
                killer.set_exam_domain(exam_url)

            # Perform the termination
            results = killer.kill_all_targeted(exam_url)

        # Emit real-time update to all connected clients
        socketio.emit('kill_completed', results)

        return jsonify({
            'success': True,
            'data': results,
            'timestamp': time.time()
        })

    except Exception as e:
        logger.error(f"Error during kill all: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/kill-ai-only', methods=['POST'])
def kill_ai_only():
    """Kill only AI applications, preserve browser tabs"""
    global demo_ai_processes
    try:
        if demo_mode_enabled:
            # In demo mode, simulate killing AI apps only
            killed_count = len(demo_ai_processes)
            demo_ai_processes = []  # Clear all AI processes

            results = {
                'killed': [{'name': f'Demo AI Process {i}'} for i in range(killed_count)],
                'failed': [],
                'not_found': []
            }
        else:
            results = killer.kill_ai_applications()

        # Emit real-time update
        socketio.emit('ai_kill_completed', results)

        return jsonify({
            'success': True,
            'data': results,
            'timestamp': time.time()
        })

    except Exception as e:
        logger.error(f"Error killing AI applications: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/close-tabs-only', methods=['POST'])
def close_tabs_only():
    """Close browser tabs only, preserve AI applications"""
    global demo_browser_tabs, demo_summary
    try:
        data = request.get_json() or {}
        exam_url = data.get('exam_url', '')

        if demo_mode_enabled:
            # In demo mode, simulate closing tabs (keep exam tab)
            closed_count = demo_summary['total_tabs'] - 1  # Keep exam tab

            # Keep only exam tabs
            exam_tabs = []
            for browser, tabs in demo_browser_tabs.items():
                browser_exam_tabs = [tab for tab in tabs if 'localhost:5000' in tab.get('url', '')]
                demo_browser_tabs[browser] = browser_exam_tabs
                exam_tabs.extend(browser_exam_tabs)

            # Update summary
            demo_summary = {
                'total_tabs': len(exam_tabs),
                'browsers': {
                    'chrome': len([tab for tab in demo_browser_tabs['chrome']]),
                    'firefox': len([tab for tab in demo_browser_tabs['firefox']]),
                    'edge': len([tab for tab in demo_browser_tabs['edge']])
                }
            }

            results = {
                'chrome': {'closed': closed_count, 'preserved': 1, 'errors': []},
                'firefox': {'closed': 0, 'preserved': 0, 'errors': []},
                'edge': {'closed': 0, 'preserved': 0, 'errors': []},
                'total_closed': closed_count,
                'total_preserved': 1
            }
        else:
            if exam_url:
                killer.set_exam_domain(exam_url)

            results = killer.close_browser_tabs(preserve_exam_tab=True)

        # Emit real-time update
        socketio.emit('tabs_closed', results)

        return jsonify({
            'success': True,
            'data': results,
            'timestamp': time.time()
        })

    except Exception as e:
        logger.error(f"Error closing browser tabs: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

# Add demo mode route
@app.route('/api/demo-mode', methods=['POST'])
def enable_demo_mode():
    """Enable demo mode with simulated AI threats"""
    global demo_mode_enabled, demo_ai_processes, demo_browser_tabs, demo_summary
    demo_mode_enabled = True

    # Initialize fake AI processes
    demo_ai_processes = [
        {
            'pid': 99999,
            'name': 'ChatGPT.exe',
            'memory_percent': 15.5,
            'cpu_percent': 8.2,
            'status': 'running',
            'is_ai_app': True
        },
        {
            'pid': 99998,
            'name': 'Claude-Desktop.exe',
            'memory_percent': 12.3,
            'cpu_percent': 5.1,
            'status': 'running',
            'is_ai_app': True
        },
        {
            'pid': 99997,
            'name': 'GitHub-Copilot.exe',
            'memory_percent': 8.7,
            'cpu_percent': 3.2,
            'status': 'running',
            'is_ai_app': True
        }
    ]

    # Initialize fake browser tabs
    demo_browser_tabs = {
        'chrome': [
            {
                'id': 'demo1',
                'title': 'ChatGPT - OpenAI',
                'url': 'https://chat.openai.com/chat',
                'active': False
            },
            {
                'id': 'demo2',
                'title': 'Claude - Anthropic',
                'url': 'https://claude.ai/chat',
                'active': False
            },
            {
                'id': 'demo3',
                'title': 'Your Exam - Protected',
                'url': 'http://localhost:5000/',
                'active': True
            }
        ],
        'firefox': [
            {
                'id': 'demo4',
                'title': 'Gemini AI',
                'url': 'https://gemini.google.com/',
                'active': False
            }
        ],
        'edge': []
    }

    # Initialize summary
    demo_summary = {
        'total_tabs': 4,
        'browsers': {'chrome': 3, 'firefox': 1, 'edge': 0}
    }

    return jsonify({
        'success': True,
        'message': 'Demo mode enabled - fake AI threats created',
        'demo_processes': demo_ai_processes,
        'demo_tabs': demo_browser_tabs
    })

@app.route('/api/disable-demo-mode', methods=['POST'])
def disable_demo_mode():
    """Disable demo mode"""
    global demo_mode_enabled, demo_ai_processes, demo_browser_tabs, demo_summary
    demo_mode_enabled = False

    # Reset demo data
    demo_ai_processes = []
    demo_browser_tabs = {'chrome': [], 'firefox': [], 'edge': []}
    demo_summary = {'total_tabs': 0, 'browsers': {'chrome': 0, 'firefox': 0, 'edge': 0}}

    return jsonify({
        'success': True,
        'message': 'Demo mode disabled'
    })

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to ExHon monitoring server'})

    # Send initial data
    try:
        processes = monitor.get_running_processes()[:20]  # Top 20
        ai_processes = monitor.get_ai_processes()
        tabs = monitor.get_browser_tabs()
        system_info = monitor.get_system_info()

        emit('initial_data', {
            'processes': processes,
            'ai_processes': ai_processes,
            'browser_tabs': tabs,
            'system_info': system_info,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Error sending initial data: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring"""
    global monitoring_active, update_thread

    if not monitoring_active:
        monitoring_active = True
        update_thread = threading.Thread(target=monitoring_loop, daemon=True)
        update_thread.start()
        emit('monitoring_started', {'status': 'Monitoring started'})
        logger.info("Real-time monitoring started")

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop real-time monitoring"""
    global monitoring_active
    monitoring_active = False
    emit('monitoring_stopped', {'status': 'Monitoring stopped'})
    logger.info("Real-time monitoring stopped")

@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request"""
    try:
        data = get_current_monitoring_data()
        emit('monitoring_update', data)
    except Exception as e:
        logger.error(f"Error handling update request: {e}")
        emit('error', {'message': str(e)})

def monitoring_loop():
    """Background thread for real-time monitoring updates"""
    global monitoring_active

    while monitoring_active:
        try:
            data = get_current_monitoring_data()
            socketio.emit('monitoring_update', data)
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(10)  # Wait longer on error

def get_current_monitoring_data():
    """Get current monitoring data for real-time updates"""
    if demo_mode_enabled:
        # Return current demo data
        real_processes = monitor.get_running_processes()[:22]  # Make room for demo
        all_processes = real_processes + demo_ai_processes
        current_demo_tabs = demo_browser_tabs
        current_demo_summary = demo_summary
    else:
        processes = monitor.get_running_processes()[:25]  # Top 25
        ai_processes = monitor.get_ai_processes()
        all_processes = processes
        current_demo_tabs = monitor.get_browser_tabs()
        current_demo_summary = monitor.get_tab_summary()

    # Basic system stats
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
    except ImportError:
        cpu_percent = 0
        memory = type('Memory', (), {'percent': 0, 'used': 0, 'total': 0})()

    return {
        'processes': {
            'all': all_processes,
            'ai': demo_ai_processes if demo_mode_enabled else monitor.get_ai_processes(),
            'count': len(all_processes)
        },
        'browser_tabs': {
            'tabs': current_demo_tabs,
            'summary': current_demo_summary
        },
        'system_stats': {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': round(memory.used / (1024**3), 2),
            'memory_total_gb': round(memory.total / (1024**3), 2)
        },
        'timestamp': time.time(),
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': time.time()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': time.time()
    }), 500

@app.before_request
def log_request_info():
    """Log request information"""
    if request.path.startswith('/api/'):
        logger.info(f"API Request: {request.method} {request.path}")

if __name__ == '__main__':
    # Initialize browser debugging if possible
    try:
        monitor.start_browser_debugging()
        logger.info("Browser debugging protocols initialized")
    except Exception as e:
        logger.warning(f"Could not initialize browser debugging: {e}")

    # Run the application
    logger.info("Starting ExHon Proctored Exam Platform...")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, allow_unsafe_werkzeug=True)
