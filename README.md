# ExHon - Proctored Exam Platform

A comprehensive web-based proctoring system that monitors and controls running applications and browser tabs during online examinations. Built with Flask, Bootstrap, and real-time WebSocket communication.

![ExHon Dashboard](https://img.shields.io/badge/Platform-Web-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

### Core Monitoring Capabilities
- **Process Monitoring**: Real-time detection of all running applications
- **AI Application Detection**: Automatically identifies AI tools (ChatGPT, Claude, Gemini, Copilot, etc.)
- **Browser Tab Tracking**: Monitors open tabs across Chrome, Firefox, and Edge
- **System Statistics**: CPU usage, memory consumption, and system performance

### Control Features
- **Selective Termination**: Close AI applications while preserving exam environment
- **Browser Tab Management**: Close all tabs except the current exam tab
- **One-Click "Close All"**: Comprehensive cleanup with safety protections
- **Exam Tab Protection**: Automatically preserves the active exam session

### User Interface
- **Modern Dashboard**: Professional Bootstrap-based interface
- **Real-Time Updates**: Live monitoring via WebSocket connections
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Activity Logging**: Detailed log of all monitoring and control actions

## 🏗️ Architecture

```
ExHon/
├── app.py                 # Main Flask application
├── monitor.py             # System monitoring utilities
├── killer.py              # Process/tab termination logic
├── setup.py               # System setup and configuration
├── config.json            # Application configuration
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── dashboard.css  # Custom styling
│   └── js/
│       └── dashboard.js   # Frontend logic & WebSocket handling
├── templates/
│   └── dashboard.html     # Main dashboard template
└── scripts/               # Browser debugging startup scripts
    ├── start_chrome_debug.bat/.sh
    ├── start_firefox_debug.bat/.sh
    └── start_edge_debug.bat/.sh
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Chrome (recommended)
- Firefox and/or Edge (optional)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ExHon
```

2. **Run the setup script:**
```bash
python setup.py
```

This will:
- Install Python dependencies
- Configure browser debugging protocols
- Create startup scripts
- Generate configuration files
- Create desktop shortcuts

3. **Start browser debugging** (if not auto-started):
```bash
# Windows
scripts/start_chrome_debug.bat

# macOS/Linux
./scripts/start_chrome_debug.sh
```

4. **Launch the application:**
```bash
python app.py
```

5. **Access the dashboard:**
Open http://localhost:5000 in your browser

## 📖 Detailed Setup

### Manual Browser Configuration

#### Chrome Setup
```bash
# Windows
chrome.exe --remote-debugging-port=9222 --user-data-dir="%TEMP%\chrome_debug"

# macOS
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug"

# Linux
google-chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug"
```

#### Firefox Setup
1. Create a new profile: `firefox -ProfileManager`
2. Set `devtools.debugger.remote-enabled` to `true` in `about:config`
3. Start with debugging: `firefox -start-debugger-server=9229`

#### Edge Setup (Chromium-based)
```bash
msedge.exe --remote-debugging-port=9225 --user-data-dir="%TEMP%\edge_debug"
```

## 🎮 Usage Guide

### Dashboard Overview

The main dashboard displays four key sections:

1. **System Overview Cards**
   - Running Processes count
   - AI Applications detected
   - Browser Tabs open
   - Memory Usage percentage

2. **Running Applications Panel**
   - List of active processes with memory/CPU usage
   - AI applications highlighted in red
   - Process termination controls

3. **Browser Tabs Panel**
   - Organized by browser (Chrome, Firefox, Edge)
   - Tab titles and URLs
   - Protection status indicators

4. **Activity Log**
   - Real-time logging of all actions
   - Timestamped entries
   - Color-coded by severity

### Control Actions

#### Close All (Red Button)
- Terminates all detected AI applications
- Closes all browser tabs except the exam tab
- Preserves the current session
- Requires confirmation

#### Selective Actions
- **Kill AI Apps**: Terminates only AI-related processes
- **Close Tabs**: Closes browser tabs while keeping applications running
- **Refresh**: Updates the monitoring data

### Safety Features

- **Exam Tab Protection**: Automatically identifies and preserves the current exam tab
- **Confirmation Dialogs**: All destructive actions require user confirmation
- **Activity Logging**: Complete audit trail of all actions
- **Graceful Termination**: Attempts clean shutdown before force-killing processes

## 🔧 Configuration

### config.json Structure
```json
{
    "browser_debugging": {
        "chrome_port": 9222,
        "firefox_port": 9229,
        "edge_port": 9225
    },
    "monitoring": {
        "update_interval": 5,
        "max_log_entries": 50,
        "auto_start_monitoring": true
    },
    "security": {
        "preserve_exam_tab": true,
        "require_confirmation": true,
        "log_all_actions": true
    }
}
```

### Customizing AI Application Detection

Edit the `killer.py` file to add or modify AI application detection:

```python
self.ai_keywords = [
    'chatgpt', 'claude', 'gemini', 'copilot',
    'your-custom-ai-app',  # Add custom applications here
    # ... other keywords
]
```

## 🌐 API Endpoints

### Monitoring Endpoints
- `GET /api/system-info` - System information
- `GET /api/processes` - Running processes list
- `GET /api/browser-tabs` - Browser tabs information
- `GET /api/preview-termination` - Preview termination actions

### Control Endpoints
- `POST /api/kill-all` - Terminate AI apps and close tabs
- `POST /api/kill-ai-only` - Terminate only AI applications
- `POST /api/close-tabs-only` - Close only browser tabs

### WebSocket Events
- `connect/disconnect` - Connection status
- `monitoring_update` - Real-time data updates
- `kill_completed` - Action completion notifications

## 🔒 Security Considerations

### Browser Security
- Browser debugging protocols require local network access
- Debugging ports are only accessible from localhost
- No external network exposure by default

### Process Access
- Requires appropriate system permissions to enumerate processes
- Some system processes may be protected and inaccessible
- Antivirus software might flag process termination activities

### Data Privacy
- No user data is collected or transmitted externally
- All monitoring data remains local to the machine
- No internet connectivity required for core functionality

## 🐛 Troubleshooting

### Common Issues

#### "No processes detected"
- Check if the application has sufficient permissions
- Run as administrator on Windows
- Ensure Python has process enumeration rights

#### "Browser tabs not showing"
- Verify browser debugging is enabled
- Check that debugging ports are not blocked by firewall
- Ensure browsers were started with debugging flags

#### "Connection failed" 
- Verify Flask server is running on port 5000
- Check for port conflicts with other applications
- Ensure WebSocket connections are not blocked

#### "AI applications not detected"
- Verify the application names match detection patterns
- Check if processes are running under different names
- Update the AI keywords list in `killer.py`

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export FLASK_DEBUG=1
python app.py
```

### Log Files
- `setup.log` - Setup process log
- Console output - Real-time application logs

## 🧪 Testing

### System Testing
```bash
# Test system access
python -c "from monitor import SystemMonitor; m = SystemMonitor(); print(len(m.get_running_processes()))"

# Test process termination (dry run)
python -c "from killer import preview_termination; print(preview_termination())"
```

### Browser Testing
1. Open multiple tabs in different browsers
2. Start applications with AI-related names
3. Verify detection and termination functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bootstrap team for the excellent UI framework
- Flask and Flask-SocketIO for the web framework
- psutil library for system monitoring capabilities
- The open-source community for inspiration and tools

## 🔮 Roadmap

### Version 2.0 Planned Features
- [ ] Cross-platform Safari support
- [ ] Network monitoring capabilities
- [ ] Screen recording integration
- [ ] Multi-user proctoring support
- [ ] Cloud-based configuration management
- [ ] Advanced AI detection using ML models
- [ ] Mobile companion app
- [ ] Integration with LMS platforms

### Known Limitations
- Firefox remote debugging has limited reliability
- Safari debugging requires developer mode
- Some system processes cannot be terminated
- Browser tab detection varies by browser version
- Windows UAC may require elevation for some processes

---

**ExHon** - Empowering secure online examinations through intelligent monitoring and control.

For support, please create an issue on GitHub or contact the development team.