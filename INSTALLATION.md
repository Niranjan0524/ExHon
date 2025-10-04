# ExHon Installation Guide

Complete step-by-step installation guide for the ExHon Proctored Exam Platform.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Disk Space**: 500MB free space
- **Network**: Internet connection for initial setup

### Required Software
- **Python 3.8+**: [Download from python.org](https://www.python.org/downloads/)
- **Google Chrome** (recommended): [Download Chrome](https://www.google.com/chrome/)
- **Firefox** (optional): [Download Firefox](https://www.mozilla.org/firefox/)
- **Microsoft Edge** (optional, Windows): Usually pre-installed

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

1. **Download and extract ExHon**
   ```bash
   # Extract the downloaded files to a folder of your choice
   cd ExHon
   ```

2. **Run the automated installer**
   ```bash
   python setup.py
   ```

3. **Start the application**
   ```bash
   python start.py
   ```

4. **Access the dashboard**
   - Open your browser and go to: `http://localhost:5000`

### Option 2: Manual Installation

If the automated setup doesn't work, follow these manual steps:

## 📝 Manual Installation Steps

### Step 1: Verify Python Installation

```bash
python --version
# or
python3 --version
```

Expected output: `Python 3.8.x` or higher

**If Python is not installed:**
- Windows: Download from [python.org](https://www.python.org/downloads/) and check "Add to PATH"
- macOS: `brew install python3` or download from python.org
- Linux: `sudo apt-get install python3 python3-pip`

### Step 2: Install Python Dependencies

1. **Navigate to the ExHon directory**
   ```bash
   cd ExHon
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   **If you encounter permission errors:**
   ```bash
   # Windows
   pip install --user -r requirements.txt
   
   # macOS/Linux
   pip3 install --user -r requirements.txt
   ```

### Step 3: Set Up Browser Debugging

#### For Chrome (Primary Browser)

**Windows:**
1. Create a batch file `start_chrome_debug.bat`:
   ```batch
   @echo off
   echo Starting Chrome with debugging enabled...
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%TEMP%\chrome_debug" --disable-background-timer-throttling
   ```

2. Run the batch file before starting ExHon

**macOS:**
1. Create a shell script `start_chrome_debug.sh`:
   ```bash
   #!/bin/bash
   echo "Starting Chrome with debugging enabled..."
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug" --disable-background-timer-throttling &
   ```

2. Make executable and run:
   ```bash
   chmod +x start_chrome_debug.sh
   ./start_chrome_debug.sh
   ```

**Linux:**
1. Create a shell script `start_chrome_debug.sh`:
   ```bash
   #!/bin/bash
   echo "Starting Chrome with debugging enabled..."
   google-chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_debug" --disable-background-timer-throttling &
   ```

2. Make executable and run:
   ```bash
   chmod +x start_chrome_debug.sh
   ./start_chrome_debug.sh
   ```

#### For Firefox (Optional)

1. Start Firefox with debugging:
   ```bash
   # Windows
   firefox.exe -start-debugger-server=9229
   
   # macOS/Linux
   firefox -start-debugger-server=9229
   ```

#### For Edge (Optional, Windows Only)

1. Create batch file `start_edge_debug.bat`:
   ```batch
   @echo off
   echo Starting Edge with debugging enabled...
   "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9225 --user-data-dir="%TEMP%\edge_debug"
   ```

### Step 4: Test the Installation

1. **Run the test script**
   ```bash
   python test_system.py
   ```

2. **Expected output:**
   ```
   ✅ Python 3.x.x - Compatible
   ✅ psutil - System process monitoring
   ✅ flask - Web framework
   ✅ System monitoring - Retrieved successfully
   ...
   🎉 All tests passed! ExHon is ready to run.
   ```

### Step 5: Start the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Alternative startup**
   ```bash
   python start.py  # Includes automatic browser setup
   ```

3. **Access the dashboard**
   - Open your browser
   - Navigate to: `http://localhost:5000`

## 🛠️ Configuration

### Basic Configuration

Create or edit `config.json`:

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

### Advanced Configuration

#### Custom AI Application Detection

Edit `killer.py` to add custom AI applications:

```python
self.ai_keywords = [
    'chatgpt', 'claude', 'gemini', 'copilot',
    'your-custom-ai-tool',  # Add your custom apps here
    'another-ai-app'
]
```

#### Custom Browser Ports

If the default ports conflict, edit `monitor.py` and `killer.py`:

```python
self.browser_ports = {
    'chrome': [9222, 9223, 9224],    # Change these ports
    'edge': [9225, 9226, 9227],      # if needed
    'firefox': [9228, 9229, 9230]
}
```

## 🔧 Troubleshooting

### Common Issues

#### Issue: "No processes detected"
**Solution:**
- Run as Administrator (Windows) or with sudo (Linux/macOS)
- Check antivirus software isn't blocking process enumeration
- Verify Python has necessary permissions

#### Issue: "Browser tabs not showing"
**Solution:**
- Ensure browser debugging is enabled
- Check that browsers were started with debug flags
- Verify ports 9222, 9229, 9225 are not blocked
- Try restarting browsers with debug commands

#### Issue: "Connection failed"
**Solution:**
- Check if port 5000 is available
- Verify Flask server started successfully
- Check firewall settings
- Try using `127.0.0.1:5000` instead of `localhost:5000`

#### Issue: "Module not found" errors
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# If using virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Issue: "Permission denied" on Linux/macOS
**Solution:**
```bash
# Give execution permissions
chmod +x start.py
chmod +x test_system.py

# Or run with python
python3 start.py
```

### Debug Mode

Enable verbose logging:

```bash
# Set debug environment variable
export FLASK_DEBUG=1  # Linux/macOS
set FLASK_DEBUG=1     # Windows

python app.py
```

### Port Conflicts

If port 5000 is in use, change it in `app.py`:

```python
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5001, debug=True)  # Change port here
```

## 🧪 Verification

### Quick Health Check

1. **System Test**
   ```bash
   python test_system.py
   ```

2. **Manual Verification**
   - Open the dashboard: `http://localhost:5000`
   - Verify process list shows running applications
   - Check that AI applications are highlighted
   - Confirm browser tabs are detected
   - Test the "Close All" button (with confirmation)

### Expected Dashboard Features

- **Live Process Monitoring**: Real-time list of running applications
- **AI Detection**: Red highlighting of AI-related processes
- **Browser Integration**: Tabs from Chrome, Firefox, Edge
- **Control Panel**: Buttons for selective termination
- **Activity Log**: Real-time logging of all actions
- **System Stats**: CPU and memory usage

## 📚 Next Steps

1. **Bookmark the Dashboard**: `http://localhost:5000`
2. **Read the User Guide**: See `README.md` for usage instructions
3. **Configure Exam Settings**: Set your exam domain for tab protection
4. **Test in Safe Mode**: Try all functions before actual exam use
5. **Create Shortcuts**: Use the auto-generated desktop shortcuts

## 🆘 Getting Help

### Log Files
- **Setup Log**: `setup.log`
- **Application Logs**: Console output when running `python app.py`
- **Browser Logs**: Check browser console (F12) for JavaScript errors

### Support Resources
1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: `README.md` for detailed usage
3. **Test Suite**: `python test_system.py` for diagnostics

### Contact Information
For technical support or questions:
- Create an issue on the GitHub repository
- Include your system information and error logs
- Provide steps to reproduce any issues

---

**Installation Complete!** 🎉

Your ExHon Proctored Exam Platform is now ready to use. Access it at `http://localhost:5000` and enjoy secure, monitored exam sessions.