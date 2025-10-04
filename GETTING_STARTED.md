# 🚀 Getting Started with ExHon

**ExHon Proctored Exam Platform** - Your complete solution for secure online examinations with AI application monitoring and browser tab control.

## ⚡ Quick Start (30 seconds)

### Windows Users
1. **Download ExHon** and extract to a folder
2. **Double-click** `launch_exhon.bat`
3. **Wait** for Chrome to open with debugging
4. **Access dashboard** at `http://localhost:5000`
5. **Click "Close All"** when ready to start your exam

### Mac/Linux Users
1. **Download ExHon** and extract to a folder
2. **Run** `./launch_exhon.sh` in terminal
3. **Wait** for Chrome to open with debugging
4. **Access dashboard** at `http://localhost:5000`
5. **Click "Close All"** when ready to start your exam

---

## 🎯 What ExHon Does

### Real-Time Monitoring
- **Detects AI Applications**: Automatically finds ChatGPT, Claude, Gemini, Copilot, etc.
- **Tracks Browser Tabs**: Monitors all open tabs across Chrome, Firefox, Edge
- **System Overview**: Shows CPU, memory usage, and running processes
- **Live Updates**: Dashboard refreshes every 5 seconds automatically

### Smart Control
- **One-Click Cleanup**: "Close All" button terminates AI apps and browser tabs
- **Exam Protection**: Your current exam tab is automatically preserved
- **Selective Actions**: Kill only AI apps OR close only browser tabs
- **Safety Confirmations**: All destructive actions require user confirmation

### Professional Interface
- **Modern Dashboard**: Clean, responsive Bootstrap design
- **Activity Logging**: Complete audit trail of all actions
- **Real-Time Stats**: Live system performance monitoring
- **Color Coding**: AI applications highlighted in red for easy identification

---

## 📱 Dashboard Overview

When you access `http://localhost:5000`, you'll see:

### Top Cards (System Stats)
- **Running Processes**: Total count of active applications
- **AI Applications**: Number of detected AI tools (highlighted in red)
- **Browser Tabs**: Total open tabs across all browsers
- **Memory Usage**: Current system memory consumption percentage

### Left Panel: Running Applications
- **Process List**: All running applications with memory/CPU usage
- **AI Detection**: Red highlighting for ChatGPT, Claude, Gemini, etc.
- **Process Details**: PID, status, and resource consumption
- **Quick Actions**: "Kill AI Apps" button for selective termination

### Right Panel: Browser Tabs
- **Chrome Tabs**: Expandable list with titles and URLs
- **Firefox Tabs**: Mozilla browser tab detection
- **Edge Tabs**: Microsoft Edge tab monitoring
- **Protection Status**: Green badges for protected exam tabs

### Bottom Panel: Activity Log
- **Real-Time Logging**: Timestamped record of all actions
- **Color Coding**: Success (green), errors (red), warnings (yellow)
- **Auto-Scroll**: Latest entries always visible
- **Detailed History**: Complete audit trail for exam compliance

---

## 🛡️ Safety Features

### Exam Tab Protection
ExHon automatically detects and protects your current exam tab:
- **Domain Matching**: Identifies tabs from the same domain as your exam
- **Smart Preservation**: "Close All" keeps your exam session active
- **Visual Indicators**: Protected tabs show green "Protected" badges
- **Zero Data Loss**: Your exam progress is never interrupted

### Confirmation Dialogs
Every destructive action requires confirmation:
- **Preview Mode**: See exactly what will be closed before proceeding
- **Clear Descriptions**: Detailed explanation of each action
- **Cancel Option**: Easy to abort if you change your mind
- **Keyboard Shortcuts**: Ctrl+Shift+K for quick "Close All"

### Comprehensive Logging
- **Action Tracking**: Every button click and system change is logged
- **Timestamp Records**: Precise timing for compliance verification
- **Error Reporting**: Detailed messages for troubleshooting
- **Audit Trail**: Complete history for exam security verification

---

## 🔧 Common Use Cases

### Before Starting an Exam
1. **Open ExHon Dashboard**: `http://localhost:5000`
2. **Review Detected Items**: Check AI applications and browser tabs
3. **Test the System**: Use "Kill AI Apps" to verify functionality
4. **Navigate to Exam**: Open your exam in the Chrome window with debugging
5. **Final Cleanup**: Click "Close All" to clear distractions

### During an Exam
1. **Keep Dashboard Open**: Monitor system in a separate tab
2. **Check Periodically**: Verify no new AI applications started
3. **Use Selective Control**: Close specific browser tabs if needed
4. **Monitor Activity Log**: Track all system changes

### After an Exam
1. **Review Logs**: Check activity history for compliance
2. **Close Dashboard**: Stop the monitoring server (Ctrl+C)
3. **Normal Browsing**: Restart browsers normally without debugging

---

## 🎮 Control Actions Explained

### "Close All" (Red Button)
**What it does:**
- Terminates all AI applications (ChatGPT, Claude, Gemini, etc.)
- Closes all browser tabs in Chrome, Firefox, and Edge
- Preserves your current exam tab automatically
- Updates the dashboard immediately

**When to use:**
- Right before starting your exam
- When you notice AI applications running
- For comprehensive system cleanup

### "Kill AI Apps"
**What it does:**
- Terminates only AI-related processes
- Leaves all browser tabs open
- Preserves your browsing session

**When to use:**
- If you only want to close AI tools
- During an exam when new AI apps appear
- For selective cleanup without losing tabs

### "Close Tabs"
**What it does:**
- Closes all browser tabs except exam tabs
- Keeps all applications running
- Preserves system processes

**When to use:**
- Too many distracting tabs open
- Want to clean browsing without affecting apps
- Need to reduce browser memory usage

---

## 🔍 AI Application Detection

ExHon automatically detects these AI applications:

### Confirmed Detection
- **ChatGPT** desktop apps and web versions
- **Claude** (Anthropic) applications
- **Google Gemini/Bard** instances
- **GitHub Copilot** and related tools
- **Microsoft Copilot** applications
- **OpenAI** official applications

### Extended Detection
- **Tabnine** code completion
- **AWS CodeWhisperer** tools
- **Cursor** AI editor
- **Windsurf** development tools
- **Bolt** and **Replit** AI features
- **Perplexity** search applications

### Custom Applications
You can add detection for custom AI tools by editing `killer.py` and adding keywords to the `ai_keywords` list.

---

## ⚙️ Browser Compatibility

### Google Chrome (Primary)
- **Full Support**: Complete tab detection and control
- **Debugging Protocol**: Uses Chrome DevTools Protocol
- **Recommended**: Best compatibility and features

### Mozilla Firefox
- **Basic Support**: Tab detection with limited control
- **Remote Debugging**: Uses Firefox debugging protocol
- **Optional**: Secondary browser monitoring

### Microsoft Edge
- **Full Support**: Same as Chrome (Chromium-based)
- **Windows Optimized**: Best performance on Windows
- **Alternative**: Good substitute for Chrome

### Safari (macOS)
- **Not Supported**: Requires developer mode and manual setup
- **Future Enhancement**: Planned for version 2.0

---

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Ubuntu 18.04
- **Python**: Version 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 500MB free space
- **Browser**: Chrome, Firefox, or Edge

### Recommended Setup
- **OS**: Latest Windows 11, macOS Monterey, or Ubuntu 22.04
- **Python**: Python 3.11+ for best performance
- **RAM**: 8GB+ for smooth operation
- **Storage**: 1GB for logs and temporary files
- **Network**: Stable internet for initial setup

---

## Troubleshooting Quick Fixes

### "No processes detected"
```bash
# Windows: Run as Administrator
Right-click launch_exhon.bat → Run as Administrator

# Mac/Linux: Use sudo if needed
sudo ./launch_exhon.sh
```

### "Browser tabs not showing"
1. **Restart browsers** with debugging enabled
2. **Check firewall** isn't blocking ports 9222, 9229, 9225
3. **Use provided scripts** in the `scripts/` folder

### "Connection failed"
1. **Check port 5000** isn't blocked or in use
2. **Try localhost instead** of 127.0.0.1
3. **Disable antivirus** temporarily if blocking

### "Module not found errors"
```bash
# Reinstall dependencies
pip install -r requirements.txt

# If still failing, try:
python -m pip install --upgrade pip
pip install --user -r requirements.txt
```

---

## Support & Help

### Self-Help Resources
1. **Run System Test**: `python test_system.py`
2. **Check Logs**: Look at console output and `setup.log`
3. **Read Documentation**: `README.md` and `INSTALLATION.md`
4. **Verify Setup**: Ensure all required files are present

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides in repository
- **Community**: User discussions and solutions
- **Email Support**: Available for critical issues

### Before Contacting Support
Please provide:
- Your operating system and version
- Python version (`python --version`)
- Complete error messages
- Steps to reproduce the issue
- Output from `python test_system.py`

---

## What's Next?

### Immediate Next Steps
1. **Bookmark the Dashboard**: `http://localhost:5000`
2. **Test All Functions**: Try each button in a safe environment
3. **Configure for Your Exam**: Set up browser debugging
4. **Practice Run**: Do a complete test before your actual exam

### Advanced Usage
- **Custom AI Detection**: Add your specific tools to the detection list
- **Port Configuration**: Change default ports if needed
- **Logging Setup**: Configure detailed audit trails
- **Integration**: Connect with your institution's exam platform

---

## Exam Day Checklist

### Pre-Exam Setup (10 minutes)
- [ ] Download and extract ExHon to a dedicated folder
- [ ] Run the appropriate launcher (`launch_exhon.bat` or `launch_exhon.sh`)
- [ ] Verify Chrome opens with debugging enabled
- [ ] Access dashboard at `http://localhost:5000`
- [ ] Test "Close All" button with confirmation dialog
- [ ] Ensure your exam website works in the debug Chrome window

### Exam Start (2 minutes)
- [ ] Navigate to your exam URL in the debug Chrome window
- [ ] Open ExHon dashboard in a separate tab
- [ ] Click "Close All" to eliminate all distractions
- [ ] Verify your exam tab remains protected and functional
- [ ] Begin your exam with confidence

### During Exam (ongoing)
- [ ] Keep dashboard tab open for monitoring
- [ ] Check occasionally for new AI applications
- [ ] Use selective controls if needed
- [ ] Monitor activity log for any issues

### Post-Exam
- [ ] Review activity log for compliance record
- [ ] Stop ExHon server (Ctrl+C in terminal)
- [ ] Close debug Chrome window
- [ ] Return to normal browser usage

---

**You're Ready!**

ExHon is now configured and ready to provide secure, monitored exam sessions. The platform will help ensure academic integrity while maintaining a smooth examination experience.

For detailed technical information, see `README.md`.
For installation troubleshooting, see `INSTALLATION.md`.

**Good luck with your exam!**