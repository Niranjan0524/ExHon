/**
 * ExHon Dashboard JavaScript
 * Handles real-time monitoring, WebSocket communication, and user interactions
 */

class ExHonDashboard {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.monitoringActive = false;
    this.currentExamUrl = this.getCurrentUrl();
    this.updateInterval = null;

    // UI Elements
    this.elements = {
      connectionStatus: document.getElementById("connection-status"),
      closeAllBtn: document.getElementById("close-all-btn"),
      killAiBtn: document.getElementById("kill-ai-btn"),
      closeTabsBtn: document.getElementById("close-tabs-btn"),
      refreshProcessesBtn: document.getElementById("refresh-processes-btn"),
      refreshTabsBtn: document.getElementById("refresh-tabs-btn"),
      processesTable: document.getElementById("processes-table"),
      activityLog: document.getElementById("activity-log"),
      alertContainer: document.getElementById("alert-container"),
      loadingOverlay: document.getElementById("loading-overlay"),

      // Stats
      totalProcesses: document.getElementById("total-processes"),
      aiProcesses: document.getElementById("ai-processes"),
      browserTabs: document.getElementById("browser-tabs"),
      memoryUsage: document.getElementById("memory-usage"),

      // Browser tab elements
      chromeCount: document.getElementById("chrome-count"),
      firefoxCount: document.getElementById("firefox-count"),
      edgeCount: document.getElementById("edge-count"),
      chromeTabList: document.getElementById("chrome-tab-list"),
      firefoxTabList: document.getElementById("firefox-tab-list"),
      edgeTabList: document.getElementById("edge-tab-list"),

      // Modal elements
      confirmModal: new bootstrap.Modal(
        document.getElementById("confirmModal"),
      ),
      actionPreview: document.getElementById("action-preview"),
      confirmAction: document.getElementById("confirm-action"),
    };

    this.pendingAction = null;
    this.init();
  }

  init() {
    this.initializeSocketIO();
    this.bindEvents();
    this.startPeriodicUpdates();
    this.logActivity("System initialized", "info");
  }

  getCurrentUrl() {
    return window.location.href;
  }

  initializeSocketIO() {
    try {
      this.socket = io();

      this.socket.on("connect", () => {
        this.isConnected = true;
        this.updateConnectionStatus(true);
        this.logActivity("Connected to monitoring server", "success");
        this.socket.emit("start_monitoring");
      });

      this.socket.on("disconnect", () => {
        this.isConnected = false;
        this.updateConnectionStatus(false);
        this.logActivity("Disconnected from monitoring server", "error");
      });

      this.socket.on("initial_data", (data) => {
        this.updateUI(data);
        this.enableControls();
      });

      this.socket.on("monitoring_update", (data) => {
        this.updateUI(data);
      });

      this.socket.on("kill_completed", (data) => {
        this.handleKillCompleted(data);
      });

      this.socket.on("ai_kill_completed", (data) => {
        this.handleAiKillCompleted(data);
      });

      this.socket.on("tabs_closed", (data) => {
        this.handleTabsClosed(data);
      });

      this.socket.on("error", (data) => {
        this.showAlert("Error: " + data.message, "danger");
        this.logActivity("Error: " + data.message, "error");
      });
    } catch (error) {
      console.error("Socket.IO initialization failed:", error);
      this.logActivity("Failed to initialize real-time monitoring", "error");
    }
  }

  bindEvents() {
    // Main action buttons
    this.elements.closeAllBtn.addEventListener("click", () => {
      this.showConfirmationModal("kill-all");
    });

    this.elements.killAiBtn.addEventListener("click", () => {
      this.showConfirmationModal("kill-ai");
    });

    this.elements.closeTabsBtn.addEventListener("click", () => {
      this.showConfirmationModal("close-tabs");
    });

    // Refresh buttons
    this.elements.refreshProcessesBtn.addEventListener("click", () => {
      this.refreshProcesses();
    });

    this.elements.refreshTabsBtn.addEventListener("click", () => {
      this.refreshTabs();
    });

    // Modal confirmation
    this.elements.confirmAction.addEventListener("click", () => {
      this.executeAction();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === "K") {
        e.preventDefault();
        this.showConfirmationModal("kill-all");
      }
    });
  }

  updateConnectionStatus(connected) {
    const status = this.elements.connectionStatus;
    if (connected) {
      status.innerHTML = '<i class="bi bi-wifi"></i> Connected';
      status.className = "badge bg-success connection-online";
    } else {
      status.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
      status.className = "badge bg-danger connection-offline";
    }
  }

  enableControls() {
    this.elements.closeAllBtn.disabled = false;
    this.elements.killAiBtn.disabled = false;
    this.elements.closeTabsBtn.disabled = false;
  }

  updateUI(data) {
    if (data.processes) {
      this.updateProcessesDisplay(data.processes);
    }

    if (data.browser_tabs) {
      this.updateBrowserTabsDisplay(data.browser_tabs);
    }

    if (data.system_stats) {
      this.updateSystemStats(data.system_stats);
    }

    // Update timestamp
    if (data.datetime) {
      this.logActivity(`Updated: ${data.datetime}`, "info");
    }
  }

  updateProcessesDisplay(processesData) {
    const tbody = this.elements.processesTable;
    const allProcesses = processesData.all || [];
    const aiProcesses = processesData.ai || [];

    // Update stats
    this.elements.totalProcesses.textContent =
      processesData.count || allProcesses.length;
    this.elements.aiProcesses.textContent = aiProcesses.length;

    // Clear and populate table
    tbody.innerHTML = "";

    if (allProcesses.length === 0) {
      tbody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-muted py-3">
                        <i class="bi bi-exclamation-triangle"></i> No processes detected
                    </td>
                </tr>
            `;
      return;
    }

    allProcesses.slice(0, 25).forEach((process) => {
      const row = this.createProcessRow(process);
      tbody.appendChild(row);
    });
  }

  createProcessRow(process) {
    const row = document.createElement("tr");
    const isAI = process.is_ai_app;
    const isHighMemory = process.memory_percent > 10;

    if (isAI) {
      row.classList.add("ai-process-row");
    } else if (isHighMemory) {
      row.classList.add("high-memory-process");
    }

    const statusIcon = this.getStatusIcon(process.status);
    const badges = this.createProcessBadges(process);

    row.innerHTML = `
            <td>
                <div class="d-flex align-items-center">
                    ${statusIcon}
                    <div class="ms-2">
                        <div class="fw-medium">${this.escapeHtml(process.name)}</div>
                        <small class="text-muted">PID: ${process.pid}</small>
                        ${badges}
                    </div>
                </div>
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="progress me-2" style="width: 60px; height: 6px;">
                        <div class="progress-bar ${isHighMemory ? "bg-warning" : "bg-success"}"
                             style="width: ${Math.min(process.memory_percent, 100)}%"></div>
                    </div>
                    <span class="small">${process.memory_percent}%</span>
                </div>
            </td>
            <td>
                <span class="small">${process.cpu_percent}%</span>
            </td>
            <td>
                <span class="badge ${this.getStatusBadgeClass(process.status)} small">
                    ${process.status}
                </span>
            </td>
        `;

    return row;
  }

  createProcessBadges(process) {
    let badges = "";
    if (process.is_ai_app) {
      badges += '<span class="badge badge-ai ms-1">AI</span>';
    }
    if (process.memory_percent > 10) {
      badges += '<span class="badge badge-high-memory ms-1">High Memory</span>';
    }
    return badges;
  }

  getStatusIcon(status) {
    const icons = {
      running: '<span class="status-indicator status-running"></span>',
      sleeping: '<span class="status-indicator status-sleeping"></span>',
      stopped: '<span class="status-indicator status-stopped"></span>',
    };
    return icons[status] || '<span class="status-indicator"></span>';
  }

  getStatusBadgeClass(status) {
    const classes = {
      running: "bg-success",
      sleeping: "bg-warning text-dark",
      stopped: "bg-danger",
      zombie: "bg-secondary",
    };
    return classes[status] || "bg-secondary";
  }

  updateBrowserTabsDisplay(tabsData) {
    const tabs = tabsData.tabs || {};
    const summary = tabsData.summary || {};

    // Update counts
    this.elements.chromeCount.textContent = summary.browsers?.chrome || 0;
    this.elements.firefoxCount.textContent = summary.browsers?.firefox || 0;
    this.elements.edgeCount.textContent = summary.browsers?.edge || 0;
    this.elements.browserTabs.textContent = summary.total_tabs || 0;

    // Update tab lists
    this.updateBrowserTabList("chrome", tabs.chrome || []);
    this.updateBrowserTabList("firefox", tabs.firefox || []);
    this.updateBrowserTabList("edge", tabs.edge || []);
  }

  updateBrowserTabList(browser, tabs) {
    const listElement = this.elements[`${browser}TabList`];
    listElement.innerHTML = "";

    if (tabs.length === 0) {
      listElement.innerHTML = `
                <div class="list-group-item text-muted">
                    <i class="bi bi-info-circle"></i> No ${browser} tabs detected
                </div>
            `;
      return;
    }

    tabs.forEach((tab) => {
      const item = this.createTabItem(tab);
      listElement.appendChild(item);
    });
  }

  createTabItem(tab) {
    const div = document.createElement("div");
    div.className = "list-group-item tab-item";

    const isProtected = this.isExamTab(tab.url);
    if (isProtected) {
      div.classList.add("protected");
    } else {
      div.classList.add("will-close");
    }

    const protectedBadge = isProtected
      ? '<span class="badge bg-success ms-2"><i class="bi bi-shield-check"></i> Protected</span>'
      : '<span class="badge bg-danger ms-2"><i class="bi bi-x-circle"></i> Will Close</span>';

    div.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <div class="tab-title">${this.escapeHtml(tab.title || "Untitled")}</div>
                    <div class="tab-url">${this.escapeHtml(this.truncateUrl(tab.url))}</div>
                </div>
                ${protectedBadge}
            </div>
        `;

    return div;
  }

  isExamTab(url) {
    if (!url) return false;
    const currentDomain = new URL(this.currentExamUrl).hostname;
    try {
      const tabDomain = new URL(url).hostname;
      return tabDomain === currentDomain;
    } catch {
      return false;
    }
  }

  truncateUrl(url, maxLength = 60) {
    if (!url) return "";
    return url.length > maxLength ? url.substring(0, maxLength) + "..." : url;
  }

  updateSystemStats(stats) {
    if (stats.memory_percent !== undefined) {
      this.elements.memoryUsage.textContent = `${stats.memory_percent}%`;
    }
  }

  showConfirmationModal(action) {
    this.pendingAction = action;

    const actionMessages = {
      "kill-all": {
        title: "Close All Applications and Tabs",
        items: [
          "Terminate all AI applications (ChatGPT, Claude, Gemini, etc.)",
          "Close all browser tabs except this exam tab",
          "Preserve the current exam session",
        ],
      },
      "kill-ai": {
        title: "Terminate AI Applications",
        items: [
          "Close all AI-related applications",
          "Keep browser tabs open",
          "Preserve exam environment",
        ],
      },
      "close-tabs": {
        title: "Close Browser Tabs",
        items: [
          "Close all browser tabs in Chrome, Firefox, and Edge",
          "Preserve this exam tab",
          "Keep applications running",
        ],
      },
    };

    const config = actionMessages[action];
    document.querySelector("#confirmModal .modal-title").textContent =
      config.title;

    this.elements.actionPreview.innerHTML = config.items
      .map((item) => `<li>${item}</li>`)
      .join("");

    this.elements.confirmModal.show();
  }

  executeAction() {
    if (!this.pendingAction) return;

    this.showLoading(true);
    this.elements.confirmModal.hide();

    const actions = {
      "kill-all": () => this.killAll(),
      "kill-ai": () => this.killAI(),
      "close-tabs": () => this.closeTabs(),
    };

    const actionFn = actions[this.pendingAction];
    if (actionFn) {
      actionFn();
    }

    this.pendingAction = null;
  }

  async killAll() {
    try {
      const response = await fetch("/api/kill-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exam_url: this.currentExamUrl }),
      });

      const result = await response.json();

      if (result.success) {
        this.showAlert(
          "Successfully terminated all targeted applications and tabs",
          "success",
        );
        this.logActivity("Kill All operation completed", "success");
      } else {
        this.showAlert("Error during termination: " + result.error, "danger");
        this.logActivity("Kill All operation failed: " + result.error, "error");
      }
    } catch (error) {
      this.showAlert("Network error during termination", "danger");
      this.logActivity("Network error: " + error.message, "error");
    } finally {
      this.showLoading(false);
    }
  }

  async killAI() {
    try {
      const response = await fetch("/api/kill-ai-only", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();

      if (result.success) {
        const killedCount = result.data.killed?.length || 0;
        this.showAlert(
          `Successfully terminated ${killedCount} AI applications`,
          "success",
        );
        this.logActivity(
          `AI applications terminated: ${killedCount}`,
          "success",
        );
      } else {
        this.showAlert(
          "Error terminating AI applications: " + result.error,
          "danger",
        );
      }
    } catch (error) {
      this.showAlert("Network error during AI termination", "danger");
    } finally {
      this.showLoading(false);
    }
  }

  async closeTabs() {
    try {
      const response = await fetch("/api/close-tabs-only", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exam_url: this.currentExamUrl }),
      });

      const result = await response.json();

      if (result.success) {
        const closedCount = result.data.total_closed || 0;
        this.showAlert(
          `Successfully closed ${closedCount} browser tabs`,
          "success",
        );
        this.logActivity(`Browser tabs closed: ${closedCount}`, "success");
      } else {
        this.showAlert("Error closing tabs: " + result.error, "danger");
      }
    } catch (error) {
      this.showAlert("Network error during tab closure", "danger");
    } finally {
      this.showLoading(false);
    }
  }

  handleKillCompleted(data) {
    this.showLoading(false);
    const aiKilled = data.ai_applications?.killed?.length || 0;
    const tabsClosed = data.browser_tabs?.total_closed || 0;

    // Immediately update UI counters
    this.animateCounter(this.elements.aiProcesses, 0);
    this.animateCounter(this.elements.threatCounter, 0);
    this.animateCounter(this.elements.aiCountBadge, 0);

    // Update browser tab counters (keep only protected exam tabs)
    this.animateCounter(this.elements.browserTabs, 1); // Usually 1 exam tab remains

    // Update status messages
    if (this.elements.aiStatus) {
      this.elements.aiStatus.textContent = "No threats";
      this.elements.aiStatus.className = "opacity-75 text-success";
    }

    this.showAlert(
      `🎯 Operation completed: ${aiKilled} AI apps terminated, ${tabsClosed} tabs closed`,
      "success",
    );

    this.logActivity(
      `🎯 Kill All completed - AI apps: ${aiKilled}, Tabs: ${tabsClosed}`,
      "success",
    );

    // Request fresh data to sync with server
    setTimeout(() => this.requestUpdate(), 500);
  }

  handleAiKillCompleted(data) {
    const killedCount = data.killed?.length || 0;

    // Immediately update AI-related counters to 0
    this.animateCounter(this.elements.aiProcesses, 0);
    this.animateCounter(this.elements.threatCounter, 0);
    this.animateCounter(this.elements.aiCountBadge, 0);

    // Update AI status
    if (this.elements.aiStatus) {
      this.elements.aiStatus.textContent = "No threats";
      this.elements.aiStatus.className = "opacity-75 text-success";
    }

    // Remove AI processes from table immediately
    this.removeAiProcessesFromTable();

    this.logActivity(
      `🤖 AI applications terminated: ${killedCount}`,
      "success",
    );
    setTimeout(() => this.requestUpdate(), 500);
  }

  handleTabsClosed(data) {
    const closedCount = data.total_closed || 0;
    const preservedCount = data.total_preserved || 1;

    // Update browser tab counters immediately
    this.animateCounter(this.elements.browserTabs, preservedCount);
    this.animateCounter(this.elements.chromeCount, preservedCount);
    this.animateCounter(this.elements.firefoxCount, 0);
    this.animateCounter(this.elements.edgeCount, 0);

    // Update tab status
    if (this.elements.tabStatus) {
      if (preservedCount === 0) {
        this.elements.tabStatus.textContent = "No tabs detected";
        this.elements.tabStatus.className = "opacity-75";
      } else {
        this.elements.tabStatus.textContent = `${preservedCount} protected session`;
        this.elements.tabStatus.className = "opacity-75 text-success";
      }
    }

    // Clear browser tab lists and show only protected tabs
    this.clearNonProtectedTabs();

    this.logActivity(
      `🌐 Browser tabs closed: ${closedCount}, preserved: ${preservedCount}`,
      "success",
    );
    setTimeout(() => this.requestUpdate(), 500);
  }

  refreshProcesses() {
    this.logActivity("Refreshing processes...", "info");
    if (this.socket && this.isConnected) {
      this.socket.emit("request_update");
    }
  }

  refreshTabs() {
    this.logActivity("Refreshing browser tabs...", "info");
    if (this.socket && this.isConnected) {
      this.socket.emit("request_update");
    }
  }

  requestUpdate() {
    if (this.socket && this.isConnected) {
      this.socket.emit("request_update");
    }
  }

  startPeriodicUpdates() {
    // Update every 10 seconds
    this.updateInterval = setInterval(() => {
      if (this.isConnected) {
        this.requestUpdate();
      }
    }, 10000);
  }

  showAlert(message, type = "info") {
    const alertDiv = document.createElement("div");
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
            <i class="bi bi-${this.getAlertIcon(type)}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

    this.elements.alertContainer.appendChild(alertDiv);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (alertDiv.parentNode) {
        alertDiv.remove();
      }
    }, 5000);
  }

  getAlertIcon(type) {
    const icons = {
      success: "check-circle-fill",
      danger: "exclamation-triangle-fill",
      warning: "exclamation-triangle-fill",
      info: "info-circle-fill",
    };
    return icons[type] || "info-circle-fill";
  }

  logActivity(message, level = "info") {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement("div");
    logEntry.className = `log-entry ${level}`;

    const icon = this.getLogIcon(level);
    logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <i class="bi bi-${icon} ms-2 me-1"></i>
            ${message}
        `;

    this.elements.activityLog.appendChild(logEntry);
    this.elements.activityLog.scrollTop =
      this.elements.activityLog.scrollHeight;

    // Keep only last 50 entries
    const entries = this.elements.activityLog.children;
    while (entries.length > 50) {
      entries[0].remove();
    }
  }

  getLogIcon(level) {
    const icons = {
      success: "check-circle",
      error: "exclamation-triangle",
      warning: "exclamation-triangle",
      info: "info-circle",
    };
    return icons[level] || "info-circle";
  }

  showLoading(show) {
    if (show) {
      this.elements.loadingOverlay.classList.add("show");
    } else {
      this.elements.loadingOverlay.classList.remove("show");
    }
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  removeAiProcessesFromTable() {
    const tbody = this.elements.processesTable;
    if (!tbody) return;

    // Remove all rows with AI processes (those with is_ai_app class)
    const aiRows = tbody.querySelectorAll(
      'tr.ai-process-row, tr[data-is-ai="true"]',
    );
    aiRows.forEach((row) => {
      row.style.transition = "opacity 0.3s ease, transform 0.3s ease";
      row.style.opacity = "0";
      row.style.transform = "translateX(-100%)";
      setTimeout(() => {
        if (row.parentNode) {
          row.remove();
        }
      }, 300);
    });
  }

  clearNonProtectedTabs() {
    // Clear Chrome tabs (keep only protected ones)
    const chromeList = this.elements.chromeTabList;
    if (chromeList) {
      const protectedTabs = chromeList.querySelectorAll(".protected");
      chromeList.innerHTML = "";
      protectedTabs.forEach((tab) => chromeList.appendChild(tab));

      if (protectedTabs.length === 0) {
        chromeList.innerHTML = `
                    <div class="list-group-item text-muted">
                        <i class="bi bi-info-circle"></i> No Chrome tabs detected
                    </div>
                `;
      }
    }

    // Clear Firefox and Edge tabs completely
    const firefoxList = this.elements.firefoxTabList;
    const edgeList = this.elements.edgeTabList;

    if (firefoxList) {
      firefoxList.innerHTML = `
                <div class="list-group-item text-muted">
                    <i class="bi bi-info-circle"></i> No Firefox tabs detected
                </div>
            `;
    }

    if (edgeList) {
      edgeList.innerHTML = `
                <div class="list-group-item text-muted">
                    <i class="bi bi-info-circle"></i> No Edge tabs detected
                </div>
            `;
    }
  }

  animateCounter(element, newValue) {
    if (!element) return;

    const currentValue = parseInt(element.textContent) || 0;
    if (currentValue === newValue) return;

    // Animate counter change
    const duration = 500;
    const steps = 20;
    const stepValue = (newValue - currentValue) / steps;
    const stepDuration = duration / steps;

    let currentStep = 0;
    const timer = setInterval(() => {
      currentStep++;
      const value = Math.round(currentValue + stepValue * currentStep);
      element.textContent = currentStep === steps ? newValue : value;

      if (currentStep >= steps) {
        clearInterval(timer);

        // Add visual effect for significant changes
        if (Math.abs(newValue - currentValue) > 0) {
          element.style.transform = "scale(1.2)";
          element.style.transition = "transform 0.2s ease";
          setTimeout(() => {
            element.style.transform = "scale(1)";
          }, 200);
        }
      }
    }, stepDuration);
  }

  destroy() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    if (this.socket) {
      this.socket.disconnect();
    }
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.exhonDashboard = new ExHonDashboard();
});

// Cleanup on page unload
window.addEventListener("beforeunload", () => {
  if (window.exhonDashboard) {
    window.exhonDashboard.destroy();
  }
});
