// Open the side panel when the extension icon is clicked
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Also handle individual tab clicks (fallback for older Chrome versions)
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// NOTE: the service worker must NOT hold the native-messaging port
// (com.vp1591.tts_server). MV3 workers are ephemeral — they get torn down
// after ~30s idle, which would disconnect the port and make the panel
// misreport the server as dead. Server start/stop lives in panel.js.