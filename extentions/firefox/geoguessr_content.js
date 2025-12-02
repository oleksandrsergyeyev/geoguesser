// geoguessr_content.js
(() => {
  const LOG_PREFIX = "[GeoLead Bridge]";

  function log(...args) {
    console.log(LOG_PREFIX, ...args);
  }

  log(
    "GeoGuessr content script loaded (API watcher runs in background)."
  );

  // Debug helper: from the page console you can run:
  //   window.__geoleadDumpLastGame()
  // to see what is currently stored.
  window.__geoleadDumpLastGame = function () {
    try {
      chrome.storage.local.get("geoguessrLastGame", (res) => {
        log("Dump geoguessrLastGame = ", res.geoguessrLastGame);
      });
    } catch (e) {
      console.error(LOG_PREFIX, "Failed to read storage:", e);
    }
  };
})();
