// geolead_content.js
(function () {
  const EXT_LOG_PREFIX = "[GeoLead Bridge CT]";

  // Mark presence so the page can detect the content script.
  try {
    window.__geoleadBridgeActive = true;
  } catch (e) {
    // ignore
  }

  function log(...args) {
    console.log(EXT_LOG_PREFIX, ...args);
  }

  function showAlert(msg) {
    window.alert(msg);
  }

  // Compare dates in the timezone encoded in played_at
  function isToday(isoString) {
    try {
      const played = new Date(isoString);
      if (Number.isNaN(played.getTime())) return false;

      const now = new Date();

      const playedY = played.getFullYear();
      const playedM = played.getMonth();
      const playedD = played.getDate();

      const nowY = now.getFullYear();
      const nowM = now.getMonth();
      const nowD = now.getDate();

      return playedY === nowY && playedM === nowM && playedD === nowD;
    } catch {
      return false;
    }
  }

  function importFromStorage(boardSlug) {
    log("Import requested for board:", boardSlug);

    chrome.storage.local.get("geoguessrLastGame", async (result) => {
      log("chrome.storage.local.get result:", result);

      const data = result.geoguessrLastGame;
      if (!data) {
        showAlert(
          "GeoLead Bridge:\nNo recent GeoGuessr game found.\nFinish today's daily game on geoguessr.com first."
        );
        return;
      }

      log("Loaded geoguessrLastGame:", data);

      if (!data.played_at) {
        showAlert(
          "GeoLead Bridge:\nCannot detect when you last played.\nPlease finish today's game again and then import."
        );
        return;
      }

      if (!isToday(data.played_at)) {
        showAlert(
          "GeoLead Bridge:\nYour last saved GeoGuessr game is not from today.\nFinish today's daily game and try again."
        );
        return;
      }

      if (!Array.isArray(data.rounds) || data.rounds.length === 0) {
        showAlert(
          "GeoLead Bridge:\nSaved game has no rounds.\nFinish today's daily game and try again."
        );
        return;
      }

      // Build rounds payload
      const rounds = (data.rounds || []).map((r, idx) => {
        const roundPayload = {
          score: typeof r.score === "number" ? r.score : 0,
          distance_m:
            typeof r.distance_m === "number" ? r.distance_m : null,
          guess_lat:
            r.guess && typeof r.guess.lat === "number" ? r.guess.lat : null,
          guess_lng:
            r.guess && typeof r.guess.lng === "number" ? r.guess.lng : null,
          target_lat:
            r.correct && typeof r.correct.lat === "number"
              ? r.correct.lat
              : null,
          target_lng:
            r.correct && typeof r.correct.lng === "number"
              ? r.correct.lng
              : null
        };
        log(`Round ${idx + 1} payload:`, roundPayload);
        return roundPayload;
      });

      // Ensure player_name is ALWAYS a non-empty string
      let rawNick =
        data.user && typeof data.user.nick === "string"
          ? data.user.nick
          : "";

      if (typeof rawNick === "string") {
        rawNick = rawNick.trim();
      } else {
        rawNick = String(rawNick || "");
      }

      if (!rawNick) {
        // fallback so Pydantic min_length=1 is satisfied
        rawNick = "GeoLead player";
      }

      const playerName = rawNick;
      log("Using player_name:", playerName);

      const totalDistance = (data.rounds || []).reduce((acc, r) => {
        const d = typeof r.distance_m === "number" ? r.distance_m : 0;
        return acc + d;
      }, 0);

      const body = {
        player_name: playerName,
        board_slug: boardSlug,
        total_score:
          typeof data.total_score === "number"
            ? data.total_score
            : null,
        total_distance_m: totalDistance || null,
        game_id: data.dailyQuizId || data.token || null,
        played_at: data.played_at,
        rounds
      };

      log("Final payload to /api/geoguessr/import:", body);

      // ---- IMPORTANT: build absolute URL so Firefox is happy ----
      let apiUrl;
      try {
        apiUrl = new URL(
          "/api/geoguessr/import",
          window.location.origin
        ).toString();
      } catch (e) {
        console.error(
          EXT_LOG_PREFIX,
          "Failed to build API URL from origin",
          window.location.origin,
          e
        );
        showAlert(
          "GeoLead Bridge:\nCould not build API URL on this page."
        );
        return;
      }

      log("Resolved API URL:", apiUrl);
      // -----------------------------------------------------------

      try {
        const resp = await window.fetch(apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          credentials: "include"
        });

        const text = await resp.text();
        let json = null;
        try {
          json = JSON.parse(text);
        } catch {
          // not JSON, keep as text
        }

        log("Import response status:", resp.status);
        log("Raw import response text:", text);
        log("Parsed import response JSON:", json);

        if (!resp.ok) {
          const detail =
            (json && (json.detail || json.error)) ||
            (typeof text === "string" && text.trim().length
              ? text
              : `HTTP ${resp.status}`);
          showAlert(
            "GeoLead Bridge:\nFailed to import game.\n" + detail
          );
          return;
        }

        const totalScore = json?.total_score ?? body.total_score ?? "n/a";
        const totalDistanceM =
          json?.total_distance_m ?? body.total_distance_m ?? null;

        let totalDistanceKmText = "n/a";
        if (typeof totalDistanceM === "number" && totalDistanceM > 0) {
          totalDistanceKmText = (totalDistanceM / 1000).toFixed(1) + " km";
        }

        showAlert(
          "GeoLead Bridge:\n" +
            "Imported game successfully!\n" +
            "Total score: " +
            totalScore +
            "\n" +
            "Total distance: " +
            totalDistanceKmText
        );

        // Redirect to today's round for this board
        window.location.href = `/board/${encodeURIComponent(
          boardSlug
        )}/today`;
      } catch (err) {
        console.error(EXT_LOG_PREFIX, "Import error:", err);
        showAlert(
          "GeoLead Bridge:\nNetwork error while importing game.\nCheck that GeoLead is running."
        );
      }
    });
  }

  function setupOnSubmitPage() {
    const path = window.location.pathname || "";
    const match = path.match(/^\/board\/([^\/]+)\/submit\/?$/);
    if (!match) {
      return; // not on /board/{slug}/submit
    }
    const boardSlug = match[1];

    const btn = document.getElementById("geolead-import-btn");
    if (!btn) {
      log(
        "Submit page detected, but #geolead-import-btn not found in DOM."
      );
      return;
    }

    btn.addEventListener("click", () => importFromStorage(boardSlug));
    log(
      "Submit page detected for board:",
      boardSlug,
      "Import button wired."
    );
  }

  // Run on page load
  if (
    document.readyState === "complete" ||
    document.readyState === "interactive"
  ) {
    setupOnSubmitPage();
  } else {
    window.addEventListener("DOMContentLoaded", setupOnSubmitPage, {
      once: true
    });
  }
})();
