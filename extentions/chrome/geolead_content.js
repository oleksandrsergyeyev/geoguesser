// geolead_content.js
(function () {
  const EXT_LOG_PREFIX = "[GeoLead Bridge]";

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
    chrome.storage.local.get("geoguessrLastGame", async (result) => {
      const data = result.geoguessrLastGame;
      if (!data) {
        showAlert(
          "GeoLead Bridge:\nNo recent GeoGuessr game found.\nFinish today's daily game on geoguessr.com first."
        );
        return;
      }

      log("Imported GeoGuessr data from storage:", data);

      // --- date checks (same as before) ---
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

      // ---------- NEW: validate that we actually have non-zero data ----------
      const roundsSource = Array.isArray(data.rounds) ? data.rounds : [];

      // Compute total distance from rounds
      const totalDistanceM = roundsSource.reduce((acc, r) => {
        const d = typeof r.distance_m === "number" ? r.distance_m : 0;
        return acc + d;
      }, 0);

      // Prefer total_score from GeoGuessr, otherwise sum per-round scores
      let totalScore =
        typeof data.total_score === "number" ? data.total_score : 0;

      if (totalScore <= 0) {
        totalScore = roundsSource.reduce((acc, r) => {
          const s = typeof r.score === "number" ? r.score : 0;
          return acc + s;
        }, 0);
      }

      // At least one round must have a positive score OR distance
      const nonZeroRounds = roundsSource.filter((r) => {
        const s =
          typeof r.score === "number" ? r.score : null;
        const d =
          typeof r.distance_m === "number" ? r.distance_m : null;
        return (s !== null && s > 0) || (d !== null && d > 0);
      });

      if (!nonZeroRounds.length || (totalScore <= 0 && totalDistanceM <= 0)) {
        // This is the key fix: we refuse to submit "all zeros" games
        showAlert(
          "GeoLead Bridge:\n" +
            "Could not find a finished GeoGuessr game with non-zero results.\n" +
            "Make sure you have completed today's daily game and are on the final results screen."
        );
        return;
      }

      // ---------- Build payload for GeoLead backend ----------
      const rounds = roundsSource.map((r) => ({
        score: typeof r.score === "number" ? Math.round(r.score) : 0,
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
      }));

      const body = {
        // player_name is ignored server-side; still sent for debugging
        player_name: data.user?.nick || null,
        board_slug: boardSlug,
        total_score: totalScore,
        total_distance_m: totalDistanceM,
        game_id: data.dailyQuizId || data.token || null,
        played_at: data.played_at,
        rounds
      };

      log("Sending payload to GeoLead /api/geoguessr/import:", body);

      try {
        const resp = await fetch("/api/geoguessr/import", {
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
          // ignore parse errors, we'll fall back to body values
        }

        if (!resp.ok) {
          const detail =
            (json && json.detail) || `HTTP ${resp.status}`;
          showAlert(
            `GeoLead Bridge:\nFailed to import game.\n${detail}`
          );
          return;
        }

        const finalScore =
          json?.total_score ?? body.total_score ?? "n/a";
        const finalDistanceM =
          json?.total_distance_m ?? body.total_distance_m ?? 0;

        const finalDistanceKm = (finalDistanceM / 1000).toFixed(1);

        showAlert(
          `GeoLead Bridge:\nImported game successfully!\n` +
            `Total score: ${finalScore}\n` +
            `Total distance: ${finalDistanceKm} km`
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
