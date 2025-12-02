// background.js
const LOG_PREFIX = "[GeoLead Bridge BG]";

function log(...args) {
  console.log(LOG_PREFIX, ...args);
}

log("Service worker loaded.");

// We watch all daily-quiz API requests from GeoGuessr.
const DAILY_QUIZ_URL_PATTERN =
  "https://www.geoguessr.com/api/v4/daily-quiz/*";

// Used to detect our own fetch() calls and avoid infinite loops
const EXTENSION_ORIGIN = `chrome-extension://${chrome.runtime.id}`;

// Listen when a daily-quiz request finishes
chrome.webRequest.onCompleted.addListener(
  (details) => {
    log("onCompleted for", details.url, {
      statusCode: details.statusCode,
      method: details.method,
      tabId: details.tabId,
      initiator: details.initiator
    });

    // 1) Ignore our own background fetch() calls to avoid recursion
    if (
      details.tabId === -1 || // background / extension context
      (details.initiator &&
        details.initiator.startsWith(EXTENSION_ORIGIN))
    ) {
      log("Ignored request (extension-initiated).");
      return;
    }

    // 2) Only handle successful calls
    if (details.statusCode !== 200) {
      log("Ignored request (status not 200).");
      return;
    }

    // 3) GeoGuessr uses PUT/POST for these; treat them like GET
    const ALLOWED_METHODS = ["GET", "PUT", "POST"];
    if (!ALLOWED_METHODS.includes(details.method)) {
      log("Ignored request (method not GET/PUT/POST).");
      return;
    }

    // Fetch and parse the snapshot ourselves
    fetchAndStoreDailyQuiz(details.url);
  },
  { urls: [DAILY_QUIZ_URL_PATTERN] }
);

/**
 * Fetch the daily-quiz snapshot from GeoGuessr and store a compact
 * game summary in chrome.storage.local.geoguessrLastGame
 */
async function fetchAndStoreDailyQuiz(url) {
  log("Fetching daily-quiz snapshot from", url);

  try {
    const resp = await fetch(url, {
      credentials: "include"
    });

    log("Fetch response status:", resp.status);

    if (!resp.ok) {
      log("Fetch not OK – aborting.");
      return;
    }

    const data = await resp.json();
    log("Received daily-quiz JSON. Root keys:", Object.keys(data || {}));

    const snapshot = data && (data.snapshot || data);
    if (
      !snapshot ||
      !Array.isArray(snapshot.rounds) ||
      !Array.isArray(snapshot.guesses)
    ) {
      log("Snapshot missing rounds/guesses – not storing.", snapshot);
      return;
    }

    const game = buildGameFromSnapshot(snapshot);

    log("Parsed game summary:", {
      quizId: game.dailyQuizId,
      total_score: game.total_score,
      total_distance_m: game.total_distance_m,
      rounds: game.rounds.length,
      played_at: game.played_at
    });

    chrome.storage.local.set({ geoguessrLastGame: game }, () => {
      if (chrome.runtime.lastError) {
        console.error(
          LOG_PREFIX,
          "chrome.storage.local.set error:",
          chrome.runtime.lastError
        );
      } else {
        log("Stored geoguessrLastGame in chrome.storage.local.");
      }
    });
  } catch (err) {
    console.error(LOG_PREFIX, "fetchAndStoreDailyQuiz failed:", err);
  }
}

/**
 * Turn the GeoGuessr snapshot into the format our GeoLead side expects.
 * Uses your sample JSON structure:
 *  - snapshot.rounds[] for correct coords
 *  - snapshot.guesses[] for scores/distances/guesses
 */
function buildGameFromSnapshot(snapshot) {
  const guessesByRound = new Map();

  for (const g of snapshot.guesses || []) {
    if (g && typeof g.roundNumber === "number") {
      guessesByRound.set(g.roundNumber, g);
    }
  }

  const rounds = [];

  for (const r of snapshot.rounds || []) {
    const rn = r.roundNumber;
    const guess = guessesByRound.get(rn) || {};

    const pano = r?.question?.panoramaQuestionPayload?.panorama || {};

    const distance =
      typeof guess.distance === "number" ? guess.distance : null;
    const score = typeof guess.score === "number" ? guess.score : null;

    const roundSummary = {
      round_number: rn,
      score,
      distance_m: distance,
      guess:
        typeof guess.lat === "number" && typeof guess.lng === "number"
          ? { lat: guess.lat, lng: guess.lng }
          : null,
    correct:
        typeof pano.lat === "number" && typeof pano.lng === "number"
          ? { lat: pano.lat, lng: pano.lng }
          : null
    };

    rounds.push(roundSummary);
  }

  const total_distance_m = rounds.reduce((sum, r) => {
    return sum + (typeof r.distance_m === "number" ? r.distance_m : 0);
  }, 0);

  return {
    source: "geoguessr",
    mode: "daily-quiz-api",
    dailyQuizId: snapshot.quizId || null,
    played_at:
      snapshot.createdAt || snapshot.rounds?.[0]?.startAt || null,
    total_score:
      typeof snapshot.totalScore === "number"
        ? snapshot.totalScore
        : null,
    max_score:
      typeof snapshot.maxScore === "number"
        ? snapshot.maxScore
        : null,
    total_rounds: snapshot.totalRounds || rounds.length,
    total_distance_m,
    rounds
  };
}

// Optional debug helper so we can inspect storage from a content script
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "geolead-debug-read-last-game") {
    chrome.storage.local.get("geoguessrLastGame", (res) => {
      sendResponse({ game: res.geoguessrLastGame || null });
    });
    return true; // async
  }
});
