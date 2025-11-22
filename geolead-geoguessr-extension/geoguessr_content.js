// geoguessr_content.js
(function () {
  const EXT_LOG_PREFIX = "[GeoLead Bridge]";

  function log(...args) {
    console.log(EXT_LOG_PREFIX, ...args);
  }

  // ---------- Read __NEXT_DATA__ ----------

  function getNextData() {
    const script = document.getElementById("__NEXT_DATA__");
    if (!script) {
      log("__NEXT_DATA__ script not found.");
      return null;
    }
    try {
      const raw = script.textContent || script.innerText || "{}";
      return JSON.parse(raw);
    } catch (e) {
      console.warn(EXT_LOG_PREFIX, "Failed to parse __NEXT_DATA__", e);
      return null;
    }
  }

  function extractFromNextData() {
    const data = getNextData();
    if (!data) return null;

    const props = data.props || {};
    const pageProps = props.pageProps || {};
    const accountProps = props.accountProps || {};
    const account = accountProps.account || {};
    const userRaw = account.user || {};

    const user = {
      id: userRaw.userId || userRaw.id || null,
      nick: userRaw.nick || null,
      email: account.email || null,
      countryCode: userRaw.countryCode || null
    };

    const classicGame =
      pageProps.game || (pageProps.data && pageProps.data.game) || null;
    const quizGame = pageProps.quizGame || null;

    if (quizGame) {
      return {
        type: "daily-quiz",
        dailyQuizId: pageProps.dailyQuizId || quizGame.dailyQuizId || null,
        game: quizGame,
        user
      };
    }

    if (classicGame) {
      return {
        type: "classic",
        game: classicGame,
        user
      };
    }

    log("No game object found in __NEXT_DATA__ (no quizGame or game).");
    return null;
  }

  // ---------- Builders ----------

  function buildClassicSummary(game, user) {
    const map = game.map || game.challenge?.map || {};
    const mapImage =
      map.coverUrl || map.thumbnailUrl || map.imageUrl || null;

    const players = game.players || (game.player ? [game.player] : []);
    const myPlayer = players[0] || null;

    const totalScore =
      myPlayer?.totalScore?.amount ??
      myPlayer?.totalScore ??
      myPlayer?.score ??
      null;

    const rounds = game.rounds || [];
    const currentRoundNumber =
      game.currentRoundNumber || game.round || rounds.length;

    const roundsSummary = rounds.slice(0, currentRoundNumber).map((rnd, i) => {
      const guesses = rnd.guesses || [];
      const guess = guesses[0] || rnd.guess || null;

      let distance = null;
      if (typeof guess?.distance === "number") {
        distance = guess.distance;
      } else if (typeof guess?.distanceInMeters === "number") {
        distance = guess.distanceInMeters;
      }

      let score = null;
      if (typeof guess?.roundScoreInPoints === "number") {
        score = guess.roundScoreInPoints;
      } else if (typeof guess?.scoreInPoints === "number") {
        score = guess.scoreInPoints;
      } else if (typeof guess?.score === "number") {
        score = guess.score;
      }

      return {
        round: i + 1,
        distance_m: distance,
        distance_km:
          typeof distance === "number" ? distance / 1000 : null,
        score,
        correct: { lat: rnd.lat, lng: rnd.lng },
        guess: guess
          ? { lat: guess.lat, lng: guess.lng }
          : null
      };
    });

    const playedAt = new Date().toISOString();

    const total_distance_m = roundsSummary.reduce(
      (acc, r) => acc + (typeof r.distance_m === "number" ? r.distance_m : 0),
      0
    );

    return {
      source: "geoguessr",
      mode: "classic",
      token: game.token || null,
      user,
      played_at: playedAt,
      map: {
        id: map.id || null,
        name: map.name || null,
        slug: map.slug || null,
        image: mapImage
      },
      total_score: totalScore,
      total_distance_m,
      rounds: roundsSummary
    };
  }

  // Daily free game (/free) – this is what we care about
  function buildDailyQuizSummary(quizGame, user, dailyQuizId) {
    const guesses = quizGame.guesses || [];
    const rounds = quizGame.rounds || [];

    const guessByRound = {};
    for (const g of guesses) {
      if (!g) continue;
      const rn = g.roundNumber ?? g.round ?? null;
      if (rn == null) continue;
      guessByRound[rn] = g;
    }

    const roundsSummary = rounds.map((r, idx) => {
      const roundNumber = r.roundNumber ?? r.round ?? idx + 1;
      const guess = guessByRound[roundNumber] || null;

      const pano =
        r.question?.panoramaQuestionPayload?.panorama || {};
      const correctLat = pano.lat ?? pano.latitude ?? null;
      const correctLng = pano.lng ?? pano.longitude ?? null;

      let distance = null;
      if (typeof guess?.distance === "number") {
        distance = guess.distance;
      } else if (typeof guess?.distanceInMeters === "number") {
        distance = guess.distanceInMeters;
      }

      let score = null;
      if (typeof guess?.score === "number") {
        score = guess.score;
      } else if (typeof guess?.scoreInPoints === "number") {
        score = guess.scoreInPoints;
      } else if (typeof guess?.roundScoreInPoints === "number") {
        score = guess.roundScoreInPoints;
      }

      return {
        round: roundNumber,
        distance_m: distance,
        distance_km:
          typeof distance === "number" ? distance / 1000 : null,
        score,
        correct:
          typeof correctLat === "number" && typeof correctLng === "number"
            ? { lat: correctLat, lng: correctLng }
            : null,
        guess:
          guess &&
          typeof guess.lat === "number" &&
          typeof guess.lng === "number"
            ? { lat: guess.lat, lng: guess.lng }
            : null
      };
    });

    // When we captured this (used for "today" check)
    const playedAt = new Date().toISOString();

    // Total score: use GeoGuessr field if present, otherwise sum rounds
    let totalScore = null;
    if (typeof quizGame.totalScore === "number") {
      totalScore = quizGame.totalScore;
    } else if (typeof quizGame.score === "number") {
      totalScore = quizGame.score;
    }

    if (totalScore == null) {
      totalScore = roundsSummary.reduce(
        (acc, r) => acc + (typeof r.score === "number" ? r.score : 0),
        0
      );
    }

    const total_distance_m = roundsSummary.reduce(
      (acc, r) => acc + (typeof r.distance_m === "number" ? r.distance_m : 0),
      0
    );

    const totalRounds =
      quizGame.totalRounds || rounds.length || roundsSummary.length || null;

    return {
      source: "geoguessr",
      mode: "daily-quiz",
      dailyQuizId: dailyQuizId || quizGame.dailyQuizId || null,
      quizId: quizGame.quizId || null,
      user,
      played_at: playedAt,
      total_score: totalScore,
      max_score: typeof quizGame.maxScore === "number" ? quizGame.maxScore : null,
      total_rounds: totalRounds,
      total_distance_m,
      rounds: roundsSummary
    };
  }

  function buildSummaryFromNextData() {
    const extracted = extractFromNextData();
    if (!extracted) return null;

    const { type, game, user, dailyQuizId } = extracted;

    if (type === "daily-quiz") {
      return buildDailyQuizSummary(game, user, dailyQuizId);
    }

    if (type === "classic") {
      return buildClassicSummary(game, user);
    }

    return null;
  }

  function storeSummary() {
    const summary = buildSummaryFromNextData();
    if (!summary) return;

    chrome.storage.local.set({ geoguessrLastGame: summary }, () => {
      if (chrome.runtime.lastError) {
        console.warn(
          EXT_LOG_PREFIX,
          "Failed to store game summary:",
          chrome.runtime.lastError
        );
      } else {
        log("Stored latest game summary from __NEXT_DATA__:", summary);
      }
    });
  }

  // ---------- Observe updates ----------

  function setupObserver() {
    const script = document.getElementById("__NEXT_DATA__");
    if (!script) {
      log("__NEXT_DATA__ not found, cannot observe.");
      return;
    }

    let lastSignature = null;

    const update = () => {
      const data = getNextData();
      if (!data) return;

      const props = data.props || {};
      const pageProps = props.pageProps || {};
      const quizGame = pageProps.quizGame || {};
      const classicGame =
        pageProps.game || (pageProps.data && pageProps.data.game) || {};

      const sig =
        (quizGame.quizId || "") +
        "|" +
        (quizGame.currentRound || "") +
        "|" +
        (classicGame.token || "") +
        "|" +
        (classicGame.currentRoundNumber || "");

      if (sig && sig === lastSignature) {
        return;
      }
      lastSignature = sig;

      storeSummary();
    };

    // Initial run
    update();

    const observer = new MutationObserver(() => {
      update();
    });

    observer.observe(script, {
      characterData: true,
      childList: true,
      subtree: true
    });

    log("MutationObserver set on __NEXT_DATA__.");
  }

  if (
    document.readyState === "complete" ||
    document.readyState === "interactive"
  ) {
    setupObserver();
  } else {
    window.addEventListener("DOMContentLoaded", setupObserver, {
      once: true
    });
  }
})();
