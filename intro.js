/* ============================================================================
   intro.js — LAPTOP intro-tour overlay controller (D-058).

   Shows the shared tour deck (deploy/tour.js) full-screen ON TOP of the widget
   while it boots in the browser, with the widget blurred/shaded behind. The
   tour is optional, never a gate:

     - First visit  → the FULL deck (localStorage flag flp_tour_seen unset).
       Return visit → a MINIMAL loading card (the cover slide only).
     - The bottom button is the loading→ready signal: a muted, non-clickable
       "Preparing the widget…" while Pyodide boots; when the widget's first real
       paint is detected (same [data-testid="stSidebar"] hook the old boot cover
       used — works on stlite 1.57 web / 1.59 local), it TRANSFORMS into a blue
       "Go to the widget →" with a green dot and a single gentle pulse.
     - Leaving is always the user's choice: an always-present × closes it
       immediately (even mid-load); the ready button enters; and click-anywhere
       on the backdrop dismisses ONLY AFTER ready (before ready, backdrop clicks
       do nothing, so nobody bounces into a half-loaded app).

   The overlay NEVER auto-dismisses. The widget title's (i) button (injected by
   ui/theme.py) reopens the full deck via window.__flpOpenTour.

   This file lives at the repo ROOT, outside the src/ tree deploy/sync.sh
   rewrites. Pairs with intro.css (overlay chrome) + tour.css/tour.js (deck). */

(function () {
  "use strict";

  var SEEN_KEY = "flp_tour_seen";
  var root = document.documentElement;
  var ready = false;
  var gone = false;

  function seen() { try { return localStorage.getItem(SEEN_KEY) === "1"; } catch (e) { return false; } }
  function markSeen() { try { localStorage.setItem(SEEN_KEY, "1"); } catch (e) {} }

  /* ---------------- build / rebuild the overlay ---------------- */

  function buildOverlay(mode) {          // mode: "full" | "min"
    if (!window.FLPTour) return null;    // shared deck engine must be loaded first
    var el = document.getElementById("intro");
    if (!el) { el = document.createElement("div"); el.id = "intro"; document.body.appendChild(el); }
    gone = false;
    el.className = "tour-overlay";
    el.setAttribute("data-mode", mode);
    el.setAttribute("data-ready", ready ? "1" : "0");
    // one contained card holds the deck body + boot bar; the surrounding
    // shaded field is the backdrop (dismiss target once ready)
    el.innerHTML =
      '<div class="tour-card">' +
        '<button class="tour-x" type="button" aria-label="Close the introduction" title="Close">×</button>' +
        '<div class="tour-stage"><div class="tour-deck"></div></div>' +
        '<div class="tour-cta">' +
          '<button class="tour-go" type="button"></button>' +
          '<div class="tour-sub"></div>' +
        '</div>' +
      '</div>';
    root.setAttribute("data-tour-open", "1");

    // mount the shared deck: full = env-filtered slides; min = the cover only
    var host = el.querySelector(".tour-deck");
    var slides = null;
    if (mode === "min") {
      var cover = {};
      for (var k in window.FLPTour.SLIDES[0]) cover[k] = window.FLPTour.SLIDES[0][k];
      delete cover.hint;                 // a single card needs no "step through" nudge
      slides = [cover];
    }
    window.FLPTour.mountDeck(host, { env: "laptop", slides: slides });

    wire(el);
    renderButton(el);
    return el;
  }

  function wire(el) {
    el.querySelector(".tour-x").addEventListener("click", function (e) {
      e.stopPropagation(); leave();
    });
    el.querySelector(".tour-go").addEventListener("click", function (e) {
      e.stopPropagation();
      if (ready) leave();                // muted / non-clickable until ready
    });
    // click-anywhere-on-the-backdrop → dismiss, but ONLY after ready, and never
    // when the click lands inside the card (deck, boot bar, or a control)
    el.addEventListener("click", function (e) {
      if (!ready) return;
      if (e.target.closest(".tour-card")) return;
      leave();
    });
  }

  function renderButton(el) {
    el = el || document.getElementById("intro");
    if (!el) return;
    el.setAttribute("data-ready", ready ? "1" : "0");
    var go = el.querySelector(".tour-go"), sub = el.querySelector(".tour-sub");
    if (!go) return;
    if (ready) {
      go.innerHTML = '<span class="tour-dot" aria-hidden="true"></span>Go to the widget &#8594;';
      if (sub) sub.textContent = "Ready — take your time.";
    } else {
      // narrow card: keep the button label short, move the reassurance to the sub
      go.innerHTML = '<span class="tour-spin" aria-hidden="true"></span>Preparing the widget…';
      if (sub) sub.textContent = "~30–60 s · runs entirely in your browser";
    }
  }

  /* ---------------- leaving (always the user's choice) ---------------- */

  function leave() {
    var el = document.getElementById("intro");
    if (!el || gone) return;
    gone = true;
    markSeen();                          // first leave sets the return-visit flag
    el.classList.add("i-gone");
    root.removeAttribute("data-tour-open");
    setTimeout(function () { if (el && el.parentNode) el.parentNode.removeChild(el); }, 320);
  }

  /* ---------------- ready detection (same signal as the old boot cover) ----------------
     Streamlit's first real paint = the app container plus a sidebar exist. The
     testids are stable across stlite 1.57 (web) and 1.59 (local); both container
     testids are accepted for safety. */
  var poll = setInterval(function () {
    var app = document.querySelector('[data-testid="stApp"], [data-testid="stAppViewContainer"]');
    if (app && app.querySelector('[data-testid="stSidebar"]')) {
      clearInterval(poll);
      ready = true;
      renderButton();                    // flip the button (blue + pulse) in place; never auto-dismiss
    }
  }, 300);

  /* ---------------- (i) reopen hook (called from ui/theme.py's title-bar JS) ----------------
     Reopens the FULL slideable deck with the (now loaded) widget blurred behind. Must work from
     ANY state and be re-openable repeatedly (D-071). The old guard bailed whenever ANY #intro
     existed — so a lingering/stale overlay (e.g. the inline boot-fallback #intro if FLPTour was
     slow to load) turned the (i) into a permanent no-op. Now: if a LIVE full deck is already up,
     no-op; otherwise tear down whatever's there and mount the full deck fresh. */
  window.__flpOpenTour = function () {
    var existing = document.getElementById("intro");
    if (existing) {
      var liveFull = existing.getAttribute("data-mode") === "full" &&
        existing.classList.contains("tour-overlay") && !existing.classList.contains("i-gone");
      if (liveFull) { var tr = existing.querySelector(".track"); if (tr) try { tr.focus(); } catch (e) {} return; }
      if (existing.parentNode) existing.parentNode.removeChild(existing);   // drop stale/min/fading overlay
      gone = true;                                                          // let buildOverlay reset it
    }
    var el = buildOverlay("full");
    if (el) { var t = el.querySelector(".track"); if (t) try { t.focus(); } catch (e) {} }
  };

  // Escape mirrors the × (an explicit user choice, even mid-load)
  window.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && document.getElementById("intro")) leave();
  });

  /* arrow keys drive the deck from anywhere in the overlay, not only when the track is focused
     (mirrors the phone page). Reuses the on-screen prev/next controls so nav stays single-sourced. */
  window.addEventListener("keydown", function (e) {
    var el = document.getElementById("intro");
    if (!el || el.classList.contains("i-gone")) return;
    if (e.target && e.target.closest && e.target.closest(".track")) return;   // deck handles its own focused keys
    if (e.key === "ArrowRight") { var n = el.querySelector(".next"); if (n) { e.preventDefault(); n.click(); } }
    else if (e.key === "ArrowLeft") { var p = el.querySelector(".prev"); if (p) { e.preventDefault(); p.click(); } }
  });

  /* ---------------- first paint ----------------
     Always the FULL slideable deck — first visit AND return. The old return-visit "minimal" 1-slide
     card (gated on flp_tour_seen) read as "just one page you can't slide over" and, because the (i)
     lives BEHIND the overlay, left returning visitors with no path to the whole intro (D-071). The
     deck is optional either way — the × / ready-button / backdrop dismiss it in one click. The "min"
     mode remains available in buildOverlay() but is no longer auto-shown. */
  buildOverlay("full");
})();
