/* First-load intro: a "make a call" hook plus an optional 4-step model primer, shown
   while stlite boots Python in the browser (D-049; designed from mockups 2026-07-22).

   Structure — three states inside the #intro overlay:
   - minimal cover (static HTML in index.html): instant paint; if the app is ready within
     a short grace window (cached visits), the overlay just fades and the intro never shows;
   - guess hook (screen 1): question + three priors; picking one reveals the three decisive
     dials; a progress ring morphs into the "Open the explorer" button when the app is ready;
   - model primer (screen 2, optional): 4 click-through steps with the core equations.

   Boot progress is real, not simulated: PerformanceObserver resource-timing entries mark
   the completion of the big downloads (stlite bundle, Pyodide runtime + wasm + stdlib,
   package wheels), and the same DOM poll that always gated the old #boot cover — the
   Streamlit sidebar painting — marks "ready". Between milestones the bar eases forward
   asymptotically, so motion is cosmetic but every jump is a real event.

   This file lives at the repo ROOT, outside the src/ tree that deploy/sync.sh rewrites. */

(function () {
  "use strict";

  var GRACE_MS = 2500;       // ready within this → never show the intro (cached visits)
  var STUCK_MS = 150000;     // after this, admit something looks slow

  var intro = document.getElementById("intro");
  var t0 = performance.now();
  var ready = false;
  var state = "min";         // min | hook | primer  (post-skip goes back to "min")
  var frac = 0, target = 0.06;
  var wheelsDone = 0;

  /* ---------------- boot milestones (real network events) ---------------- */

  var STAGES = [
    "Fetching the app",
    "Downloading the Python runtime (~15 MB)",
    "Starting the Python interpreter",
    "Installing packages — numpy, plotly, streamlit",
    "Launching the explorer",
    "Ready.",
  ];
  var stageIdx = 0;
  var stage = STAGES[0];

  // Monotonic: late resource fetches (stlite pulls extra chunks after boot) must never
  // move the label backwards, and nothing changes once the app is ready.
  function bump(t, idx) {
    if (ready) return;
    if (t > target) target = t;
    if (idx > stageIdx && idx < STAGES.length) { stageIdx = idx; stage = STAGES[idx]; }
  }
  try {
    new PerformanceObserver(function (list) {
      list.getEntries().forEach(function (e) {
        var n = e.name;
        if (/@stlite\/browser.*\.js/.test(n)) bump(0.12, 1);
        else if (/pyodide\.m?js(\?|$)/.test(n)) bump(0.18, 1);
        else if (/python_stdlib\.zip/.test(n)) bump(0.32, 1);
        else if (/pyodide\.asm\.wasm/.test(n)) bump(0.48, 2);
        else if (/\.whl(\?|$)/.test(n)) {
          wheelsDone += 1;
          bump(Math.min(0.9, 0.48 + 0.06 * wheelsDone), 3);
        }
      });
    }).observe({ type: "resource", buffered: true });
  } catch (e) { /* no resource timing → the time floor below still moves the bar */ }

  setInterval(function () {
    var el = (performance.now() - t0) / 1000;
    // time floor (belt & braces) + asymptotic easing toward the last real milestone
    if (!ready) target = Math.max(target, Math.min(0.9, el / 70));
    if (target > 0.85) bump(target, 4);
    frac += (Math.min(target, 0.97) - frac) * 0.06;
    if (ready) frac = 1;
    render();
    if (!ready && el * 1000 > STUCK_MS)
      stage = "Still working — a slow connection can take a few minutes";
  }, 150);

  /* ---------------- ready detection (same signal as the old #boot cover) ---------------- */

  var poll = setInterval(function () {
    var app = document.querySelector('[data-testid="stApp"], [data-testid="stAppViewContainer"]');
    if (app && app.querySelector('[data-testid="stSidebar"]')) {
      clearInterval(poll);
      ready = true;
      stage = "Ready.";
      console.log("boot-to-first-paint: " + ((performance.now() - t0) / 1000).toFixed(1) + "s");
      // grace window or post-skip: straight in — unless the phone is held in portrait,
      // where the overlay stays up asking for landscape (it auto-dismisses on rotation)
      if (state === "min" && !phonePortrait()) dismiss();
      render();
    }
  }, 300);

  var gone = false;
  function dismiss() {
    if (gone) return;
    gone = true;
    intro.classList.add("i-gone");
    setTimeout(function () { intro.remove(); }, 700);
  }

  /* ---------------- phone-portrait gate: the explorer needs landscape ----------------
     A portrait PHONE (smaller viewport dimension < 620px, matching the app's own phone
     breakpoint) keeps the intro up with a "rotate to landscape" banner; the moment the
     device rotates to landscape (and the app is ready) the overlay closes by itself. */
  function phonePortrait() {
    return window.innerHeight > window.innerWidth &&
           Math.min(window.innerWidth, window.innerHeight) < 620;
  }
  function syncRotate() {
    if (gone) return;
    var pp = phonePortrait();
    var b = intro.querySelector(".i-rotatebar");
    if (pp && !b) {
      b = document.createElement("div");
      b.className = "i-rotatebar";
      b.innerHTML = "📱 Rotate your phone to <b>landscape</b> to start — " +
                    "the explorer opens automatically when you do.";
      intro.prepend(b);
    } else if (!pp && b) { b.remove(); }
    if (ready && !pp) dismiss();     // rotated to landscape → straight into the app
    render();
  }
  window.addEventListener("resize", syncRotate);
  window.addEventListener("orientationchange", syncRotate);
  syncRotate();

  /* ---------------- grace window: cached loads never see the intro ---------------- */

  setTimeout(function () {
    if (!ready) { state = "hook"; intro.innerHTML = HOOK_HTML + PRIMER_HTML; wire(); syncRotate(); render(); }
  }, GRACE_MS);

  /* ---------------- markup ---------------- */

  var RING_CIRC = (2 * Math.PI * 52).toFixed(1);

  var HOOK_HTML =
    '<div class="i-hook">' +
      '<div class="i-kicker">While the explorer boots in your browser</div>' +
      '<h1>Frontier AI labs lose billions every year.<br>Will the leading edge ever pay?</h1>' +
      '<p class="i-sub">Frontier labs spend tens of billions a year training ever-better models. ' +
        'Whether the frontier ever pays hinges on a tension: the physical buildout that drives it — ' +
        'power, fabs, capital — must eventually slow, even as recursive self-improvement could ' +
        'accelerate algorithmic progress.</p>' +
      '<div class="i-dials i-on">' +
        '<h2>The model says: it hinges on three dials</h2>' +
        '<div class="i-dial"><div class="i-num">1</div><div>' +
          '<h3>How fast the compute race decelerates</h3>' +
          "<p>Today's ~4×/yr compute scaling cannot persist — power and fabs bind. The training " +
          'bill (paid in advance, for the <i>next</i> model) stops exploding only when scaling ' +
          'slows.</p></div></div>' +
        '<div class="i-dial"><div class="i-num">2</div><div>' +
          '<h3>Where the value of capability saturates</h3>' +
          "<p>Each capability leap multiplies what customers will pay — until it doesn't. An " +
          'early bend caps the rent forever; a late bend lets the harvest run for a decade.' +
          '</p></div></div>' +
        '<div class="i-dial"><div class="i-num">3</div><div>' +
          '<h3>How fast followers close the gap</h3>' +
          "<p>Distillation from served models plus ambient diffusion of know-how sets the " +
          "durable lead — and it's the dial a <b>release delay</b> could throttle, at the price " +
          'of revenue today.</p></div></div>' +
      '</div>' +
      '<div><button class="i-primerlink">How does the model think? — a 60-second tour →</button></div>' +
      '<div class="i-ringwrap">' +
        '<svg width="120" height="120" viewBox="0 0 120 120">' +
          '<circle class="i-ring-bg" cx="60" cy="60" r="52" stroke-width="5"/>' +
          '<circle class="i-ring-fg" cx="60" cy="60" r="52" stroke-width="5" ' +
            'stroke-dasharray="' + RING_CIRC + '" stroke-dashoffset="' + RING_CIRC + '"/>' +
        '</svg>' +
        '<button class="i-ringbtn">booting…<br><span class="i-ringpct">0%</span></button>' +
      '</div>' +
      '<div class="i-stage"></div>' +
      '<div class="i-privacy">Runs entirely in your browser (Python via WebAssembly) — ' +
        'nothing is sent to a server. First load only; visits after this are fast.</div>' +
    '</div>';

  var STEPS = [
    { h: "Two players, one race",
      body:
        '<p>A <span class="i-lead">leader</span> (the frontier labs, taken together) pushes AI ' +
        'capability upward, powered by two engines: <b>more compute</b> and <b>better ' +
        'algorithms</b>. A <span class="i-foll">follower</span> (open-source / fast rivals) ' +
        'trails behind. Capability is measured in orders of magnitude — one unit means ' +
        '&ldquo;10× the effective scale.&rdquo;</p>' +
        '<svg class="i-diagram" viewBox="0 0 560 130">' +
          '<line x1="30" y1="108" x2="540" y2="108" class="i-ax"/>' +
          '<path d="M30,100 C170,84 350,52 540,14" class="i-ln-l"/>' +
          '<path d="M30,112 C170,102 350,76 540,42" class="i-ln-f"/>' +
          '<text x="490" y="12" class="i-lt">leader xᴸ</text>' +
          '<text x="483" y="60" class="i-ft">follower xᶠ</text>' +
          '<text x="30" y="125">time →</text></svg>' },
    { h: "The gap wants to close",
      body:
        "<p>The follower doesn't just push on its own — it <b>learns from the leader</b>: " +
        'published methods, migrating talent, and above all <b>distillation</b> from the models ' +
        'the leader serves. The wider the gap Δ, the faster it closes. Today’s observed ' +
        'lag: about <b>8 months</b> — and that number is doing a lot of work.</p>' +
        '<div class="i-eq">Δ̇ &nbsp;=&nbsp; ẋᴸ − δ·Δ' +
          ' &nbsp;&nbsp;<span class="i-plain">(the lead grows with the leader’s speed, ' +
          'shrinks at catch-up rate δ)</span></div>' +
        '<svg class="i-diagram" viewBox="0 0 560 130">' +
          '<line x1="30" y1="108" x2="540" y2="108" class="i-ax"/>' +
          '<path d="M30,100 C170,84 350,52 540,14" class="i-ln-l"/>' +
          '<path d="M30,112 C170,98 350,62 540,26" class="i-ln-f"/>' +
          '<line x1="450" y1="27" x2="450" y2="47" stroke="currentColor" stroke-width="1.2"/>' +
          '<text x="458" y="42">gap Δ shrinking</text></svg>' },
    { h: "Where the money is",
      body:
        '<p>The leader earns a <b>rent on its lead</b>: a margin θ times the extra value its ' +
        "model offers over the follower's. Against that stands the training bill — paid <b>in " +
        'advance</b>, for the <i>next, bigger</i> model. That one timing detail is why a lab can ' +
        'lose billions while every model it has shipped roughly paid for itself.</p>' +
        '<div class="i-eq">revenue = θ·[W(xᴸ) − W(xᶠ)]' +
          ' &nbsp;−&nbsp; cost of the <i>next</i> model</div>' +
        '<svg class="i-diagram" viewBox="0 0 560 130">' +
          '<line x1="30" y1="108" x2="540" y2="108" class="i-ax"/>' +
          '<path d="M30,100 C170,84 350,52 540,14 L540,42 C350,76 170,102 30,112 Z" class="i-gapfill"/>' +
          '<path d="M30,100 C170,84 350,52 540,14" class="i-ln-l"/>' +
          '<path d="M30,112 C170,102 350,76 540,42" class="i-ln-f"/>' +
          '<text x="290" y="66" class="i-at">the leader’s rent lives in this wedge</text></svg>' },
    { h: "What you can do here",
      body:
        '<p>The explorer builds the model in <b>six levels</b> — each adds one mechanism, so you ' +
        'always know which assumption drives what:</p>' +
        '<div class="i-ladder"><span class="i-rung"><b>1</b> Basics</span>' +
          '<span class="i-rung"><b>2</b> Training in advance</span>' +
          '<span class="i-rung"><b>3</b> Growth engine</span>' +
          '<span class="i-rung"><b>4</b> Compute slowdown</span>' +
          '<span class="i-rung"><b>5</b> Value saturation</span>' +
          '<span class="i-rung"><b>6</b> Catch-up channels</span></div>' +
        '<p>Every parameter has a calibrated range with sources; <b>Monte-Carlo mode</b> draws ' +
        'them jointly to turn your assumptions into forecast distributions for the two ' +
        'questions: <b>is the frontier profitable?</b> and <b>does delaying releases pay?</b></p>' +
        '<p class="i-small">The app runs entirely in your browser — a full Python stack, ' +
        'nothing sent to a server.</p>' },
  ];

  var PRIMER_HTML =
    '<div class="i-primer">' +
      '<div class="i-topnav">' +
        '<button class="i-backhook">← Back to the question</button>' +
        '<button class="i-skip">Skip intro</button>' +
      '</div>' +
      '<div class="i-card">' +
        STEPS.map(function (s, i) {
          return '<div class="i-step' + (i === 0 ? " i-on" : "") + '">' +
            '<div class="i-kicker">A 60-second tour of the model · step ' + (i + 1) +
            ' of ' + STEPS.length + '</div><h2>' + s.h + '</h2>' + s.body + '</div>';
        }).join("") +
        '<div class="i-stepnav">' +
          '<button class="i-navbtn i-back" disabled>← Back</button>' +
          '<span class="i-crumbs">1 / ' + STEPS.length + '</span>' +
          '<button class="i-navbtn i-next">Next →</button>' +
        '</div>' +
      '</div>' +
      '<div class="i-footbar">' +
        '<div class="i-fstage"><span class="i-fstagetext"></span><span class="i-fpct"></span></div>' +
        '<div class="i-ftrack"><div class="i-ffill"></div></div>' +
      '</div>' +
    '</div>';

  /* ---------------- behavior ---------------- */

  var step = 0;
  function $(sel) { return intro.querySelector(sel); }

  function wire() {
    $(".i-primerlink").addEventListener("click", function () {
      state = "primer"; intro.classList.add("i-show-primer"); render();
    });
    $(".i-backhook").addEventListener("click", function () {
      state = "hook"; intro.classList.remove("i-show-primer"); render();
    });
    $(".i-skip").addEventListener("click", function () {
      if (ready) { dismiss(); return; }
      state = "min";
      intro.classList.remove("i-show-primer");
      intro.innerHTML =
        '<div class="i-min"><div class="i-spin"></div><div>Loading the explorer…</div>' +
        '<div class="i-minbar"><i></i></div>' +
        '<small>It opens by itself the moment Python finishes booting.</small></div>';
      syncRotate();
    });
    $(".i-ringbtn").addEventListener("click", function () { if (ready && !phonePortrait()) dismiss(); });
    $(".i-back").addEventListener("click", function () { if (step > 0) { step--; render(); } });
    $(".i-next").addEventListener("click", function () {
      if (step < STEPS.length - 1) { step++; render(); }
      else if (ready && !phonePortrait()) dismiss();
    });
  }

  function render() {
    if (state === "min") {
      var mb = $(".i-minbar i");
      if (mb) mb.style.width = (frac * 100).toFixed(0) + "%";
      return;
    }
    var pct = (frac * 100).toFixed(0);
    var ring = $(".i-ring-fg");
    if (ring) ring.style.strokeDashoffset = (RING_CIRC * (1 - frac)).toFixed(1);
    var wrap = $(".i-ringwrap"), btn = $(".i-ringbtn");
    if (ready && wrap) {
      wrap.classList.add("i-done");
      btn.innerHTML = phonePortrait() ? "rotate to<br>landscape 📱" : "Open the<br>explorer →";
    } else if (!ready && btn) {
      btn.innerHTML = "booting…<br><span class='i-ringpct'>" + pct + "%</span>";
    }
    var st = $(".i-stage");
    if (st) st.textContent = stage;
    // primer screen
    intro.querySelectorAll(".i-step").forEach(function (s, i) {
      s.classList.toggle("i-on", i === step);
    });
    var crumbs = $(".i-crumbs");
    if (crumbs) crumbs.textContent = (step + 1) + " / " + STEPS.length;
    var back = $(".i-back"), next = $(".i-next");
    if (back) back.disabled = step === 0;
    if (next) {
      if (step === STEPS.length - 1) {
        next.textContent = !ready ? "Finishing boot… " + pct + "%"
          : (phonePortrait() ? "Rotate to landscape 📱" : "Open the explorer →");
        next.classList.toggle("i-primary", ready && !phonePortrait());
      } else {
        next.textContent = "Next →";
        next.classList.remove("i-primary");
      }
    }
    var fs = $(".i-fstagetext"), fp = $(".i-fpct"), ff = $(".i-ffill");
    if (fs) { fs.textContent = stage; fp.textContent = pct + "%"; ff.style.width = pct + "%"; }
  }
})();
