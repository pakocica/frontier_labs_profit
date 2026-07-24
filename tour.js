/* ============================================================================
   tour.js — SHARED tour/guide content + deck engine (D-058).

   The single source of truth for the introduction's SLIDES and the swipeable
   deck that renders them. Loaded by BOTH hosts:
     - mobile.html         → the phone page (deck inside a phone bezel)
     - index.html/intro.js → the laptop intro overlay (deck full-screen, on top
                              of the booting widget)
   Edit a slide once here and both stay in sync.

   Exposes a single global: window.FLPTour = { SLIDES, slideHTML, mountDeck }.
   CSP-safe: no eval, no external deps (hand-set math markup; will move to
   KaTeX/MathML for the final build).

   Per-host differences are data-driven, not forked:
     - a slide may set  only:"phone" | only:"laptop"  to appear on one host;
     - cover `hint` and outro `foot` swap wording by env (see slideHTML).
   ========================================================================== */
window.FLPTour = (function () {
  "use strict";

  // ---- hand-set math helpers (CSP-safe) ----
  var V = function (s) { return "<i>" + s + "</i>"; };
  var S = function (s) { return "<sup>" + s + "</sup>"; };
  var b = function (s) { return "<sub>" + s + "</sub>"; };
  var o = function (s) { return '<span class="mo">' + s + "</span>"; };
  var dot = '<span class="mo dot">&middot;</span>';
  var xL = V("x") + S("L"), xF = V("x") + S("F");
  var line = function (s) { return '<span class="line">' + s + "</span>"; };

  // ---- content model (shared) ----
  var SLIDES = [
    { kind: "cover",
      eyebrow: "An interactive model",
      title: "Will frontier<br>AI labs be<br>profitable?",
      gloss: "A race between a <b>leader</b> &#8212; the top labs, charging a premium for being ahead &#8212; and a <b>follower</b>: open-weight models sold at cost. When is the lead worth more than the bill for keeping it?",
      hint: true },

    { kind: "notice", only: "phone",
      eyebrow: "A note on your phone",
      title: "The explorer needs a bigger screen",
      gloss: "The full widget &#8212; sliders, live graphs, layered models &#8212; is built for a laptop. On a phone, here is a short guided introduction to the ideas and the numbers behind it.",
      foot: "Open on a laptop for the interactive explorer." },

    { kind: "features",
      eyebrow: "What the explorer does",
      title: "Build the model up, one mechanism at a time",
      list: [
        ["Levels", "Start from the bare model and <b>add one mechanism per level</b> &mdash; training-in-advance, AI-accelerated R&amp;D, the compute slowdown, value saturation&hellip;"],
        ["Calibrate", "Move <b>observables you can argue about</b> &mdash; compute scaling, follower lag, margins &mdash; and watch the implied parameters update live."],
        ["See", "<b>Profit</b> over time, <b>revenue vs cost</b>, and the <b>capability gap</b> between leader and follower."]
      ] },

    { kind: "thesis",
      eyebrow: "The basic model",
      title: "A lead worth paying for",
      gloss: "The leader pushes the capability frontier; the follower trails a few months behind and gives the same thing away at cost. The leader earns a premium for that gap &mdash; and pays more every year to keep extending it. Whether it profits is a race between those two forces, and the next slides build that race from the ground up.",
      foot: "Capability is measured in orders of magnitude (OOM) above today's frontier &mdash; today is zero." },

    { kind: "eq", eyebrow: "01 &middot; The race",
      title: "Capability climbs at a steady pace",
      eq: [ line(V("c&#775;") + S("L") + o("=") + V("g") + b("c") + '<span class="amp">and</span>' + V("a&#775;") + S("L") + o("=") + V("g") + b("a")) ],
      gloss: "The leader's capability rises at a fixed pace each year &mdash; part more compute, part better algorithms. Together the frontier advances &asymp; <b>1.07 OOM/yr</b>, roughly a 12&times; yearly gain in effective compute.",
      params: [
        [V("g") + b("c"), "0.623 OOM/yr", "compute &times;4.2 / yr &mdash; Epoch, several series agree"],
        [V("g") + b("a"), "0.447 OOM/yr", "algorithmic efficiency &times;2.8 / yr, two anchors"]
      ] },

    { kind: "eq", eyebrow: "02 &middot; The gap",
      title: "The follower catches up",
      eq: [ line(V("x&#775;") + S("F") + o("=") + V("&delta;") + "(" + xL + o("&minus;") + xF + ")" + o("=") + V("&delta;") + V("&Delta;")) ],
      gloss: "The follower has <b>no engine of its own</b> &mdash; pure catch-up. It closes the gap &Delta; at rate &delta;; the further behind, the faster it gains. Calibrated so the gap holds <b>steady</b> at &Delta;&#8320;.",
      params: [
        [V("&Delta;") + b("0"), "0.70 OOM", "&asymp; 8-month lag behind the frontier (METR, private benches)"],
        [V("&delta;"), "&asymp; 1.53 / yr", "set by the lag, so the gap stays constant"]
      ] },

    { kind: "eq", eyebrow: "03 &middot; Value",
      title: "What capability is worth",
      eq: [ line(V("W") + "(" + V("x") + ")" + o("=") + V("W") + b("0") + dot + "10" + S(V("&nu;") + V("x"))) ],
      gloss: "A model's dollar value climbs with capability &mdash; <b>each order of magnitude is worth several times the last</b>, so here value compounds without limit (a saturating bend arrives only in a later model). How steep that climb is (&nu;) isn't tightly pinned down &mdash; it's one of the dials you set in the explorer." },

    { kind: "eq", eyebrow: "04 &middot; Revenue",
      title: "Revenue is the rent on the lead",
      eq: [ line("revenue" + o("=") + V("&theta;") + dot + "[ " + V("W") + "(" + xL + ")" + o("&minus;") + V("W") + "(" + xF + ") ]") ],
      gloss: "The leader can only charge for the <b>value gap</b> over the free follower &mdash; the rent it collects for being ahead. A margin &theta; scales that rent down for competition among the leading labs; how large a margin survives is a judgement call you can vary." },

    { kind: "eq", eyebrow: "05 &middot; Cost",
      title: "Paying for the compute",
      eq: [ line("cost(" + V("t") + ")" + o("=") + V("S") + b("0") + dot + "10" + S(V("c") + S("L") + "(" + V("t") + ")&minus;" + V("c") + S("L") + "(0)") + dot + "10" + S("&minus;" + V("g") + b("p") + V("t"))) ],
      gloss: "The bill is the compute of the model running <b>now</b>, at prices that fall each year. Cost today is just S&#8320; &mdash; today's training spend &mdash; yet the net bill still grows &asymp; <b>2.4&times;/yr</b>: compute scales faster than prices drop.",
      params: [
        [V("g") + b("p"), "&asymp; 0.243 OOM/yr", "set so the net bill grows &times;2.4 / yr (compute prices, Cottier)"]
      ] },

    { kind: "eq", eyebrow: "06 &middot; Profit",
      title: "Does the rent beat the bill?",
      eq: [
        line(V("&Pi;") + o("=") + "revenue" + o("&minus;") + "cost"),
        '<span class="big">' + V("&nu;") + "(" + V("g") + b("c") + o("+") + V("g") + b("a") + ")"
          + o("&gt;") + V("g") + b("c") + o("&minus;") + V("g") + b("p") + "</span>"
      ],
      gloss: "With value compounding and the gap steady, the whole question collapses to a <b>race between two growth rates</b>. When that inequality holds &mdash; the value of the lead outpacing the cost of holding it &mdash; the leader turns profitable and stays there; otherwise, never. Nudge value-per-OOM past the pivot and the verdict flips." },

    { kind: "outro",
      eyebrow: "The full picture",
      title: "There's a lot more to explore",
      gloss: "The explorer layers more mechanisms on top of this basic model &mdash; AI-accelerated R&amp;D, the compute slowdown, value saturation, the follower's own engine &mdash; and lets you test which ones actually change the verdict.",
      // phone (D-073): the deck is the whole, TERMINAL phone experience — no widget to
      // send to, so this is a plain signpost, not a link. laptop: you're already there —
      // the overlay's own button does the leaving, so footLaptop is a quiet reassurance.
      foot: "Open on a laptop for the interactive explorer.",
      footLaptop: "Close this intro whenever you like &mdash; the explorer is right behind it.",
      credit: "Independent work in progress by Pavel Kocourek &mdash;<br>developed during the CAIS Fellowship." }
  ];

  // ---- render one slide (env: "phone" | "laptop") ----
  function slideHTML(s, env) {
    var h = '<article class="slide" data-kind="' + s.kind + '">';
    h += '<div class="slide-top"><div class="eyebrow">' + s.eyebrow + '</div></div>';
    h += '<div class="slide-body"><h1 class="title">' + s.title + '</h1>';
    if (s.eq)    h += '<div class="eq">' + s.eq.join("") + '</div>';
    if (s.gloss) h += '<p class="gloss">' + s.gloss + '</p>';
    if (s.list) {
      h += '<ul class="list">';
      s.list.forEach(function (it) { h += '<li><span class="k">' + it[0] + '</span><span>' + it[1] + '</span></li>'; });
      h += '</ul>';
    }
    if (s.params) {
      h += '<ul class="params">';
      s.params.forEach(function (p) {
        h += '<li><span class="row1"><span class="sym">' + p[0] + '</span><span class="val">' + p[1] + '</span></span>'
          + '<span class="note">' + p[2] + '</span></li>';
      });
      h += '</ul>';
    }
    var foot = (env === "laptop" && s.footLaptop) ? s.footLaptop : s.foot;
    if (foot)     h += '<p class="foot">' + foot + '</p>';
    if (s.credit) h += '<p class="credit">' + s.credit + '</p>';
    if (s.hint) {
      var hintTxt = env === "laptop"
        ? '<span>Use the arrows to step through</span><span class="ar">&#8250;</span>'
        : '<span>Swipe to begin</span><span class="ar">&#8250;</span>';
      h += '<p class="swipehint">' + hintTxt + '</p>';
    }
    h += '</div></article>';
    return h;
  }

  /* Build the deck (track + navigator) inside `host` and wire nav / scroll /
     keyboard. Returns a small controller { go, setActive, current(), N }.

     opts:
       env    "phone" | "laptop"   (default "phone") — filters `only` slides,
                                    swaps hint/foot wording.
       slides explicit slide array — overrides the env-filtered SLIDES (used for
              the minimal single-slide return-visit card).
       onIndex(i, N) — optional callback fired on every slide change (the laptop
              overlay uses it to keep its own affordances in step). */
  function mountDeck(host, opts) {
    opts = opts || {};
    var env = opts.env || "phone";
    var data = opts.slides || SLIDES.filter(function (s) { return !s.only || s.only === env; });

    host.classList.add("flp-deck");
    host.innerHTML =
      '<div class="track" tabindex="0" aria-label="Introduction slides"></div>' +
      '<div class="nav">' +
        '<div class="progress"><i class="bar"></i></div>' +
        '<button class="arrow prev" aria-label="Previous slide">&#8249;</button>' +
        '<div class="dots" role="tablist" aria-label="Slides"></div>' +
        '<button class="arrow next" aria-label="Next slide">&#8250;</button>' +
        '<span class="count"></span>' +
      '</div>';

    var track = host.querySelector(".track");
    var dotsEl = host.querySelector(".dots");
    var bar = host.querySelector(".bar");
    var count = host.querySelector(".count");
    var prev = host.querySelector(".prev");
    var next = host.querySelector(".next");
    var navEl = host.querySelector(".nav");

    track.innerHTML = data.map(function (s) { return slideHTML(s, env); }).join("");
    var slides = [].slice.call(track.children), N = slides.length;

    // single-slide (minimal) mount → no navigator at all
    if (N <= 1) navEl.classList.add("is-hidden");

    slides.forEach(function (_, i) {
      var dt = document.createElement("button");
      dt.setAttribute("role", "tab"); dt.setAttribute("aria-label", "Slide " + (i + 1));
      dt.addEventListener("click", function () { go(i); });
      dotsEl.appendChild(dt);
    });
    var dots = [].slice.call(dotsEl.children);

    var cur = -1;
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    function setActive(i) {
      if (i === cur) return; cur = i;
      slides.forEach(function (s, j) { s.classList.toggle("is-active", j === i); });
      dots.forEach(function (dd, j) { dd.setAttribute("aria-current", j === i ? "true" : "false"); });
      if (bar) bar.style.width = (N > 1 ? (i / (N - 1) * 100) : 100) + "%";
      if (count) count.textContent = pad(i + 1) + " / " + pad(N);
      prev.disabled = i === 0; next.disabled = i === N - 1;
      if (opts.onIndex) opts.onIndex(i, N);
    }
    function go(i) {
      i = Math.max(0, Math.min(N - 1, i));
      track.scrollTo({ left: i * track.clientWidth, behavior: "auto" });
      setActive(i);
    }
    var raf = 0;
    track.addEventListener("scroll", function () {
      if (raf) return;
      raf = requestAnimationFrame(function () { raf = 0; setActive(Math.round(track.scrollLeft / track.clientWidth)); });
    });
    prev.addEventListener("click", function () { go(cur - 1); });
    next.addEventListener("click", function () { go(cur + 1); });
    track.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); go(cur + 1); }
      if (e.key === "ArrowLeft"  || e.key === "PageUp")   { e.preventDefault(); go(cur - 1); }
      if (e.key === "Home") { e.preventDefault(); go(0); }
      if (e.key === "End")  { e.preventDefault(); go(N - 1); }
    });
    var rt = 0;
    window.addEventListener("resize", function () {
      clearTimeout(rt); rt = setTimeout(function () { track.scrollLeft = cur * track.clientWidth; }, 120);
    });

    setActive(0);
    return {
      go: go, setActive: setActive, N: N,
      current: function () { return cur; },
      focus: function () { try { track.focus(); } catch (e) {} }
    };
  }

  return { SLIDES: SLIDES, slideHTML: slideHTML, mountDeck: mountDeck };
})();
