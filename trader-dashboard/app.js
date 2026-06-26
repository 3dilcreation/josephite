/* ============================================================
   APEX Terminal — Trader Dashboard
   Self-contained simulated trading environment.
   No external dependencies; charts drawn on <canvas>.
   ============================================================ */

(() => {
  "use strict";

  /* ---------- Instrument universe ---------- */
  const INSTRUMENTS = [
    { sym: "BTC",  name: "Bitcoin",        price: 67250.0, vol: 0.018 },
    { sym: "ETH",  name: "Ethereum",       price: 3540.0,  vol: 0.022 },
    { sym: "SOL",  name: "Solana",         price: 172.4,   vol: 0.030 },
    { sym: "AAPL", name: "Apple Inc.",     price: 214.8,   vol: 0.010 },
    { sym: "NVDA", name: "NVIDIA Corp.",   price: 126.3,   vol: 0.020 },
    { sym: "TSLA", name: "Tesla Inc.",     price: 248.5,   vol: 0.025 },
    { sym: "EURUSD", name: "Euro / USD",   price: 1.0842,  vol: 0.004 },
    { sym: "XAU",  name: "Gold (oz)",      price: 2338.0,  vol: 0.008 },
  ];

  /* Timeframe -> (candle count, ms per bar for live build) */
  const TIMEFRAMES = {
    "1m":  { bars: 90,  step: 1 },
    "5m":  { bars: 90,  step: 5 },
    "15m": { bars: 80,  step: 15 },
    "1h":  { bars: 72,  step: 60 },
    "1d":  { bars: 60,  step: 1440 },
  };

  /* ---------- State ---------- */
  const state = {
    activeSym: "BTC",
    timeframe: "5m",
    side: "buy",
    cash: 100000,
    positions: {},        // sym -> { qty, avg }   qty>0 long, qty<0 short
    series: {},           // sym -> { "5m": [candles], ... }
    last: {},             // sym -> last price
    prevClose: {},        // sym -> session open for % change
  };

  const $ = (id) => document.getElementById(id);
  const fmt = (n, d = 2) =>
    n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtUsd = (n, d = 2) => (n < 0 ? "-$" : "$") + fmt(Math.abs(n), d);
  const priceDecimals = (p) => (p < 2 ? 4 : p < 1000 ? 2 : 1);

  /* ---------- Seeded candle generation ----------
     Generates a random walk, then rescales the whole series so its final
     close lands exactly on `target`. This keeps every timeframe anchored to
     one shared "current price", so switching timeframes never makes the
     quoted price jump. */
  function genCandles(target, vol, count, drift = 0) {
    const raw = [];
    let price = 1; // unit walk; rescaled to `target` afterwards
    for (let i = 0; i < count; i++) {
      const open = price;
      const shock = (Math.random() - 0.5) * vol * 2 + drift;
      const close = Math.max(open * (1 + shock), 1e-6);
      const hi = Math.max(open, close) * (1 + Math.random() * vol);
      const lo = Math.min(open, close) * (1 - Math.random() * vol);
      raw.push({ open, high: hi, low: lo, close, volume: Math.round((0.6 + Math.random()) * 1000) });
      price = close;
    }
    const scale = target / raw[raw.length - 1].close; // force last close == target
    return raw.map((c) => ({
      open: c.open * scale, high: c.high * scale,
      low: c.low * scale, close: c.close * scale, volume: c.volume,
    }));
  }

  function ensureSeries(sym) {
    if (!state.series[sym]) state.series[sym] = {};
    const inst = INSTRUMENTS.find((i) => i.sym === sym);
    // current spot price for the symbol — shared across all timeframes
    if (state.last[sym] === undefined) state.last[sym] = inst.price;
    const spot = state.last[sym];
    for (const tf in TIMEFRAMES) {
      if (!state.series[sym][tf]) {
        const v = inst.vol * Math.sqrt(TIMEFRAMES[tf].step);
        state.series[sym][tf] = genCandles(spot, v, TIMEFRAMES[tf].bars, 0.0006);
      }
    }
    if (state.prevClose[sym] === undefined) {
      const day = state.series[sym]["1d"];
      state.prevClose[sym] = day[Math.max(0, day.length - 2)].close;
    }
  }

  INSTRUMENTS.forEach((i) => ensureSeries(i.sym));

  /* ---------- Live tick engine ---------- */
  function tick() {
    INSTRUMENTS.forEach((inst) => {
      const sym = inst.sym;
      const last = state.last[sym];
      const shock = (Math.random() - 0.5) * inst.vol * 0.9;
      let next = Math.max(last * (1 + shock), 0.0001);
      state.last[sym] = next;

      // update the most recent candle on the active timeframe view
      const candles = state.series[sym][state.timeframe];
      const cur = candles[candles.length - 1];
      cur.close = next;
      cur.high = Math.max(cur.high, next);
      cur.low = Math.min(cur.low, next);
      cur.volume += Math.round(Math.random() * 40);
    });

    // occasionally roll a new candle on the active symbol/timeframe
    if (Math.random() < 0.08) rollCandle(state.activeSym, state.timeframe);

    renderLiveBits();
  }

  // Snap the active timeframe's final candle to the current spot price so the
  // chart's last bar and the live price line always agree after a switch.
  function syncLastCandle(sym, tf) {
    const candles = state.series[sym][tf];
    const cur = candles[candles.length - 1];
    const spot = state.last[sym];
    cur.close = spot;
    cur.high = Math.max(cur.high, spot);
    cur.low = Math.min(cur.low, spot);
  }

  function rollCandle(sym, tf) {
    const candles = state.series[sym][tf];
    const last = state.last[sym];
    candles.push({ open: last, high: last, low: last, close: last, volume: 0 });
    if (candles.length > TIMEFRAMES[tf].bars) candles.shift();
  }

  /* ---------- Derived values ---------- */
  function positionList() {
    return Object.entries(state.positions)
      .filter(([, p]) => Math.abs(p.qty) > 1e-9)
      .map(([sym, p]) => {
        const last = state.last[sym];
        const mktValue = p.qty * last;
        const cost = p.qty * p.avg;
        const pnl = mktValue - cost;
        return { sym, ...p, last, mktValue, pnl };
      });
  }

  function portfolioValue() {
    return state.cash + positionList().reduce((s, p) => s + p.mktValue, 0);
  }

  function dayPnl() {
    return positionList().reduce((s, p) => {
      const ref = state.prevClose[p.sym] || p.avg;
      return s + p.qty * (p.last - ref);
    }, 0);
  }

  /* ---------- Canvas chart ---------- */
  const canvas = $("priceChart");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    const wrap = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = wrap.clientWidth * dpr;
    canvas.height = wrap.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawChart();
  }

  function drawChart() {
    const sym = state.activeSym;
    const candles = state.series[sym][state.timeframe];
    const W = canvas.clientWidth, H = canvas.clientHeight;
    const padL = 8, padR = 64, padT = 14, padB = 22;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    ctx.clearRect(0, 0, W, H);

    let hi = -Infinity, lo = Infinity;
    candles.forEach((c) => { hi = Math.max(hi, c.high); lo = Math.min(lo, c.low); });
    const span = (hi - lo) || 1;
    hi += span * 0.08; lo -= span * 0.08;
    const range = hi - lo;

    const x = (i) => padL + (i + 0.5) * (plotW / candles.length);
    const y = (p) => padT + (1 - (p - lo) / range) * plotH;
    const d = priceDecimals(state.last[sym]);

    // grid + axis labels
    ctx.font = "11px monospace";
    ctx.textBaseline = "middle";
    const lines = 5;
    for (let g = 0; g <= lines; g++) {
      const py = padT + (plotH * g) / lines;
      ctx.strokeStyle = "#16203250";
      ctx.strokeStyle = "rgba(30,42,63,0.5)";
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + plotW, py); ctx.stroke();
      const pv = hi - (range * g) / lines;
      ctx.fillStyle = "#7c8aa3";
      ctx.textAlign = "left";
      ctx.fillText(fmt(pv, d), padL + plotW + 6, py);
    }

    // candles
    const bw = Math.max(2, (plotW / candles.length) * 0.62);
    candles.forEach((c, i) => {
      const up = c.close >= c.open;
      const col = up ? "#22c55e" : "#ef4444";
      const cx = x(i);
      ctx.strokeStyle = col; ctx.fillStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, y(c.high)); ctx.lineTo(cx, y(c.low)); ctx.stroke();
      const yo = y(c.open), yc = y(c.close);
      const top = Math.min(yo, yc);
      const h = Math.max(1, Math.abs(yc - yo));
      ctx.fillRect(cx - bw / 2, top, bw, h);
    });

    // current price line
    const last = state.last[sym];
    const ly = y(last);
    ctx.strokeStyle = "rgba(59,130,246,0.7)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, ly); ctx.lineTo(padL + plotW, ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(padL + plotW, ly - 9, padR, 18);
    ctx.fillStyle = "#fff"; ctx.textAlign = "left";
    ctx.fillText(fmt(last, d), padL + plotW + 6, ly);

    // meta overlay (OHLC of last candle)
    const lastC = candles[candles.length - 1];
    $("chartMeta").innerHTML =
      `O <b>${fmt(lastC.open, d)}</b> H <b>${fmt(lastC.high, d)}</b> ` +
      `L <b>${fmt(lastC.low, d)}</b> C <b>${fmt(lastC.close, d)}</b>`;
  }

  /* ---------- Order book (synthetic) ---------- */
  function renderOrderBook() {
    const sym = state.activeSym;
    const mid = state.last[sym];
    const d = priceDecimals(mid);
    const tick = mid * 0.0004;
    const levels = 7;
    let asks = [], bids = [];
    let aTot = 0, bTot = 0;
    for (let i = 1; i <= levels; i++) {
      const size = +(Math.random() * 6 + 0.4).toFixed(3);
      aTot += size; bTot += size;
      asks.push({ price: mid + tick * i, size, total: aTot });
      bids.push({ price: mid - tick * i, size, total: bTot });
    }
    const maxTot = Math.max(aTot, bTot);
    const row = (lvl, cls) =>
      `<div class="book-row ${cls}">
        <span class="b-price">${fmt(lvl.price, d)}</span>
        <span class="b-size">${fmt(lvl.size, 3)}</span>
        <span class="b-total">${fmt(lvl.total, 2)}</span>
        <span class="bar" style="width:${(lvl.total / maxTot) * 100}%"></span>
      </div>`;
    const spread = asks[0].price - bids[0].price;
    const spreadPct = (spread / mid) * 100;
    $("spreadLabel").textContent = `spread ${fmt(spread, d)} (${fmt(spreadPct, 3)}%)`;
    $("orderBook").innerHTML =
      asks.reverse().map((l) => row(l, "ask")).join("") +
      `<div class="book-mid">${fmt(mid, d)} <span class="arrow">●</span></div>` +
      bids.map((l) => row(l, "bid")).join("");
  }

  /* ---------- Sparklines ---------- */
  function sparkline(candles) {
    const w = 60, h = 26, n = Math.min(candles.length, 30);
    const pts = candles.slice(-n).map((c) => c.close);
    const hi = Math.max(...pts), lo = Math.min(...pts), rng = hi - lo || 1;
    const up = pts[pts.length - 1] >= pts[0];
    const col = up ? "#22c55e" : "#ef4444";
    const path = pts
      .map((p, i) => `${(i / (n - 1)) * w},${h - ((p - lo) / rng) * (h - 4) - 2}`)
      .join(" ");
    return `<svg class="spark" viewBox="0 0 ${w} ${h}"><polyline fill="none" stroke="${col}" stroke-width="1.5" points="${path}"/></svg>`;
  }

  /* ---------- Renderers ---------- */
  function renderWatchlist() {
    $("watchList").innerHTML = INSTRUMENTS.map((inst) => {
      const last = state.last[inst.sym];
      const ref = state.prevClose[inst.sym];
      const chg = ((last - ref) / ref) * 100;
      const cls = chg >= 0 ? "up" : "down";
      const d = priceDecimals(last);
      return `<div class="watch-row ${inst.sym === state.activeSym ? "active" : ""}" data-sym="${inst.sym}">
        <div><div class="w-sym">${inst.sym}</div><div class="w-name">${inst.name}</div></div>
        ${sparkline(state.series[inst.sym]["5m"])}
        <div class="w-price">${fmt(last, d)}</div>
        <div class="w-chg ${cls}">${chg >= 0 ? "+" : ""}${fmt(chg, 2)}%</div>
      </div>`;
    }).join("");
    $("watchList").querySelectorAll(".watch-row").forEach((r) =>
      r.addEventListener("click", () => selectSymbol(r.dataset.sym))
    );
  }

  function renderTicker() {
    $("tickerStrip").innerHTML = INSTRUMENTS.map((inst) => {
      const last = state.last[inst.sym];
      const ref = state.prevClose[inst.sym];
      const chg = ((last - ref) / ref) * 100;
      const cls = chg >= 0 ? "up" : "down";
      return `<span class="ticker-item"><span class="t-sym">${inst.sym}</span>
        <span class="t-price">${fmt(last, priceDecimals(last))}</span>
        <span class="${cls}">${chg >= 0 ? "▲" : "▼"}${fmt(Math.abs(chg), 2)}%</span></span>`;
    }).join("");
  }

  function renderSymbolHeader() {
    const sym = state.activeSym;
    const inst = INSTRUMENTS.find((i) => i.sym === sym);
    const last = state.last[sym];
    const ref = state.prevClose[sym];
    const chg = last - ref;
    const chgPct = (chg / ref) * 100;
    const cls = chg >= 0 ? "up" : "down";
    const d = priceDecimals(last);
    $("symName").textContent = sym + "/USD";
    $("symFull").textContent = inst.name;
    $("symPrice").textContent = "$" + fmt(last, d);
    $("symChange").className = "chg " + cls;
    $("symChange").textContent = `${chg >= 0 ? "+" : ""}${fmt(chg, d)} (${chg >= 0 ? "+" : ""}${fmt(chgPct, 2)}%)`;
    $("submitSym").textContent = sym;

    const candles = state.series[sym][state.timeframe];
    let hi = -Infinity, lo = Infinity, vol = 0;
    candles.forEach((c) => { hi = Math.max(hi, c.high); lo = Math.min(lo, c.low); vol += c.volume; });
    $("chartStats").innerHTML =
      `<div><span>24h High</span><b>${fmt(hi, d)}</b></div>
       <div><span>24h Low</span><b>${fmt(lo, d)}</b></div>
       <div><span>Volume</span><b>${fmt(vol, 0)}</b></div>
       <div><span>Last</span><b class="${cls}">${fmt(last, d)}</b></div>`;
  }

  function renderKpis() {
    const pv = portfolioValue();
    const dp = dayPnl();
    const positions = positionList();
    const exposure = positions.reduce((s, p) => s + Math.abs(p.mktValue), 0);
    const dpPct = pv ? (dp / pv) * 100 : 0;

    $("kpiValue").textContent = fmtUsd(pv);
    const vd = $("kpiValueDelta");
    vd.textContent = `${dp >= 0 ? "▲" : "▼"} ${fmtUsd(Math.abs(dp))} today`;
    vd.className = "kpi-delta " + (dp >= 0 ? "up" : "down");

    $("kpiPnl").textContent = (dp >= 0 ? "+" : "") + fmtUsd(dp);
    $("kpiPnl").className = "kpi-value " + (dp >= 0 ? "up" : "down");
    const pd = $("kpiPnlDelta");
    pd.textContent = `${dp >= 0 ? "+" : ""}${fmt(dpPct, 2)}%`;
    pd.className = "kpi-delta " + (dp >= 0 ? "up" : "down");

    $("kpiBuyingPower").textContent = fmtUsd(state.cash);
    $("kpiPositions").textContent = positions.length;
    $("kpiExposure").textContent = fmtUsd(exposure) + " exposure";
    $("equityValue").textContent = fmtUsd(pv);
  }

  function renderPositions() {
    const positions = positionList();
    const body = $("positionsBody");
    $("positionsEmpty").style.display = positions.length ? "none" : "block";
    body.innerHTML = positions.map((p) => {
      const d = priceDecimals(p.last);
      const side = p.qty >= 0 ? "buy" : "sell";
      const pnlCls = p.pnl >= 0 ? "up" : "down";
      return `<tr>
        <td class="sym-cell">${p.sym}</td>
        <td><span class="pill ${side}">${side === "buy" ? "LONG" : "SHORT"}</span></td>
        <td class="num">${fmt(Math.abs(p.qty), 4)}</td>
        <td class="num">${fmt(p.avg, d)}</td>
        <td class="num">${fmt(p.last, d)}</td>
        <td class="num">${fmtUsd(p.mktValue)}</td>
        <td class="num ${pnlCls}">${p.pnl >= 0 ? "+" : ""}${fmtUsd(p.pnl)}</td>
        <td class="num"><button class="close-btn" data-close="${p.sym}">Close</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll(".close-btn").forEach((b) =>
      b.addEventListener("click", () => closePosition(b.dataset.close))
    );
  }

  /* parts that update on every tick (cheap) */
  function renderLiveBits() {
    renderTicker();
    renderSymbolHeader();
    renderKpis();
    renderPositions();
    renderWatchlist();
    renderOrderBook();
    drawChart();
    updateTicketSummary();
  }

  /* ---------- Trading actions ---------- */
  function currentQty() {
    return Math.max(0, parseFloat($("qty").value) || 0);
  }

  function execPrice() {
    const type = $("orderType").value;
    if (type === "market") return state.last[state.activeSym];
    const lp = parseFloat($("limitPrice").value);
    return lp > 0 ? lp : state.last[state.activeSym];
  }

  function updateTicketSummary() {
    const qty = currentQty();
    const price = execPrice();
    const cost = qty * price;
    const fee = cost * 0.001;
    const total = state.side === "buy" ? cost + fee : cost - fee;
    const d = priceDecimals(price);
    $("estCost").textContent = fmtUsd(cost);
    $("estFee").textContent = fmtUsd(fee);
    $("estTotal").textContent = fmtUsd(Math.abs(total));
  }

  function placeOrder() {
    const sym = state.activeSym;
    const qtyRaw = currentQty();
    const price = execPrice();
    const msg = $("ticketMsg");
    if (qtyRaw <= 0) { msg.textContent = "Enter a quantity."; msg.className = "ticket-msg down"; return; }

    const signed = state.side === "buy" ? qtyRaw : -qtyRaw;
    const cost = qtyRaw * price;
    const fee = cost * 0.001;

    if (state.side === "buy" && cost + fee > state.cash + 1e-6) {
      msg.textContent = "Insufficient buying power."; msg.className = "ticket-msg down"; return;
    }

    const pos = state.positions[sym] || { qty: 0, avg: price };
    const newQty = pos.qty + signed;

    // weighted average only when increasing in same direction
    if (pos.qty === 0 || Math.sign(pos.qty) === Math.sign(signed)) {
      pos.avg = (Math.abs(pos.qty) * pos.avg + qtyRaw * price) / (Math.abs(pos.qty) + qtyRaw);
    } else if (Math.sign(newQty) !== Math.sign(pos.qty) && newQty !== 0) {
      pos.avg = price; // flipped direction
    }
    pos.qty = Math.abs(newQty) < 1e-9 ? 0 : newQty;
    state.positions[sym] = pos;

    // cash impact: buying spends, selling receives; fee always paid
    state.cash += (state.side === "buy" ? -cost : cost) - fee;

    toast(state.side, `${state.side === "buy" ? "Bought" : "Sold"} ${fmt(qtyRaw, 4)} ${sym}`,
      `@ ${fmt(price, priceDecimals(price))} · fee ${fmtUsd(fee)}`);
    msg.textContent = "Order filled."; msg.className = "ticket-msg up";
    setTimeout(() => (msg.textContent = ""), 2500);
    renderLiveBits();
  }

  function closePosition(sym) {
    const pos = state.positions[sym];
    if (!pos || Math.abs(pos.qty) < 1e-9) return;
    const price = state.last[sym];
    const proceeds = pos.qty * price;   // qty<0 (short) -> negative -> we pay back
    const fee = Math.abs(proceeds) * 0.001;
    state.cash += proceeds - fee;
    const pnl = pos.qty * (price - pos.avg);
    pos.qty = 0;
    toast(pnl >= 0 ? "buy" : "sell", `Closed ${sym}`,
      `P&L ${pnl >= 0 ? "+" : ""}${fmtUsd(pnl)}`);
    renderLiveBits();
  }

  /* ---------- UI wiring ---------- */
  function selectSymbol(sym) {
    state.activeSym = sym;
    ensureSeries(sym);
    syncLastCandle(sym, state.timeframe);
    const lp = $("limitPrice");
    lp.value = fmt(state.last[sym], priceDecimals(state.last[sym]));
    renderLiveBits();
  }

  function setSide(side) {
    state.side = side;
    document.querySelectorAll(".ticket-tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.side === side));
    const btn = $("submitOrder");
    btn.className = "submit-order " + side;
    btn.firstChild.textContent = (side === "buy" ? "Buy " : "Sell ");
    updateTicketSummary();
  }

  function toast(kind, title, sub) {
    const el = document.createElement("div");
    el.className = "toast " + kind;
    el.innerHTML = `<div class="t-title">${title}</div><div class="muted small">${sub}</div>`;
    $("toastWrap").appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; }, 3200);
    setTimeout(() => el.remove(), 3600);
  }

  function setupEvents() {
    // timeframe
    $("timeframes").querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        state.timeframe = b.dataset.tf;
        $("timeframes").querySelectorAll("button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        ensureSeries(state.activeSym);
        syncLastCandle(state.activeSym, state.timeframe);
        renderLiveBits();
      }));

    // side tabs
    document.querySelectorAll(".ticket-tabs button").forEach((b) =>
      b.addEventListener("click", () => setSide(b.dataset.side)));

    // order type toggle
    $("orderType").addEventListener("change", () => {
      const t = $("orderType").value;
      $("limitField").style.display = t === "market" ? "none" : "flex";
      if (t !== "market" && !$("limitPrice").value)
        $("limitPrice").value = fmt(state.last[state.activeSym], priceDecimals(state.last[state.activeSym]));
      updateTicketSummary();
    });

    $("qty").addEventListener("input", updateTicketSummary);
    $("limitPrice").addEventListener("input", updateTicketSummary);
    $("submitOrder").addEventListener("click", placeOrder);

    // quantity pills (size relative to buying power / position)
    $("qtyPills").querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        const pct = +b.dataset.pct / 100;
        const price = execPrice();
        if (state.side === "buy") {
          $("qty").value = +((state.cash * pct) / price).toFixed(4);
        } else {
          const pos = state.positions[state.activeSym];
          const held = pos ? Math.abs(pos.qty) : 0;
          $("qty").value = held > 0 ? +(held * pct).toFixed(4) : +((state.cash * pct) / price).toFixed(4);
        }
        updateTicketSummary();
      }));

    // sidebar nav (visual only — single view app)
    document.querySelectorAll(".nav-item").forEach((n) =>
      n.addEventListener("click", (e) => {
        e.preventDefault();
        document.querySelectorAll(".nav-item").forEach((x) => x.classList.remove("active"));
        n.classList.add("active");
      }));

    window.addEventListener("resize", resizeCanvas);
  }

  function startClock() {
    const upd = () => { $("clock").textContent = new Date().toLocaleTimeString("en-US", { hour12: false }); };
    upd(); setInterval(upd, 1000);
  }

  /* ---------- Boot ---------- */
  function init() {
    setupEvents();
    setSide("buy");
    selectSymbol("BTC");
    resizeCanvas();
    startClock();
    renderLiveBits();
    setInterval(tick, 900);
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
