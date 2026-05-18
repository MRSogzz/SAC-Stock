import { useEffect, useRef } from "react";

export default function PortfolioChart({ portfolio, bh, dates }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !portfolio?.length) return;
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.offsetWidth, H = 200;
    canvas.width = W * 2; canvas.height = H * 2; ctx.scale(2, 2);
    ctx.clearRect(0, 0, W, H);

    const all = [...portfolio, ...(bh || [])];
    const min = Math.min(...all), max = Math.max(...all), range = max - min || 1;
    const pad = { l: 48, r: 16, t: 16, b: 32 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const x = (i) => pad.l + (i / (portfolio.length - 1)) * iw;
    const y = (v) => pad.t + ih - ((v - min) / range) * ih;

    // Grid lines
    ctx.strokeStyle = "rgba(128,128,128,0.12)"; ctx.lineWidth = 0.5;
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const yy = pad.t + ih - t * ih;
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + iw, yy); ctx.stroke();
      const val = (min + t * range - 1) * 100;
      ctx.fillStyle = "rgba(128,128,128,0.5)"; ctx.font = "10px sans-serif";
      ctx.fillText((val >= 0 ? "+" : "") + val.toFixed(0) + "%", 2, yy + 3);
    });

    // X-axis date labels
    if (dates?.length) {
      ctx.fillStyle = "rgba(128,128,128,0.5)"; ctx.font = "9px sans-serif";
      [0, Math.floor(dates.length / 2), dates.length - 1].forEach((i) => {
        ctx.fillText(dates[Math.min(i, dates.length - 1)]?.slice(0, 7) || "", x(i) - 16, H - 4);
      });
    }

    // Buy-and-hold line (dashed)
    if (bh?.length) {
      ctx.beginPath(); ctx.moveTo(x(0), y(bh[0]));
      bh.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      ctx.strokeStyle = "rgba(128,128,128,0.4)"; ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }

    // Portfolio line + fill
    ctx.beginPath(); ctx.moveTo(x(0), y(portfolio[0]));
    portfolio.forEach((v, i) => ctx.lineTo(x(i), y(v)));
    const c = portfolio[portfolio.length - 1] >= portfolio[0] ? "#1D9E75" : "#E24B4A";
    ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke();

    const g = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
    g.addColorStop(0, c + "30"); g.addColorStop(1, c + "00");
    ctx.lineTo(x(portfolio.length - 1), pad.t + ih);
    ctx.lineTo(x(0), pad.t + ih);
    ctx.closePath(); ctx.fillStyle = g; ctx.fill();
  }, [portfolio, bh, dates]);

  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height: 200, display: "block" }} />
      <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 20, height: 2, background: "#1D9E75", display: "inline-block" }} />
          AI 組合
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 20, height: 2, borderTop: "2px dashed rgba(128,128,128,0.4)", display: "inline-block" }} />
          等權重持有
        </span>
      </div>
    </div>
  );
}