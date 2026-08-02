// Precision & recall vs decision threshold, drawn as inline SVG (no chart library).
// Both series live on one 0–100% axis, so there is no misleading dual-scale. A vertical
// marker shows the currently selected threshold. Colours: recall = brand blue,
// precision = violet — neither collides with the fraud-red / legit-green status colours.
import { CurvePoint } from "../api";

const RECALL = "#3987e5";
const PRECISION = "#9085e9";

const W = 640;
const H = 260;
const PAD = { top: 20, right: 56, bottom: 36, left: 44 };

function recall(p: CurvePoint) {
  const denom = p.TP + p.FN;
  return denom === 0 ? null : p.TP / denom;
}
function precision(p: CurvePoint) {
  const denom = p.TP + p.FP;
  return denom === 0 ? null : p.TP / denom;
}

export function TradeoffChart({
  curve,
  threshold,
}: {
  curve: CurvePoint[];
  threshold: number;
}) {
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (t: number) => PAD.left + t * plotW;
  const y = (v: number) => PAD.top + (1 - v) * plotH;

  const line = (fn: (p: CurvePoint) => number | null) =>
    curve
      .map((p) => {
        const v = fn(p);
        return v == null ? null : `${x(p.threshold)},${y(v)}`;
      })
      .filter(Boolean)
      .join(" ");

  const markerX = x(Math.min(1, Math.max(0, threshold)));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
         aria-label="Precision and recall versus decision threshold">
      {/* horizontal gridlines at 0/25/50/75/100% */}
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(g)} y2={y(g)} stroke="#26292e" />
          <text x={PAD.left - 8} y={y(g) + 4} textAnchor="end" fontSize="11" fill="#6b6f77">
            {g * 100}%
          </text>
        </g>
      ))}
      {/* x-axis ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <text key={t} x={x(t)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize="11"
              fill="#6b6f77">
          {t.toFixed(2)}
        </text>
      ))}
      <text x={PAD.left + plotW / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="#a2a6ad">
        Decision threshold
      </text>

      {/* current-threshold marker */}
      <line x1={markerX} x2={markerX} y1={PAD.top} y2={H - PAD.bottom} stroke="#a2a6ad"
            strokeDasharray="4 3" />

      {/* series */}
      <polyline points={line(recall)} fill="none" stroke={RECALL} strokeWidth="2.5" />
      <polyline points={line(precision)} fill="none" stroke={PRECISION} strokeWidth="2.5" />

      {/* direct labels at the right edge */}
      <text x={W - PAD.right + 6} y={y(0.5)} fontSize="11" fill={RECALL} fontWeight="600">
        Recall
      </text>
      <text x={W - PAD.right + 6} y={y(0.5) + 16} fontSize="11" fill={PRECISION}
            fontWeight="600">
        Precision
      </text>
    </svg>
  );
}
