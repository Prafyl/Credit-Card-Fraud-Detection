// Loads every model's 201-point threshold sweep, which the 3D decision surface needs.
//
// There is no bulk endpoint, so this is 8 separate GET /model-info calls. They are fired in
// parallel and cached at module level, so switching tabs or remounting a canvas never
// refetches. The backend is deliberately untouched.
import { useEffect, useState } from "react";
import { api, CurvePoint } from "../api";

const cache = new Map<string, CurvePoint[]>();

export function useModelCurves(modelNames: string[]) {
  const [curves, setCurves] = useState<Map<string, CurvePoint[]>>(new Map(cache));
  const [loading, setLoading] = useState(false);

  // Join the names so the effect re-runs when the model list actually changes, not on every
  // render (a fresh array literal from the parent would otherwise retrigger it forever).
  const key = modelNames.join("|");

  useEffect(() => {
    if (modelNames.length === 0) return;
    const missing = modelNames.filter((m) => !cache.has(m));
    if (missing.length === 0) {
      setCurves(new Map(cache));
      return;
    }

    let cancelled = false;
    setLoading(true);

    Promise.all(
      missing.map((name) =>
        api
          .modelInfo(name)
          .then((info) => ({ name, curve: info.threshold_curve }))
          .catch(() => null),
      ),
    ).then((results) => {
      for (const r of results) {
        if (r) cache.set(r.name, r.curve);
      }
      if (!cancelled) {
        setCurves(new Map(cache));
        setLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return { curves, loading };
}
