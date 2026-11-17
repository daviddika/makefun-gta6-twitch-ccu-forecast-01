"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Method, in brief: anchor on the hour-1 print and predict each later hour's
log-ratio to it with a kernel-weighted linear regression fitted, at run time,
on the case's own comparables. Only past launches that unlocked at rolling
local midnight (like GTA VI) and were not early-access contaminated are used;
simultaneous-global launches and free-to-play / slow-burn launches are left
out, because their release-day curves have a different shape (their hour 1
is often still pre-unlock). The regression is on x = ln(hour-1 print) - 11,
weighted by a Gaussian kernel around the case's own hour-1 size, so big
launches (whose hour 1 already holds most of the world) get a flatter growth
curve than small ones, and the fit saturates instead of extrapolating a
steep decline for the very largest prints. Quantile spreads come from the
weighted residual spread of the same fit, shrunk by a factor tuned on the
practice cases and on leave-one-out validation across the comparables.

The program is deterministic (no randomness, no wall clock, fixed iteration
order), standard library only, and always emits a valid forecast: every
failure mode falls back to a fixed launch-day shape, and the last resort is
a persistence forecast, so a case is never left without an output.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (tuned on the practice cases + leave-one-out runs) ---
# Regression on x = ln(p1) - 11; Gaussian kernel bandwidth around the case's x.
BANDWIDTH = 2.0
# The kernel centre is clipped to this range of x (p1 from ~4,900 to ~1.1M):
# beyond it the fit is frozen at the edge of the observed launches, so a
# record print never extrapolates into a collapse.
X_CLIP = (-2.5, 2.75)
# Comparables below this hour-1 print are dropped: their hour 1 was usually
# still pre-unlock, so their growth is a different regime.
MIN_P1 = 1000.0
# A comparable needs its hour-1 point and at least this many release-day hours.
MIN_HOURS = 18
# Spread: weighted residual sd of the fit, shrunk by this factor.
SPREAD_MULT = 0.6
# Floor on the (pre-shrink) residual sd, in log units.
SD_FLOOR = 0.25
# Minimum rows needed to fit an hour; below that the fallback shape is used.
MIN_ROWS = 5

# Fallback launch-day shape: median log-ratio to hour 1, hours 2..24, over the
# rolling-midnight comparables (used only when the fit cannot run at all).
FALLBACK_MED = [
    -0.142, -0.283, -0.310, -0.219, -0.047, -0.021, -0.060, -0.089, -0.063,
    -0.075, -0.022, 0.109, 0.263, 0.440, 0.599, 0.669, 0.725, 0.934, 0.880,
    0.873, 0.847, 0.803, 0.735,
]
FALLBACK_SD = 0.45

# Unlock-text markers. A comparable is skipped when its unlock note names an
# early-access-contaminated or weak tier, or a simultaneous global unlock.
TIER_RE = re.compile(r"Tier:\s*([a-z\-]+)")
SIMULTANEOUS_MARKERS = (
    "Single simultaneous",
    "Single global unlock",
    "Simultaneous across platforms",
    "PC single time",
    "simultaneous global",
)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp, tolerating a +00:00 suffix."""
    if not isinstance(text, str):
        return None
    try:
        if text.endswith("Z"):
            return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S+00:00")
    except (ValueError, TypeError):
        return None


def round2(x):
    """x rounded to 2 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def is_rolling_launch(comp):
    """True when the comparable looks like a rolling-midnight launch.

    Excludes early-access-contaminated and weak tiers, and simultaneous
    global unlocks. A missing or unparseable unlock note does not exclude:
    the hour-1-size filter below is the backstop for those.
    """
    note = comp.get("unlock") or ""
    if not isinstance(note, str):
        return True
    match = TIER_RE.search(note)
    if match and match.group(1) in ("early-access-contaminated", "weak"):
        return False
    lowered = note.lower()
    for marker in SIMULTANEOUS_MARKERS:
        if marker.lower() in lowered:
            return False
    return True


def release_day_points(comp):
    """Map release-day hour (1..24) -> viewers, from a comparable's hourly."""
    day_start = parse_time(comp.get("day_start"))
    if day_start is None:
        return {}
    points = {}
    hourly = comp.get("hourly") or []
    for row in hourly:
        at = parse_time(row.get("from"))
        viewers = row.get("viewers")
        if at is None or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        delta = (at - day_start).total_seconds() / 3600.0
        if 0.0 <= delta < 24.0:
            points[int(delta) + 1] = float(viewers)
    return points


def collect_comparables(case):
    """Usable (hour-1 print, {hour: log-ratio}) pairs from the comparables."""
    usable = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        if not is_rolling_launch(comp):
            continue
        points = release_day_points(comp)
        if 1 not in points or len(points) < MIN_HOURS:
            continue
        p1 = points[1]
        if p1 < MIN_P1:
            continue
        ratios = {}
        for hour in range(2, 25):
            if hour in points:
                ratios[hour] = math.log(points[hour] / p1)
        if len(ratios) >= MIN_HOURS - 1:
            usable.append((math.log(p1) - 11.0, ratios))
    return usable


def hour_one_print(case):
    """Hour 1's print, with fallbacks if it is missing or unusable."""
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    # Last whole-hour point before the checkpoint.
    for row in reversed(case.get("history_hourly") or []):
        if isinstance(row, dict):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    # Last raw history point.
    for row in reversed(case.get("history") or []):
        if isinstance(row, dict):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    return 1000.0


def weighted_line(rows, x0):
    """Kernel-weighted least squares of y on x; rows are (x, y).

    Returns (intercept, slope, residual_sd). The weights are Gaussian in x
    around x0 with BANDWIDTH. Returns None when the fit is degenerate.
    """
    weighted = []
    for x, y in rows:
        w = math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2)
        weighted.append((x, y, w))
    sw = sum(w for _, _, w in weighted)
    if sw <= 0.0 or len(weighted) < MIN_ROWS:
        return None
    mx = sum(w * x for x, _, w in weighted) / sw
    my = sum(w * y for _, y, w in weighted) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in weighted)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in weighted)
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y, _ in weighted]
    var = sum(w * r * r for (_, _, w), r in zip(weighted, resid)) / max(sw - 2.0, 1.0)
    sd = max(SD_FLOOR, math.sqrt(var)) * SPREAD_MULT
    return intercept, slope, sd


def quantile_row(p1, mu, sd, hour):
    """One forecast row: five quantiles around p1 * exp(mu), spread sd."""
    quantiles = []
    prev = 0.0
    for z in Z:
        q = p1 * math.exp(mu + z * sd)
        q = round2(q)
        q = min(max(q, 0.01), 1.0e9)
        if q < prev:
            q = prev
        prev = q
        quantiles.append(q)
    return {"hour": hour, "q": quantiles}


def forecast_case(case):
    p1 = hour_one_print(case)
    p1 = min(max(p1, 1.0), 1.0e8)
    x_case = math.log(p1) - 11.0
    x0 = max(X_CLIP[0], min(X_CLIP[1], x_case))

    comps = collect_comparables(case)

    # Per-hour regression fits: (median log-ratio, spread) for hours 2..24.
    medians = {}
    spreads = {}
    for hour in range(2, 25):
        rows = [(x, ratios[hour]) for x, ratios in comps if hour in ratios]
        fit = weighted_line(rows, x0)
        if fit is None:
            medians[hour] = FALLBACK_MED[hour - 2]
            spreads[hour] = FALLBACK_SD
        else:
            intercept, slope, sd = fit
            medians[hour] = intercept + slope * x0
            spreads[hour] = sd

    hours = []
    for row in case.get("forecast") or []:
        hour = row.get("hour")
        if not isinstance(hour, int) or hour < 2 or hour > 24:
            continue
        hours.append(quantile_row(
            p1, medians.get(hour, FALLBACK_MED[hour - 2]),
            spreads.get(hour, FALLBACK_SD), hour))
    # The forecast list must hold exactly hours 2..24, once each.
    seen = {row["hour"] for row in hours}
    for hour in range(2, 25):
        if hour not in seen:
            hours.append(quantile_row(
                p1, medians.get(hour, FALLBACK_MED[hour - 2]),
                spreads.get(hour, FALLBACK_SD), hour))
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def fallback_forecast(case):
    """Last resort: the fixed launch-day shape around the hour-1 print."""
    try:
        p1 = hour_one_print(case)
    except Exception:
        p1 = 1000.0
    p1 = min(max(p1, 1.0), 1.0e8)
    hours = [quantile_row(p1, FALLBACK_MED[hour - 2], FALLBACK_SD, hour)
             for hour in range(2, 25)]
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_forecast(case)
            except Exception:
                forecast = {"hours": [
                    {"hour": h, "q": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]}
                    for h in range(2, 25)
                ]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
