# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/daviddika/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/daviddika/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour (2–24, UTC) of its release day, 2026-11-19, from everything recorded up to the end of the day's first hour — as five quantiles (5/25/50/75/95%) per hour. Built for the make.fun bounty `gta6-twitch-ccu-forecast-01`.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the input directory and writes `<id>.out` for each.

## Method

Anchor on the hour-1 print. For each later hour, predict its log-ratio to hour 1 with a kernel-weighted linear regression fitted at run time on the case's own comparables:

- Only past launches that unlocked at rolling local midnight (like GTA VI) and were not early-access contaminated are used; simultaneous-global launches and free-to-play / slow-burn launches are excluded, since their release-day curves have a different shape.
- The regression is on `ln(hour-1 print) - 11`, weighted by a Gaussian kernel around the case's own hour-1 size, so big launches (whose hour 1 already holds most of the world) get a flatter growth curve than small ones. The kernel centre is clipped to the range of observed launches, so a record print saturates instead of extrapolating into a collapse.
- Quantile spreads come from the weighted residual spread of the same fit, shrunk by a factor tuned on the practice cases and on leave-one-out validation across the comparables.

The program never crashes and always emits a valid forecast: a fixed launch-day shape is the fallback when the fit cannot run, and a persistence forecast is the last resort.

## Validation (practice cases, answers from SullyGnome)

| entry | mean pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| this entry | **0.157** | 26.6% | 0.88 |
| house persistence baseline | 0.387 | 90.2% | 0.78 |

Per case: p00 0.180, p01 0.146, p02 0.117, p03 0.186 (baseline: 0.744, 0.287, 0.215, 0.303). Leave-one-out over the rolling-midnight comparables with hour-1 print ≥ 100k: 0.199. Runtime ~0.1 s per case; two runs on the same case are byte-identical.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2vjd5xyx&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
