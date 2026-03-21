# Simulation fidelity, scope, and limitations

## Purpose of this document

This document describes **what the V2X Smart Handoff Simulator (research edition) models**, lists **channel and evaluation parameters** that match the code, and states **limitations** so paper claims stay aligned with implementation. Use the boilerplate below in *Limitations*, *Simulation setup*, or *Discussion*; it is not internal author notes.

The study is a **controlled geometric simulation** (explicit, reproducible rules)—**not** a claim about field V2X performance. Wording fits titles and abstracts that say **“simulated.”**

---

## Wording for papers or reports (plain text)

You can paste this into a “Limitations,” “Simulation setup,” or “Discussion” subsection:

Our evaluation uses a geometric 2D channel model with log-distance path loss, binary building shadowing, and heuristic moving-blocker attenuation. While this captures key V2X phenomena (blockage, Doppler, handoff triggers), it does not implement full 3GPP TR 38.901 channel modeling or map-based mobility. Consequently, our results demonstrate *algorithmic relative performance* under controlled conditions, not absolute field performance. Future work will validate against ns-3/Simu5G packet-level simulation and real-world V2X traces.

**Optional extra sentence:** The implementation is discrete-time, with hysteresis and time-to-trigger–style handoffs; the proposed policy adds short-horizon predicted RSS and survival-style scoring (see `src/main.py`).

---

## One-sentence positioning (abstract / intro)

We evaluate the proposed handoff policy in a **simulated** vehicular-style network with geometric blockage and a log-distance channel; results characterize **relative** algorithmic behavior under controlled scenarios, not calibrated field performance.

---

## Channel model parameters

Numbers match **`src/main.py`** and **`config.py`**.

- **Path loss exponent:** η = 2.4 (urban micro, non-LOS typical)
- **Reference loss:** PL(d₀) = 32.44 dB at d₀ = 1 m, carrier **f** = 5.9 GHz
- **Shadowing:** Fixed 20 dB when the tower–UE LOS segment intersects a building rectangle
- **Dynamic blocker loss:** Class-dependent base loss and jitter (dB); see table below and scenario-specific `MovingBlocker` entries in `src/main.py`
- **Fading:** Optional Rayleigh/Rician via `config.enable_small_scale_fading` (Rician **K** = 10 when enabled)—**disabled by default** for reproducibility

**Also in code:** TX power 30 dBm; evaluation outage RSS threshold −95 dBm; map scale 0.1 m per pixel.

---

## Dynamic blocker classes (reference table)

| Class       | Base loss (dB) | Jitter scale (dB) |
|------------|----------------|-------------------|
| pedestrian | 7.2–9.0        | 2.0–2.4           |
| bus        | 12.8–13.5      | 3.2–3.8           |
| truck      | 15.0–16.0      | 4.4–4.8           |

Values are examples from urban canyon / intersection scenarios. The implemented drop also uses a harmonic term and Gaussian noise scaled by weather entropy (`blocker_shadow_drop` in `src/main.py`).

---

## Key parameters at a glance

| Quantity | Value |
|----------|--------|
| Carrier frequency | 5.9 GHz |
| Path loss exponent η | 2.4 |
| PL(d₀) at d₀ = 1 m | 32.44 dB |
| Building LOS blockage | −20 dB |
| TX power | 30 dBm |
| Outage threshold (evaluation) | −95 dBm |
| Small-scale fading | Optional; **off by default** |
| Rician K (if fading on) | 10 |

---

## Repo alignment

| What you state | Where in code |
|----------------|---------------|
| η, PL(d₀), f, TX | `PATH_LOSS_EXPONENT`, `REFERENCE_LOSS_DB`, `FREQ_HZ`, `TX_POWER_DBM` |
| Building shadow | `SHADOWING_DROP_DB` |
| Fading default off | `config.enable_small_scale_fading` |
| Outage threshold | `OUTAGE_RSS_THRESHOLD_DBM` |

Keep this file in sync with **`main.py`** / **`config.py`** if you change constants.
