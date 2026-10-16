# Synthetic Urban Transit Simulation Benchmark v2 Methodology

## 1. Executive Summary & Provenance
The **Synthetic Urban Transit Simulation Benchmark v2** (`transit_v2/`) is an academically rigorous, mode-specific physics simulation dataset designed to model urban multi-modal transport delays. 

- **Integrity**: Original source files in `transit/` remain strictly untouched.
- **Structural Invariance**: In `transit_v2/`, all 88 original columns, column headers, column ordering, row counts (Train: 455,520; Val: 56,940; Test: 56,940), trip IDs, vehicle IDs, routes, station stops, and timestamps are 100% preserved.
- **Values Regenerated**: Only delay-derived target columns (`delay_minutes`, `actual_travel_min`, `on_time_flag`, `service_status`, `reliability_score`, `consistency_score`, and corridor rolling means) were regenerated.
- **Leakage Prevention**: All historical baseline features (`delay_24h_mean`, `delay_6h_mean`, `delay_7d_mean`, `delay_cumsum_4h`) are computed strictly from the **training split only**, ensuring zero temporal data leakage into validation or test splits.
- **Academically Honest Labeling**: Labeled unambiguously as a synthetic benchmark throughout the application, documentation, and user interface. Never represented as empirical operational telemetry.

---

## 2. Domain Physics & Mathematical Formulation

Total simulated trip delay is generated via a two-stage zero-inflated semi-parametric formulation:

$$\text{Delay} = \mathbb{I}(\text{Delayed}) \times \left[ \text{Deterministic Components} + \text{Interactions} + \epsilon_{\text{stochastic}} + \text{Shock} \right]$$

### 2.1 Deterministic Mode Components

1. **Corridor Baseline & Route Duration Effect**:
   - **Bus**: Baseline delay scales moderately with corridor length:
     $$\text{Base}_{\text{bus}} = 0.50 + 0.025 \cdot \text{length\_km}$$
   - **Train / Rail**: Substantial route duration effect where long-haul journeys exhibit compounded dispatching buffers and large delay tails:
     $$\text{Base}_{\text{rail}} = 0.40 + 0.010 \cdot \text{length\_km} + 0.012 \cdot \text{duration\_min} + 0.025 \cdot \max(0, \text{duration\_min} - 60.0)$$
   - **Metro**: Fixed, grade-separated, protected right-of-way with minimal distance scaling:
     $$\text{Base}_{\text{metro}} = 0.40 + 0.010 \cdot \text{length\_km}$$

2. **Peak Hour Traffic Congestion**:
   - Morning peak: $7 \le \text{hour} \le 9$; Evening peak: $16 \le \text{hour} \le 19$.
   - **Bus**: Highly vulnerable to surface traffic (+3.5 min morning, +4.0 min evening).
   - **Train**: Moderate interlock bottlenecking (+1.6 min peak).
   - **Metro**: High turnstile volume (+2.2 min morning, +2.5 min evening).

3. **Passenger Crowding & Dwell Time Penalties**:
   - Boarding / alighting delays driven by vehicle occupancy rate ($occ$):
     $$\text{Dwell}_{\text{bus}} = 5.5 \cdot \max(0, occ - 0.45) + 3.0 \cdot \mathbb{I}(occ > 0.85)$$
     $$\text{Dwell}_{\text{rail}} = 3.8 \cdot \max(0, occ - 0.50) + 2.0 \cdot \mathbb{I}(occ > 0.80)$$
     $$\text{Dwell}_{\text{metro}} = 4.5 \cdot \max(0, occ - 0.50) + 3.0 \cdot \mathbb{I}(occ > 0.80)$$

4. **Service Headway Spacing & Bunching**:
   - **Bus**: Low headway ($< 5\text{ min}$) triggers bunching instabilities ($+0.6 \cdot (5 - hw)$); long headways cause overcrowding (+0.20 min).
   - **Metro**: Highly sensitive to headway variance ($+0.65 \cdot (hw - 4) + 2.5 \cdot \mathbb{I}(hw > 8)$).

5. **Meteorological Hierarchy**:
   - Physical separation dictates weather impact: Metro is subterranean and grade-separated, hence weakly weather-sensitive by design. Surface rail and bus suffer adhesion loss, speed restrictions, and road congestion:
   - **Clear**: Bus: 0.0, Rail: 0.0, Metro: 0.0
   - **Cloudy**: Bus: +0.5, Rail: +0.4, Metro: 0.0
   - **Fog**: Bus: +1.6, Rail: +1.5, Metro: 0.0
   - **Light Rain**: Bus: +2.5, Rail: +2.4, Metro: +0.2
   - **Heavy Rain**: Bus: +5.5, Rail: +5.5, Metro: +0.5
   - **Snow / Ice**: Bus: +8.0, Rail: +8.0, Metro: +0.6
   - **Severe Storm**: Bus: +12.0, Rail: +12.0, Metro: +1.2

6. **Ambient Temperature Extremes**:
   - Freezing conditions ($< 0^\circ\text{C}$): Bus $+0.16/^\circ\text{C}$, Rail $+0.18/^\circ\text{C}$, Metro $+0.03/^\circ\text{C}$.
   - Extreme heat ($> 32^\circ\text{C}$): Track buckling speed restrictions on Rail $+0.18/^\circ\text{C}$.

7. **Cross-Feature Interactions**:
   - Peak hour $\times$ high occupancy: $+2.0\text{ min}$ when peak and $occ > 0.70$.
   - Storm $\times$ peak congestion: $+3.0\text{ min}$ when storm during peak.

### 2.2 Natural Dwell Variation & Stochastic Dispersion
- Probability of zero delay:
  $$P(\text{delay} = 0) = \text{clip}\left(P_{\text{base}} - 0.035 \cdot \text{Deterministic Delay}, 0.08, 0.72\right)$$
  where $P_{\text{base}}$ is 0.42 (Bus), 0.55 (Rail), and 0.48 (Metro).
- Noise: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ where $\sigma_{\text{bus}}=1.8$, $\sigma_{\text{rail}}=1.8$, $\sigma_{\text{metro}}=1.0$.
- Exogenous shock tail: with probability 1.8% to 5.2%, an incident adds $20 + \text{Exp}(6.5)\text{ min}$.
- Generator target $R^2 \approx 0.45 - 0.55$, ensuring learning models cannot achieve artificial 100% accuracy.

---

## 3. Empirical Distribution & Validation

### 3.1 Target Class Mix Across Splits
| Mode | Split | On-Time ($\le 5\text{m}$) | Moderate ($5-20\text{m}$) | Severe ($>20\text{m}$) | Mean Delay | P90 Delay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bus** | Train | 79.5% | 17.0% | 3.5% | 3.26 min | 9.0 min |
| **Bus** | Val | 75.4% | 20.9% | 3.7% | 3.75 min | 11.0 min |
| **Bus** | Test | 75.1% | 21.4% | 3.4% | 3.69 min | 11.0 min |
| **Rail** | Train | 68.9% | 27.4% | 3.7% | 4.35 min | 12.0 min |
| **Rail** | Val | 67.3% | 28.9% | 3.7% | 4.55 min | 12.0 min |
| **Rail** | Test | 67.7% | 28.5% | 3.8% | 4.44 min | 12.0 min |
| **Metro** | Train | 95.5% | 3.6% | 0.9% | 1.05 min | 3.0 min |
| **Metro** | Val | 93.9% | 5.3% | 0.8% | 1.17 min | 4.0 min |
| **Metro** | Test | 94.6% | 4.5% | 0.9% | 1.08 min | 4.0 min |

### 3.2 Empirical Delay Response by Weather (Train Split)
| Mode | Weather | Sample Count | Mean Delay (min) | $P(\text{delay} > 0)$ | $P(\text{delay} > 5)$ | $P(\text{delay} > 20)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bus** | Clear | 47,918 | 2.18 | 38.1% | 10.8% | 3.1% |
| **Bus** | Cloudy | 35,480 | 2.35 | 39.4% | 12.3% | 3.2% |
| **Bus** | Fog | 6,356 | 2.92 | 43.8% | 18.2% | 3.4% |
| **Bus** | Light Rain | 18,971 | 3.38 | 47.1% | 23.0% | 3.5% |
| **Bus** | Heavy Rain | 8,630 | 5.69 | 58.8% | 49.4% | 3.7% |
| **Bus** | Snow | 5,029 | 7.98 | 69.1% | 67.8% | 3.4% |
| **Bus** | Storm | 3,760 | 13.53 | 83.0% | 83.0% | 11.6% |
| **Rail** | Clear | 50,523 | 2.84 | 44.5% | 18.7% | 2.9% |
| **Rail** | Cloudy | 37,442 | 3.10 | 45.8% | 21.6% | 3.0% |
| **Rail** | Fog | 6,586 | 3.87 | 50.6% | 31.2% | 2.8% |
| **Rail** | Light Rain | 19,875 | 4.55 | 54.3% | 38.8% | 3.0% |
| **Rail** | Heavy Rain | 9,379 | 8.06 | 69.5% | 67.3% | 3.1% |
| **Rail** | Snow | 5,414 | 11.40 | 80.1% | 80.0% | 3.2% |
| **Rail** | Storm | 3,933 | 17.02 | 87.8% | 87.8% | 27.8% |
| **Metro** | Clear | 50,513 | 0.98 | 26.8% | 3.8% | 0.9% |
| **Metro** | Cloudy | 37,321 | 0.98 | 27.2% | 3.9% | 0.9% |
| **Metro** | Fog | 6,696 | 0.98 | 27.8% | 4.1% | 0.7% |
| **Metro** | Light Rain | 19,933 | 1.07 | 28.0% | 4.6% | 0.9% |
| **Metro** | Heavy Rain | 9,328 | 1.16 | 28.9% | 5.2% | 0.9% |
| **Metro** | Snow | 5,315 | 1.14 | 28.3% | 5.1% | 0.8% |
| **Metro** | Storm | 4,046 | 2.05 | 35.3% | 17.5% | 1.1% |

### 3.3 Empirical Delay Response by Time of Day (Hour Band)
| Mode | Hour Band | Sample Count | Mean Delay (min) | $P(\text{delay} > 0)$ | $P(\text{delay} > 5)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bus** | Midday (10-15) | 31,536 | 2.53 | 39.5% | 13.9% |
| **Bus** | Morning Peak (7-9) | 15,768 | 4.72 | 53.4% | 33.5% |
| **Bus** | Evening Peak (16-19) | 21,024 | 5.10 | 54.6% | 38.2% |
| **Rail** | Midday (10-15) | 33,288 | 3.94 | 49.1% | 27.2% |
| **Rail** | Morning Peak (7-9) | 16,644 | 5.42 | 56.2% | 40.6% |
| **Rail** | Evening Peak (16-19) | 22,192 | 5.39 | 56.3% | 40.6% |
| **Metro** | Midday (10-15) | 33,288 | 0.83 | 25.5% | 2.6% |
| **Metro** | Morning Peak (7-9) | 16,644 | 1.51 | 32.4% | 8.9% |
| **Metro** | Evening Peak (16-19) | 22,192 | 1.62 | 33.6% | 9.4% |
