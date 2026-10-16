# Methodology & Parameter Specification: Synthetic Urban Transit Simulation Benchmark v2

**Dataset Designation:** Synthetic Urban Transit Simulation Benchmark v2 (`transit_v2/`)  
**Random Seed:** `42` (`numpy.random.RandomState(42)`)  
**Scope:** Replaces legacy uncalibrated delay values in GTFS-style multi-city transit benchmark while strictly preserving schema, row counts, identifiers, routes, and station topologies.  
**Academic Disclaimer:** All transit metrics in `transit_v2/` represent a synthetic engineering simulation benchmark developed for multimodal machine learning validation; they are never presented as real-world empirical municipal accuracy.

---

## 1. Simulation Mathematical Formulation

For each scheduled transit trip $i$, continuous schedule delay $D_i$ (in minutes) is formulated as:

$$D_i = \begin{cases} 
0 & \text{with probability } p_{0, i} \\
\max\left(1, \left\lfloor D_{\text{det}, i} + \epsilon_i \right\rceil\right) & \text{with probability } 1 - p_{0, i}
\end{cases}$$

where $D_{\text{det}, i}$ is the deterministic operational expectation, $\epsilon_i \sim \text{Gamma}(\alpha=2.0, \beta=1.4)$ represents right-skewed positive schedule disruption, and $p_{0, i}$ is the zero-inflation probability.

---

## 2. Deterministic Component Breakdown ($D_{\text{det}, i}$)

$$D_{\text{det}, i} = \mu_{\text{corridor}} + \Delta_{\text{peak}} + \Delta_{\text{dwell}} + \Delta_{\text{headway}} + \left(\Delta_{\text{weather}} + \Delta_{\text{temp}}\right) \times M_{\text{mode}}$$

### 2.1 Corridor Distance Baseline ($\mu_{\text{corridor}}$)
Longer transit lines accumulate headway variance and signal delay across successive intersections:
- **Bus:** $\mu_{\text{corridor}} = \text{clip}(0.4 + 0.02 \times L_{\text{km}}, 0.5, 3.0)$ mins.
- **Train / Rail:** $\mu_{\text{corridor}} = \text{clip}(0.3 + 0.01 \times L_{\text{km}}, 0.4, 2.0)$ mins.
- **Metro:** $\mu_{\text{corridor}} = \text{clip}(0.15 + 0.005 \times L_{\text{km}}, 0.2, 1.0)$ mins.
*(Modeling assumption: Bus delays scale with route length due to mixed traffic exposure; Metro has fixed platform intervals).*

### 2.2 Peak Hour Congestion ($\Delta_{\text{peak}}$)
Defined for morning rush (07:00–09:59) and evening rush (16:00–19:59):
- **Bus:** $+2.5$ mins (mixed traffic road bottlenecks and signal queue spillbacks).
- **Train / Rail:** $+1.0$ mins (interlocking throat track capacity constraints at terminal stations).
- **Metro:** $+0.3$ mins (automated headway regulation with minimal track congestion).

### 2.3 Passenger Crowding & Platform Dwell Time ($\Delta_{\text{dwell}}$)
Passenger boarding and alighting times increase non-linearly with vehicle occupancy rate ($\rho$):
$$\Delta_{\text{dwell}} = \max(0, (\rho - 0.65) \times 4.0) + \mathbb{I}(\rho > 0.88) \times 1.2$$
*(Modeling assumption: Passenger dwell delay follows transit capacity studies [TCRP Report 165], where door blocking and vestibule friction surge above 85% capacity).*

### 2.4 Headway Spacing ($\Delta_{\text{headway}}$)
Long intervals between dispatches create passenger bunching at intermediate stops:
$$\Delta_{\text{headway}} = \max(0, (H_{\text{min}} - 12.0) \times 0.05)$$

### 2.5 Meteorological Hierarchy ($\Delta_{\text{weather}}$)
Weather penalties enforce a strict physical severity ordering:
- `clear`: $0.0$ min baseline
- `cloudy`: $+0.3$ min (nominal visibility reduction)
- `fog`: $+1.0$ min (reduced line-of-sight requiring cautious driving)
- `light_rain`: $+1.5$ min (wet road traction reduction)
- `heavy_rain`: $+3.5$ min (localized surface ponding, windshield wiping speed restrictions)
- `snow`: $+5.0$ min (snow clearing, reduced adhesion)
- `storm` (Severe Thunderstorm): $+7.5$ min (lightning protocols, flash pooling, heavy traffic gridlock)

### 2.6 Modal Weather Scaling ($M_{\text{mode}}$)
Reflects physical right-of-way isolation:
- **Bus ($M = 1.25$):** Full exposure to road gridlock, surface hydroplaning, and street flooding.
- **Train / Rail ($M = 0.60$):** Dedicated steel-wheel-on-rail tracks; affected by catenary wind limits and track switch freezing.
- **Metro ($M = 0.15$):** Underground tunnel grade separation; virtually immune to surface atmospheric conditions.

### 2.7 Ambient Temperature Extremes ($\Delta_{\text{temp}}$)
$$\Delta_{\text{temp}} = \max(0, -T \times 0.15) + \max(0, (T - 32.0) \times 0.20)$$
*(Modeling assumption: Freezing temperatures $<0^\circ\text{C}$ cause rail point heating delays and pneumatic door freezing; extreme heat $>32^\circ\text{C}$ triggers thermal speed restrictions and HVAC alternator load).*

---

## 3. Zero-Inflation & Stochastic Tail Formulation

Zero-delay probability $p_{0}$ reflects transit timetable recovery slack:
$$p_{0} = \text{clip}\left(p_{\text{base}} - 0.04 \times D_{\text{det}}, 0.30, 0.96\right)$$
where $p_{\text{base}} = 0.82$ (Bus), $0.88$ (Train), $0.94$ (Metro).

When a delay occurs ($\text{delay} > 0$):
- Stochastic variance: $\epsilon_i \sim \text{Gamma}(\text{shape}=2.0, \text{scale}=1.4) \times M_{\text{noise}}$, where $M_{\text{noise}} = 1.15$ (Bus), $0.85$ (Train), $0.60$ (Metro).
- Rare cascading dispatch failure: with probability $0.005$ during adverse weather, an exponential delay shock $\text{Exp}(\lambda=1/12)$ is added.

---

## 4. Leakage-Free Historical Mean Pipeline

In `transit_v2/`, `delay_24h_mean` (mapped to `historical_delay_mean`) is computed strictly from `train.csv`:
$$\mu_{\text{route}} = \frac{1}{|S_{\text{train}, r}|} \sum_{i \in S_{\text{train}, r}} D_i$$
This pre-computed dictionary is subsequently mapped onto `val.csv` and `test (1).csv`, ensuring **zero future target leakage**.

---

## 5. Derived Column Synchronization

To maintain strict schema consistency (88 columns):
1. `actual_travel_min = scheduled_travel_min + delay_minutes`
2. `on_time_flag = 1` if `delay_minutes <= 5` else `0`
3. `service_status = 'on_time'` (delay $\le 5$), `'minor_delay'` ($6 \le \text{delay} \le 20$), `'major_delay'` ($\text{delay} > 20$)
4. `reliability_score = clip(1.0 - delay_minutes / 25.0, 0.1, 1.0)`
5. `consistency_score = clip(1.0 - delay_minutes / 35.0, 0.1, 1.0)`
