/**
 * Multimodal Transportation Delay Prediction System
 * Frontend Controller & Dynamic UI Engine
 * Populates ALL routes, stations, airlines, and airports STRICTLY from real datasets
 * Integrates live Open-Meteo meteorological telemetry for urban & airport hubs
 * Manages prediction history, risk indicators, and dynamic model outputs
 */

// Global Metadata Cache
let appMetadata = null;

// Combobox instances registry
const comboboxes = {};

document.addEventListener('DOMContentLoaded', async () => {
  const isPredictionPage = !!document.getElementById('delay-prediction-form');
  if (!isPredictionPage) {
    return;
  }

  initComboboxClickOutside();
  initPredictionHistory();

  try {
    await loadMetadata();
    initComboboxInstances();
    initModeSwitcher();
    initPredictionForm();
    initWeatherSyncButton();
  } catch (err) {
    console.error('Initialization halted due to metadata failure:', err);
  }
});

/**
 * Click-outside listener to close any open dropdowns
 */
function initComboboxClickOutside() {
  document.addEventListener('click', (e) => {
    Object.values(comboboxes).forEach(cb => {
      if (cb && cb.container && typeof cb.container.contains === 'function') {
        if (!cb.container.contains(e.target)) {
          cb.close();
        }
      }
    });
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      Object.values(comboboxes).forEach(cb => {
        if (cb && typeof cb.close === 'function') cb.close();
      });
    }
  });
}

/**
 * Searchable Combobox Component
 */
class SearchableCombobox {
  constructor(containerId, onSelectCallback = null) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.targetInputId = this.container.dataset.target;
    this.hiddenInput = document.getElementById(this.targetInputId);
    this.trigger = this.container.querySelector('.combo-trigger');
    this.triggerText = this.container.querySelector('.combo-trigger-text');
    this.dropdown = this.container.querySelector('.combo-dropdown');
    this.searchInput = this.container.querySelector('.combo-search-input');
    this.clearBtn = this.container.querySelector('.combo-clear-btn');
    this.optionsList = this.container.querySelector('.combo-options-list');
    this.onSelect = onSelectCallback;

    this.options = [];
    this.selectedValue = null;
    this.selectedOption = null;

    this.bindEvents();
  }

  bindEvents() {
    if (this.trigger) {
      this.trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.container) return;
        const isOpen = this.container.classList.contains('open');
        Object.values(comboboxes).forEach(cb => {
          if (cb !== this && cb && typeof cb.close === 'function') cb.close();
        });
        if (isOpen) {
          this.close();
        } else {
          this.open();
        }
      });

      this.trigger.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.open();
        }
      });
    }

    if (this.searchInput) {
      this.searchInput.addEventListener('input', (e) => {
        this.filterOptions(e.target.value);
      });

      this.searchInput.addEventListener('click', (e) => {
        e.stopPropagation();
      });
    }

    if (this.clearBtn) {
      this.clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.searchInput) {
          this.searchInput.value = '';
          this.filterOptions('');
          this.searchInput.focus();
        }
      });
    }
  }

  open() {
    if (!this.container) return;
    this.container.classList.add('open');
    if (this.searchInput) {
      this.searchInput.value = '';
      this.filterOptions('');
      setTimeout(() => {
        if (this.searchInput) this.searchInput.focus();
      }, 50);
    }
  }

  close() {
    if (!this.container) return;
    this.container.classList.remove('open');
  }

  setOptions(optionsList, defaultValue = null, placeholder = 'Select an option...') {
    this.options = optionsList || [];
    this.placeholder = placeholder;
    this.selectedValue = null;
    this.selectedOption = null;

    if (this.hiddenInput) {
      this.hiddenInput.value = '';
    }

    if (!this.options.length) {
      if (this.triggerText) {
        this.triggerText.textContent = placeholder;
        this.triggerText.classList.add('placeholder');
      }
      if (this.optionsList) {
        this.optionsList.innerHTML = `<div class="combo-empty">${placeholder}</div>`;
      }
      return;
    }

    this.renderOptions(this.options);

    if (defaultValue) {
      this.setValue(defaultValue, false);
    } else {
      if (this.triggerText) {
        this.triggerText.textContent = placeholder;
        this.triggerText.classList.add('placeholder');
      }
    }
  }

  renderOptions(list) {
    if (!this.optionsList) return;
    this.optionsList.innerHTML = '';
    if (!list.length) {
      this.optionsList.innerHTML = '<div class="combo-empty">No matching records found in dataset</div>';
      return;
    }

    list.forEach(opt => {
      const el = document.createElement('div');
      el.className = 'combo-option';
      if (opt.value === this.selectedValue) {
        el.classList.add('selected');
      }

      el.innerHTML = `
        <span class="combo-option-label" title="${opt.label}">${opt.label}</span>
        ${opt.badge ? `<span class="combo-option-badge">${opt.badge}</span>` : ''}
      `;

      el.addEventListener('click', (e) => {
        e.stopPropagation();
        this.setValue(opt.value, true);
        this.close();
      });

      this.optionsList.appendChild(el);
    });
  }

  filterOptions(query) {
    const q = (query || '').trim().toLowerCase();
    if (!q) {
      this.renderOptions(this.options);
      return;
    }

    const filtered = this.options.filter(opt => {
      const matchLabel = opt.label && opt.label.toLowerCase().includes(q);
      const matchVal = opt.value && opt.value.toLowerCase().includes(q);
      const matchBadge = opt.badge && opt.badge.toLowerCase().includes(q);
      return matchLabel || matchVal || matchBadge;
    });

    this.renderOptions(filtered);
  }

  setValue(value, triggerCallback = true) {
    const opt = this.options.find(o => o.value === value);
    if (opt) {
      this.selectedValue = opt.value;
      this.selectedOption = opt;
      if (this.triggerText) {
        this.triggerText.textContent = opt.label;
        this.triggerText.classList.remove('placeholder');
      }
      if (this.hiddenInput) {
        this.hiddenInput.value = opt.value;
      }
      if (triggerCallback && typeof this.onSelect === 'function') {
        this.onSelect(opt.value, opt);
      }
    } else {
      this.selectedValue = null;
      this.selectedOption = null;
      if (this.triggerText) {
        this.triggerText.textContent = this.placeholder || 'Select...';
        this.triggerText.classList.add('placeholder');
      }
      if (this.hiddenInput) {
        this.hiddenInput.value = '';
      }
    }
  }

  getValue() {
    return this.selectedValue;
  }

  getOption() {
    return this.selectedOption;
  }
}

let metadataLoadingPromise = null;

/**
 * Loads metadata from Flask API backed by real dataset files
 */
async function loadMetadata() {
  if (appMetadata) {
    return appMetadata;
  }
  if (metadataLoadingPromise) {
    return metadataLoadingPromise;
  }

  metadataLoadingPromise = (async () => {
    try {
      const response = await fetch('/api/routes-metadata', {
        headers: { Accept: 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Metadata API returned HTTP ${response.status}`);
      }

      const metadata = await response.json();

      if (
        !metadata ||
        typeof metadata !== 'object' ||
        !metadata.transit ||
        !metadata.airplane ||
        !metadata.modes
      ) {
        throw new Error('Metadata response has an unexpected structure');
      }

      appMetadata = metadata;
      console.info('Dataset metadata loaded successfully', {
        modes: metadata.modes,
        transitModes: Object.keys(metadata.transit),
        airlineCount: metadata.airplane.airlines?.length ?? 0
      });

      return metadata;
    } catch (error) {
      console.error('Dataset metadata loading failed:', error);
      showValidationAlert('Unable to load dataset metadata. Check the browser console for details.');
      throw error;
    } finally {
      metadataLoadingPromise = null;
    }
  })();

  return metadataLoadingPromise;
}

/**
 * Live Meteorological Telemetry via Open-Meteo API
 */
let lastLiveWeatherTelemetry = {};

async function fetchLiveWeather(targetLocation) {
  const badgeEl = document.getElementById('weather-live-badge');
  const detailsEl = document.getElementById('weather-live-details');
  if (badgeEl) {
    badgeEl.textContent = '📡 Contacting Open-Meteo...';
    badgeEl.className = 'badge';
  }

  try {
    const res = await fetch(`/api/live-weather?target=${encodeURIComponent(targetLocation || 'New_York')}`);
    const data = await res.json();

    const tempInput = document.getElementById('input_temperature_c');
    const weatherSelect = document.getElementById('input_weather_condition');

    if (tempInput && data.temperature_c !== undefined) {
      tempInput.value = data.temperature_c;
      lastLiveWeatherTelemetry.temperature_c = data.temperature_c;
    }
    if (weatherSelect && data.weather_condition) {
      weatherSelect.value = data.weather_condition;
      lastLiveWeatherTelemetry.weather_condition = data.weather_condition;
    }

    if (badgeEl) {
      if (data.is_live) {
        badgeEl.textContent = '🟢 LIVE METEOROLOGICAL TELEMETRY';
        badgeEl.className = 'badge badge-ontime';
      } else {
        badgeEl.textContent = '🔴 WEATHER TELEMETRY UNAVAILABLE';
        badgeEl.className = 'badge badge-severe';
      }
    }

    if (detailsEl) {
      if (data.is_live) {
        const condName = (data.weather_condition || 'clear').replace(/_/g, ' ').toUpperCase();
        const obsTimeStr = data.observed_at || 'Current UTC';
        const retTimeStr = data.retrieved_at || '';
        detailsEl.textContent = `${data.location} • ${data.temperature_c}°C • ${condName} • Observed: ${obsTimeStr}${retTimeStr ? ' • Retrieved: ' + retTimeStr : ''} (${data.source})`;
      } else {
        detailsEl.textContent = `${data.location || targetLocation} • Weather telemetry unavailable (${data.source || 'Offline / Timeout'}). Manual parameter selection active.`;
      }
    }

    checkMeteorologicalPlausibility();
  } catch (err) {
    console.warn('Live meteorological telemetry fetch error:', err);
    if (badgeEl) {
      badgeEl.textContent = '🔴 WEATHER TELEMETRY UNAVAILABLE';
      badgeEl.className = 'badge badge-severe';
    }
    if (detailsEl) {
      detailsEl.textContent = 'Open-Meteo telemetry connection error. Please select weather condition manually.';
    }
  }
}

/**
 * Validates meteorological and seasonal plausibility without altering user values.
 * Flags implausible conditions (e.g. 40.9°C storm in Berlin in October) transparently.
 */
function checkMeteorologicalPlausibility() {
  const alertBox = document.getElementById('meteorological-plausibility-alert');
  const alertText = document.getElementById('plausibility-alert-text');
  if (!alertBox || !alertText) return;

  const mode = document.getElementById('input_transport_mode')?.value || 'BUS';
  const weather = document.getElementById('input_weather_condition')?.value || 'clear';
  const tempVal = document.getElementById('input_temperature_c')?.value;
  const temp = parseFloat(tempVal);
  const month = parseInt(document.getElementById('input_month')?.value || '10', 10);

  if (isNaN(temp)) {
    alertBox.style.display = 'none';
    return;
  }

  let locName = 'Selected Hub';
  if (mode === 'AIRPLANE') {
    locName = comboboxes.flightOrigin ? (comboboxes.flightOrigin.getValue() || 'Airport') : 'Airport';
  } else {
    const city = document.getElementById('select_city')?.value || 'City';
    locName = city.replace(/_/g, ' ');
  }

  let isAnomalous = false;
  let reason = '';

  const euroCities = ['Berlin', 'London', 'Paris'];
  const isEuro = euroCities.some(c => locName.toLowerCase().includes(c.toLowerCase()));

  if (temp >= 40.0 && isEuro) {
    isAnomalous = true;
    reason = `Temperature of ${temp.toFixed(1)}°C in ${locName} is an extreme atmospheric outlier (all-time recorded high: ~38–41°C in mid-summer; improbable in autumn month ${month}).`;
  } else if (temp > 35.0 && month >= 10 && isEuro) {
    isAnomalous = true;
    reason = `Temperature of ${temp.toFixed(1)}°C is an extreme counterfactual condition for European autumn/winter (month ${month}).`;
  } else if ((weather === 'storm' || weather === 'heavy_rain') && temp < -5.0) {
    isAnomalous = true;
    reason = `Convective thunderstorms rarely develop in severe subzero environments (${temp.toFixed(1)}°C).`;
  } else if (weather === 'snow' && temp > 8.0) {
    isAnomalous = true;
    reason = `Snow precipitation is physically inconsistent with ambient temperature of ${temp.toFixed(1)}°C.`;
  }

  if (isAnomalous) {
    alertText.textContent = `⚠️ Meteorological Plausibility Advisory: ${reason} The trained ML models will evaluate this counterfactual stress test without modification.`;
    alertBox.style.display = 'flex';
  } else {
    alertBox.style.display = 'none';
  }
}

function initWeatherSyncButton() {
  const syncBtn = document.getElementById('btn-sync-weather');
  if (syncBtn) {
    syncBtn.addEventListener('click', () => {
      const mode = document.getElementById('input_transport_mode')?.value || 'BUS';
      if (mode === 'AIRPLANE') {
        const origAirport = comboboxes.flightOrigin ? comboboxes.flightOrigin.getValue() : 'ATL';
        fetchLiveWeather(origAirport || 'ATL');
      } else {
        const city = document.getElementById('select_city')?.value || 'New_York';
        fetchLiveWeather(city);
      }
    });
  }

  // Attach plausibility listeners on weather, temp, month, and city changes
  const tempInput = document.getElementById('input_temperature_c');
  if (tempInput) {
    tempInput.addEventListener('input', () => {
      checkMeteorologicalPlausibility();
      markManualOverride('input_temperature_c');
    });
    tempInput.addEventListener('change', checkMeteorologicalPlausibility);
  }

  const weatherSelect = document.getElementById('input_weather_condition');
  if (weatherSelect) {
    weatherSelect.addEventListener('change', () => {
      // NOTE: Changing weather condition MUST NOT alter temperature
      checkMeteorologicalPlausibility();
      markManualOverride('input_weather_condition');
    });
  }

  const monthSelect = document.getElementById('input_month');
  if (monthSelect) {
    monthSelect.addEventListener('change', checkMeteorologicalPlausibility);
  }

  const distInput = document.getElementById('input_distance_km');
  const durInput = document.getElementById('input_scheduled_duration_min');
  if (distInput) {
    distInput.addEventListener('input', checkSpeedPlausibility);
  }
  if (durInput) {
    durInput.addEventListener('input', checkSpeedPlausibility);
  }
}

function markManualOverride(inputId) {
  const label = document.querySelector(`label[for="${inputId}"]`);
  if (label && !label.querySelector('.badge-override')) {
    const badge = document.createElement('span');
    badge.className = 'badge badge-override';
    badge.style.fontSize = '0.7rem';
    badge.style.marginLeft = '0.5rem';
    badge.style.background = 'rgba(245, 158, 11, 0.2)';
    badge.style.color = '#fbbf24';
    badge.textContent = '(manual override)';
    label.appendChild(badge);
  }
}

function checkSpeedPlausibility() {
  const alertBox = document.getElementById('speed-plausibility-alert');
  const alertText = document.getElementById('speed-alert-text');
  if (!alertBox || !alertText) return;

  const mode = document.getElementById('input_transport_mode')?.value || 'BUS';
  const dist = parseFloat(document.getElementById('input_distance_km')?.value || '0');
  const dur = parseFloat(document.getElementById('input_scheduled_duration_min')?.value || '0');

  if (isNaN(dist) || isNaN(dur) || dist <= 0 || dur <= 0) {
    alertBox.style.display = 'none';
    return;
  }

  const speed = dist / (dur / 60.0);
  let isViolated = false;
  let msg = '';

  if (mode === 'BUS' && speed > 70.0) {
    isViolated = true;
    msg = `Speed Advisory: Implied bus speed of ${speed.toFixed(1)} km/h exceeds normal urban bus operating limits (70 km/h). (Flagged, never silently altered).`;
  } else if (mode === 'METRO' && speed > 90.0) {
    isViolated = true;
    msg = `Speed Advisory: Implied metro speed of ${speed.toFixed(1)} km/h exceeds typical metro operating limits (90 km/h). (Flagged, never silently altered).`;
  } else if (mode === 'TRAIN' && speed > 200.0) {
    isViolated = true;
    msg = `Speed Advisory: Implied train speed of ${speed.toFixed(1)} km/h exceeds conventional rail operating limits (200 km/h). (Flagged, never silently altered).`;
  } else if (mode === 'AIRPLANE' && (speed < 300.0 || speed > 950.0)) {
    isViolated = true;
    msg = `Speed Advisory: Implied flight speed of ${speed.toFixed(1)} km/h is outside commercial jet cruise envelope (300–950 km/h). (Flagged, never silently altered).`;
  }

  if (isViolated) {
    alertText.textContent = msg;
    alertBox.style.display = 'flex';
  } else {
    alertBox.style.display = 'none';
  }
}

/**
 * Instantiates Comboboxes
 */
function initComboboxInstances() {
  if (!appMetadata) return;

  // Transit Comboboxes
  comboboxes.transitRoute = new SearchableCombobox('combo-transit-route', (val, opt) => {
    onTransitRouteSelected(opt);
  });

  comboboxes.transitOrigin = new SearchableCombobox('combo-transit-origin', (val, opt) => {
    onTransitOriginSelected(opt);
  });

  comboboxes.transitDest = new SearchableCombobox('combo-transit-dest', (val, opt) => {
    onTransitDestSelected(opt);
  });

  // Aviation Comboboxes
  comboboxes.flightAirline = new SearchableCombobox('combo-flight-airline', (val, opt) => {
    onFlightAirlineSelected(val);
  });

  comboboxes.flightOrigin = new SearchableCombobox('combo-flight-origin', (val, opt) => {
    onFlightOriginSelected(val);
  });

  comboboxes.flightDest = new SearchableCombobox('combo-flight-dest', (val, opt) => {
    onFlightDestSelected(val);
  });

  // Transit City dropdown listener
  const citySelect = document.getElementById('select_city');
  if (citySelect) {
    citySelect.addEventListener('change', () => {
      onTransitCitySelected(citySelect.value);
    });
  }

  // Initial trigger for default mode (BUS)
  applyMode('BUS');
}

/**
 * Initializes Mode Buttons & Mode Dropdown
 */
function initModeSwitcher() {
  const modeButtons = document.querySelectorAll('.mode-btn');
  const modeDropdown = document.getElementById('select-mode-dropdown');

  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      applyMode(mode);
    });
  });

  if (modeDropdown) {
    modeDropdown.addEventListener('change', () => {
      applyMode(modeDropdown.value);
    });
  }
}

/**
 * Applies Mode configuration dynamically
 */
function applyMode(mode) {
  hideValidationAlert();

  // Clear manual override indicators on mode change
  const weatherLabel = document.querySelector('label[for="input_weather_condition"]');
  if (weatherLabel) {
    const overrideBadge = weatherLabel.querySelector('.badge-override');
    if (overrideBadge) overrideBadge.remove();
  }
  const tempLabel = document.querySelector('label[for="input_temperature_c"]');
  if (tempLabel) {
    const overrideBadge = tempLabel.querySelector('.badge-override');
    if (overrideBadge) overrideBadge.remove();
  }

  // Sync Hidden Input
  const modeInput = document.getElementById('input_transport_mode');
  if (modeInput) modeInput.value = mode;

  // Sync Buttons
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });

  // Sync Dropdown
  const modeDropdown = document.getElementById('select-mode-dropdown');
  if (modeDropdown && modeDropdown.value !== mode) {
    modeDropdown.value = mode;
  }

  // Update Headings
  const headings = {
    BUS: '🚌 Bus Dispatch Parameters',
    TRAIN: '🚆 Train / Rail Dispatch Parameters',
    METRO: '🚇 Metro Rapid Transit Parameters',
    AIRPLANE: '✈️ Commercial Flight Dispatch Parameters'
  };
  const headingEl = document.getElementById('form-heading');
  if (headingEl) {
    headingEl.textContent = headings[mode] || `${mode} Dispatch Parameters`;
  }

  const transitSection = document.getElementById('transit-location-fields');
  const airplaneSection = document.getElementById('airplane-location-fields');
  const headwayGroup = document.getElementById('group-headway');
  const distLabel = document.getElementById('lbl-distance');
  const distHint = document.getElementById('hint-distance');
  const histDelayLabel = document.getElementById('lbl-hist-delay');
  const locationTitle = document.getElementById('location-section-title');
  const helperText = document.getElementById('location-helper-text');

  if (mode === 'AIRPLANE') {
    if (transitSection) transitSection.style.display = 'none';
    if (airplaneSection) airplaneSection.style.display = 'block';
    if (headwayGroup) headwayGroup.style.display = 'none';

    if (distLabel) distLabel.textContent = 'Flight Great-Circle Distance (km)';
    if (distHint) distHint.textContent = 'Actual distance between origin and destination airport in km';
    if (histDelayLabel) histDelayLabel.textContent = 'Carrier & Origin Airport Historical Delay Baseline (mean mins)';
    if (locationTitle) locationTitle.innerHTML = '<span>✈️</span> Flight Carrier & Airport Route Selection';
    if (helperText) helperText.textContent = 'Cascade: Airline → Origin Airport → Destination Airport';

    const capHint = document.querySelector('#group-capacity .field-hint');
    const occHint = document.querySelector('#group-occupancy .field-hint');
    if (capHint) capHint.textContent = 'Commercial DOT dataset fleet reference (180 seats)';
    if (occHint) occHint.textContent = 'Commercial DOT dataset fleet load factor (0.82 ref)';

    populateAviationAirlines();
  } else {
    if (transitSection) transitSection.style.display = 'block';
    if (airplaneSection) airplaneSection.style.display = 'none';
    if (headwayGroup) headwayGroup.style.display = 'flex';

    if (distLabel) distLabel.textContent = 'Route Length (km)';
    if (distHint) distHint.textContent = 'Actual length of selected transit route in km';
    if (histDelayLabel) histDelayLabel.textContent = 'Prior Transit Delay Baseline (mean mins)';
    if (locationTitle) locationTitle.innerHTML = '<span>📍</span> Transit Route & Station Stop Selection';
    if (helperText) helperText.textContent = 'Cascade: City → Route ID → Origin Stop → Destination Stop';

    const capHint = document.querySelector('#group-capacity .field-hint');
    const occHint = document.querySelector('#group-occupancy .field-hint');
    if (capHint) capHint.textContent = 'Design payload capacity';
    if (occHint) occHint.textContent = 'Percentage of capacity occupied (e.g., 0.65 = 65%)';

    populateTransitCities(mode);
  }
}

/**
 * =========================================================================
 * TRANSIT DROPDOWN CASCADE (BUS / TRAIN / METRO)
 * Mode → City → Route ID → Origin Stop → Destination Stop
 * =========================================================================
 */
function populateTransitCities(mode) {
  if (!appMetadata || !appMetadata.transit || !appMetadata.transit[mode]) return;

  const modeData = appMetadata.transit[mode];
  if (!modeData.cities) return;
  const cities = Object.keys(modeData.cities).sort();
  const citySelect = document.getElementById('select_city');
  if (!citySelect) return;
  citySelect.innerHTML = '';

  if (!cities.length) {
    citySelect.innerHTML = '<option value="" disabled selected>No cities available</option>';
    return;
  }

  cities.forEach(city => {
    const opt = document.createElement('option');
    opt.value = city;
    opt.textContent = city.replace(/_/g, ' ');
    citySelect.appendChild(opt);
  });

  // Select first city and cascade
  citySelect.value = cities[0];
  onTransitCitySelected(cities[0]);
}

function onTransitCitySelected(city) {
  const modeInput = document.getElementById('input_transport_mode');
  const mode = modeInput ? modeInput.value : 'BUS';
  if (!appMetadata || !appMetadata.transit || !appMetadata.transit[mode]) return;

  // Sync live weather for this city
  fetchLiveWeather(city);

  const cityData = appMetadata.transit[mode].cities[city];
  if (!cityData || !cityData.routes || !cityData.routes.length) {
    if (comboboxes.transitRoute) comboboxes.transitRoute.setOptions([], null, 'No routes in this city');
    if (comboboxes.transitOrigin) comboboxes.transitOrigin.setOptions([], null, 'No origin station');
    if (comboboxes.transitDest) comboboxes.transitDest.setOptions([], null, 'No destination station');
    return;
  }

  // Build Route Options
  const routeOptions = cityData.routes.map(r => ({
    value: r.route_id,
    label: r.route_id,
    badge: 'CORRIDOR',
    meta: r
  }));

  if (comboboxes.transitRoute) {
    comboboxes.transitRoute.setOptions(routeOptions, routeOptions[0]?.value, 'Select Route ID...');
  }
  if (routeOptions[0]) {
    onTransitRouteSelected(routeOptions[0]);
  }
}

function onTransitRouteSelected(routeOption) {
  if (!routeOption || !routeOption.meta) return;
  const routeData = routeOption.meta;
  const modeInput = document.getElementById('input_transport_mode');
  const currentMode = modeInput ? modeInput.value : 'BUS';

  // Offer all valid terminal stops on this corridor in the origin dropdown
  const stops = routeData.stops && routeData.stops.length ? routeData.stops : [routeData.origin, routeData.destination].filter(Boolean);

  const originOptions = stops.map((s, idx) => ({
    value: s,
    label: s,
    badge: idx === 0 ? 'TERMINUS A' : 'TERMINUS B',
    meta: { stop: s, routeData }
  }));

  if (comboboxes.transitOrigin) {
    comboboxes.transitOrigin.setOptions(originOptions, originOptions[0]?.value, 'Select Origin Station...');
  }
  if (originOptions[0]) {
    onTransitOriginSelected(originOptions[0]);
  }

  // Auto-populate operational parameters from actual route in dataset
  const distEl = document.getElementById('input_distance_km');
  if (distEl && routeData.dist_km !== undefined) distEl.value = routeData.dist_km;
  const durEl = document.getElementById('input_scheduled_duration_min');
  if (durEl && routeData.dur_min !== undefined) durEl.value = routeData.dur_min;
  const headwayEl = document.getElementById('input_headway_min');
  if (headwayEl && routeData.headway_min !== undefined) headwayEl.value = routeData.headway_min;
  const capEl = document.getElementById('input_vehicle_capacity');
  if (capEl && routeData.capacity !== undefined) capEl.value = routeData.capacity;

  // Use the empirical route-specific delay mean from metadata!
  const histEl = document.getElementById('input_historical_delay_mean');
  if (histEl) {
    if (routeData.hist_delay !== undefined) {
      histEl.value = routeData.hist_delay;
    } else {
      histEl.value = currentMode === 'BUS' ? 1.1 : (currentMode === 'TRAIN' ? 0.6 : 0.4);
    }
  }

  const occEl = document.getElementById('input_occupancy_rate');
  if (occEl) occEl.value = 0.65;
}

function onTransitOriginSelected(originOpt) {
  const routeOpt = comboboxes.transitRoute ? comboboxes.transitRoute.getOption() : null;
  if (!routeOpt || !routeOpt.meta) return;
  const routeData = routeOpt.meta;
  const originVal = typeof originOpt === 'string' ? originOpt : (originOpt?.value || comboboxes.transitOrigin?.getValue());
  if (!originVal) return;

  // Find destinations reachable from this origin on this route corridor
  let destList = [];
  if (routeData.valid_destinations && routeData.valid_destinations[originVal]) {
    destList = routeData.valid_destinations[originVal];
  } else if (routeData.stops) {
    destList = routeData.stops.filter(s => s !== originVal);
  } else if (routeData.destination && routeData.destination !== originVal) {
    destList = [routeData.destination];
  }

  if (!destList.length) {
    if (comboboxes.transitDest) {
      comboboxes.transitDest.setOptions([], null, 'Single-stop loop in source dataset');
    }
    return;
  }

  const destOptions = destList.map(d => ({
    value: d,
    label: d,
    badge: 'DESTINATION'
  }));

  if (comboboxes.transitDest) {
    comboboxes.transitDest.setOptions(destOptions, destOptions[0]?.value, 'Select Destination Station...');
  }
}

function onTransitDestSelected(opt) {
  // Destination selection confirmed
}

/**
 * =========================================================================
 * AVIATION DROPDOWN CASCADE (AIRPLANE)
 * Airline → Origin Airport → Destination Airport
 * =========================================================================
 */
function populateAviationAirlines() {
  if (!appMetadata || !appMetadata.airplane) return;

  const airlines = appMetadata.airplane.airlines || [];
  const airlineOptions = airlines.map(al => ({
    value: al.code,
    label: al.display,
    badge: al.code,
    meta: al
  }));

  const defaultAirline = airlineOptions.find(o => o.value === 'DL') || airlineOptions[0];
  if (comboboxes.flightAirline) {
    comboboxes.flightAirline.setOptions(airlineOptions, defaultAirline?.value, 'Select Operating Airline...');
  }
  if (defaultAirline) {
    onFlightAirlineSelected(defaultAirline.value);
  }
}

function onFlightAirlineSelected(airlineCode) {
  if (!appMetadata || !appMetadata.airplane) return;

  const flightTree = appMetadata.airplane.flight_tree || {};
  const airports = appMetadata.airplane.airports || {};
  const airlineRoutes = flightTree[airlineCode] || {};

  const originCodes = Object.keys(airlineRoutes).sort();

  if (!originCodes.length) {
    if (comboboxes.flightOrigin) comboboxes.flightOrigin.setOptions([], null, 'No flights for this airline');
    if (comboboxes.flightDest) comboboxes.flightDest.setOptions([], null, 'No flights');
    return;
  }

  // Build Origin Airport Options
  const originOptions = originCodes.map(code => {
    const port = airports[code] || { name: code, city: '', display: code };
    return {
      value: code,
      label: port.display,
      badge: code,
      meta: port
    };
  });

  let defaultOrigin = originOptions.find(o => o.value === 'ATL') ||
                      originOptions.find(o => o.value === 'JFK') ||
                      originOptions.find(o => o.value === 'ORD') ||
                      originOptions[0];

  if (comboboxes.flightOrigin) {
    comboboxes.flightOrigin.setOptions(originOptions, defaultOrigin?.value, 'Search Origin Airport...');
  }
  if (defaultOrigin) {
    onFlightOriginSelected(defaultOrigin.value);
  }
}

function onFlightOriginSelected(originCode) {
  const airlineCode = comboboxes.flightAirline ? comboboxes.flightAirline.getValue() : null;
  if (!appMetadata || !appMetadata.airplane || !airlineCode) return;

  // Sync live meteorological telemetry for the selected origin airport
  fetchLiveWeather(originCode);

  const flightTree = appMetadata.airplane.flight_tree || {};
  const airports = appMetadata.airplane.airports || {};
  const originDests = flightTree[airlineCode]?.[originCode] || {};

  const destCodes = Object.keys(originDests).sort();

  if (!destCodes.length) {
    if (comboboxes.flightDest) comboboxes.flightDest.setOptions([], null, 'No destinations from this airport');
    return;
  }

  // Build Destination Airport Options
  const destOptions = destCodes.map(code => {
    const port = airports[code] || { name: code, city: '', display: code };
    const routeInfo = originDests[code];
    return {
      value: code,
      label: port.display,
      badge: code,
      meta: { port, routeInfo }
    };
  });

  let defaultDest = destOptions.find(o => o.value === 'LAX') ||
                    destOptions.find(o => o.value === 'SFO') ||
                    destOptions.find(o => o.value === 'MIA') ||
                    destOptions[0];

  if (comboboxes.flightDest) {
    comboboxes.flightDest.setOptions(destOptions, defaultDest?.value, 'Search Destination Airport...');
  }
  if (defaultDest) {
    onFlightDestSelected(defaultDest.value);
  }
}

function onFlightDestSelected(destCode) {
  const airlineCode = comboboxes.flightAirline ? comboboxes.flightAirline.getValue() : null;
  const originCode = comboboxes.flightOrigin ? comboboxes.flightOrigin.getValue() : null;
  if (!airlineCode || !originCode || !destCode) return;

  const flightTree = appMetadata.airplane.flight_tree || {};
  const routeData = flightTree[airlineCode]?.[originCode]?.[destCode];
  const airlineMeta = comboboxes.flightAirline ? comboboxes.flightAirline.getOption()?.meta : null;

  if (routeData) {
    const distEl = document.getElementById('input_distance_km');
    if (distEl && routeData.dist_km !== undefined) distEl.value = routeData.dist_km;
    const durEl = document.getElementById('input_scheduled_duration_min');
    if (durEl && routeData.dur_min !== undefined) durEl.value = routeData.dur_min;
  }

  const histEl = document.getElementById('input_historical_delay_mean');
  if (histEl) {
    const originMeta = comboboxes.flightOrigin ? comboboxes.flightOrigin.getOption()?.meta : null;
    const cHist = (airlineMeta && airlineMeta.hist_delay !== undefined) ? airlineMeta.hist_delay : 11.2;
    const aHist = (originMeta && originMeta.hist_delay !== undefined) ? originMeta.hist_delay : 11.2;
    histEl.value = Math.round(((cHist + aHist) / 2.0) * 10) / 10;
  }

  const capEl = document.getElementById('input_vehicle_capacity');
  if (capEl) capEl.value = 180;
  const occEl = document.getElementById('input_occupancy_rate');
  if (occEl) occEl.value = 0.82;
}

/**
 * =========================================================================
 * FORM SUBMISSION, VALIDATION, AND ERROR RECOVERY
 * =========================================================================
 */
function initPredictionForm() {
  const form = document.getElementById('delay-prediction-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideValidationAlert();

    // Hide any previous stale prediction results
    const resultsContainer = document.getElementById('prediction-results');
    if (resultsContainer) {
      resultsContainer.style.display = 'none';
    }

    const mode = document.getElementById('input_transport_mode').value;
    let carrier_or_route = '';
    let origin = '';
    let destination = '';
    let city = '';

    // Validate selections according to Mode
    if (mode === 'AIRPLANE') {
      carrier_or_route = comboboxes.flightAirline.getValue();
      origin = comboboxes.flightOrigin.getValue();
      destination = comboboxes.flightDest.getValue();

      if (!carrier_or_route) {
        showValidationAlert('Please select an operating airline from the dataset.');
        comboboxes.flightAirline.open();
        return;
      }
      if (!origin) {
        showValidationAlert('Please select a valid origin airport from the dataset.');
        comboboxes.flightOrigin.open();
        return;
      }
      if (!destination) {
        showValidationAlert('Please select a valid destination airport for the selected origin.');
        comboboxes.flightDest.open();
        return;
      }
      if (origin === destination) {
        showValidationAlert('Origin and destination airports cannot be identical.');
        return;
      }

      const originPort = appMetadata?.airplane?.airports?.[origin];
      city = originPort?.city || 'Airport City';
    } else {
      city = document.getElementById('select_city').value;
      carrier_or_route = comboboxes.transitRoute.getValue();
      origin = comboboxes.transitOrigin.getValue();
      destination = comboboxes.transitDest.getValue();

      if (!city) {
        showValidationAlert('Please select a valid city from the dataset.');
        return;
      }
      if (!carrier_or_route) {
        showValidationAlert('Please select a valid transit route ID from the dataset.');
        comboboxes.transitRoute.open();
        return;
      }
      if (!origin) {
        showValidationAlert('Please select a valid origin station for the selected route.');
        comboboxes.transitOrigin.open();
        return;
      }
      if (!destination) {
        showValidationAlert('Please select a valid destination station for the selected route.');
        comboboxes.transitDest.open();
        return;
      }
      if (origin === destination) {
        showValidationAlert('Origin and destination stations cannot be identical.');
        return;
      }
    }

    // Build Payload
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    payload.transport_mode = mode;
    payload.carrier_or_route = carrier_or_route;
    payload.origin = origin;
    payload.destination = destination;
    payload.city = city;

    if (mode === 'AIRPLANE') {
      payload.headway_min = 0;
    }

    const submitBtn = document.getElementById('btn-predict-submit');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '⏳ Computing Model Inference...';
    submitBtn.disabled = true;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const res = await response.json();
      if (res.status === 'success') {
        renderPredictionResult(res, payload);
        savePredictionToHistory(res, payload);
        renderPredictionHistoryTable();
      } else {
        showValidationAlert('Prediction error: ' + (res.message || 'Unknown server error'));
      }
    } catch (err) {
      console.error('Inference request failed:', err);
      showValidationAlert('Unable to communicate with prediction server. Please verify that the Flask backend is active on port 5050.');
    } finally {
      submitBtn.innerHTML = originalText;
      submitBtn.disabled = false;
    }
  });
}

function showValidationAlert(message) {
  const alertEl = document.getElementById('form-validation-alert');
  const alertText = document.getElementById('validation-alert-text');
  if (alertEl && alertText) {
    alertText.textContent = message;
    alertEl.style.display = 'flex';
    alertEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {
    console.warn('Validation alert:', message);
  }
}

function hideValidationAlert() {
  const alertEl = document.getElementById('form-validation-alert');
  if (alertEl) alertEl.style.display = 'none';
}

/**
 * Updates UI with prediction results
 */
function renderPredictionResult(data, payload) {
  const container = document.getElementById('prediction-results');
  container.style.display = 'block';

  document.getElementById('res-mode-badge').textContent = `TRANSPORT MODE: ${data.transport_mode}`;
  
  // Render dual model names
  const classifierEl = document.getElementById('res-classifier-name');
  if (classifierEl) {
    classifierEl.textContent = data.classifier_name || data.model_used;
  }
  const regressorEl = document.getElementById('res-regressor-name');
  if (regressorEl) {
    regressorEl.textContent = data.regressor_name || 'Mode Quantile & Gradient Boosting';
  }

  // Render classifier probability and dataset base rate
  document.getElementById('res-confidence-pct').textContent = `${data.confidence_pct}%`;
  const baseRateEl = document.getElementById('res-base-rate-text');
  if (baseRateEl && data.base_rates) {
    const rate = data.base_rates[data.predicted_category] || 82.4;
    baseRateEl.textContent = `(Dataset Base Rate: ~${rate}%)`;
  }

  // Render continuous estimated delay and quantile bounds (P50 & P90)
  const estDelayEl = document.getElementById('res-estimated-delay');
  if (estDelayEl) {
    if (data.estimated_delay_min !== null && data.estimated_delay_min !== undefined) {
      estDelayEl.textContent = `${data.estimated_delay_min} min`;
    } else {
      estDelayEl.textContent = 'N/A (Categorical Only)';
    }
  }

  const anyDelayEl = document.getElementById('res-chance-any');
  if (anyDelayEl) anyDelayEl.textContent = `${data.chance_any_delay_pct || 0}%`;

  const gt5El = document.getElementById('res-chance-gt5');
  if (gt5El) gt5El.textContent = `${data.chance_gt5_pct || 0}%`;

  const condDelayEl = document.getElementById('res-cond-delay');
  if (condDelayEl) condDelayEl.textContent = `${data.typical_delay_if_delayed_min || 0} min`;

  const p90ValEl = document.getElementById('res-p90-val');
  if (p90ValEl) p90ValEl.textContent = `${data.high_end_risk_p90_min || data.high_end_p90_delay_min || 0} min`;

  // Render Risk Assessment
  const riskBadge = document.getElementById('res-risk-badge');
  const riskDesc = document.getElementById('res-risk-desc');
  if (riskBadge) {
    riskBadge.textContent = `${(data.risk_level || 'LOW').toUpperCase()} RISK`;
    riskBadge.className = `badge ${data.risk_badge || 'badge-ontime'}`;
  }
  if (riskDesc) {
    riskDesc.textContent = data.risk_desc || 'Nominal operating conditions expected.';
  }

  // Render Model Disagreement / Consensus Disclosure
  const disagreeBox = document.getElementById('res-disagreement-box');
  const disagreeText = document.getElementById('res-disagreement-text');
  if (disagreeBox && disagreeText) {
    if (data.disagreement_disclosure) {
      disagreeText.textContent = data.disagreement_disclosure;
      disagreeBox.style.display = 'block';
    } else {
      disagreeBox.style.display = 'none';
    }
  }

  // Render Data Provenance & Timestamp
  const srcEl = document.getElementById('res-data-source');
  if (srcEl) srcEl.textContent = data.data_source || 'Historical Ground-Truth Data + Open-Meteo Weather';
  const limEl = document.getElementById('res-limitations');
  if (limEl) limEl.textContent = data.limitations || 'Empirical historical delay patterns.';
  const timeEl = document.getElementById('res-timestamp');
  if (timeEl) timeEl.textContent = data.prediction_timestamp || new Date().toLocaleString();

  const pill = document.getElementById('res-severity-pill');
  pill.textContent = data.predicted_category;

  if (data.predicted_category === 'ON TIME') {
    pill.style.background = 'rgba(16, 185, 129, 0.2)';
    pill.style.color = '#34d399';
    pill.style.border = '1px solid rgba(16, 185, 129, 0.4)';
  } else if (data.predicted_category === 'MODERATELY DELAYED') {
    pill.style.background = 'rgba(245, 158, 11, 0.2)';
    pill.style.color = '#fbbf24';
    pill.style.border = '1px solid rgba(245, 158, 11, 0.4)';
  } else {
    pill.style.background = 'rgba(239, 68, 68, 0.2)';
    pill.style.color = '#f87171';
    pill.style.border = '1px solid rgba(239, 68, 68, 0.4)';
  }

  // Selected Journey Summary Pill
  const summaryEl = document.getElementById('res-journey-summary');
  if (summaryEl) {
    if (data.transport_mode === 'AIRPLANE') {
      const alName = comboboxes.flightAirline?.getOption()?.label || payload.carrier_or_route;
      const origName = comboboxes.flightOrigin?.getOption()?.label || payload.origin;
      const destName = comboboxes.flightDest?.getOption()?.label || payload.destination;
      summaryEl.innerHTML = `
        <strong>Flight Itinerary:</strong> ${alName} • 
        <span style="color: var(--accent-cyan);">${origName}</span> ➔ 
        <span style="color: var(--accent-cyan);">${destName}</span> • 
        Distance: ${Number(payload.distance_km).toLocaleString()} km • Duration: ${payload.scheduled_duration_min} mins
      `;
    } else {
      summaryEl.innerHTML = `
        <strong>Transit Journey:</strong> ${payload.city.replace(/_/g, ' ')} • Route <span style="color: var(--accent-cyan);">${payload.carrier_or_route}</span> • 
        Stop <span style="color: var(--accent-cyan);">${payload.origin}</span> ➔ 
        Stop <span style="color: var(--accent-cyan);">${payload.destination}</span> • 
        Length: ${payload.distance_km} km • Travel Time: ${payload.scheduled_duration_min} mins
      `;
    }
  }

  // Calibrated probabilities
  const probs = data.probabilities;
  const pOntime = probs['ON TIME'] || 0;
  const pModerate = probs['MODERATELY DELAYED'] || 0;
  const pSevere = probs['SEVERELY DELAYED'] || 0;

  document.getElementById('bar-ontime').style.width = `${pOntime}%`;
  document.getElementById('val-ontime').textContent = `${pOntime}%`;

  document.getElementById('bar-moderate').style.width = `${pModerate}%`;
  document.getElementById('val-moderate').textContent = `${pModerate}%`;

  document.getElementById('bar-severe').style.width = `${pSevere}%`;
  document.getElementById('val-severe').textContent = `${pSevere}%`;

  // Model-Derived Local Counterfactual Explanations
  const factorsList = document.getElementById('res-factors-list');
  factorsList.innerHTML = '';

  if (data.important_factors && data.important_factors.length) {
    data.important_factors.forEach(item => {
      const el = document.createElement('div');
      el.className = 'factor-item';
      const badgeClass = item.impact.includes('Driver') || item.impact.includes('Severe') ? 'badge-severe' : (item.impact.includes('Moderate') ? 'badge-moderate' : 'badge-ontime');
      
      el.innerHTML = `
        <div>
          <strong>${item.factor}</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${item.value}</div>
        </div>
        <span class="badge ${badgeClass}">
          ${item.impact}
        </span>
      `;
      factorsList.appendChild(el);
    });
  } else {
    factorsList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">Standard nominal operational conditions observed.</div>';
  }

  // Smooth scroll to results
  container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * =========================================================================
 * PREDICTION HISTORY MANAGER (LOCAL STORAGE)
 * =========================================================================
 */
const HISTORY_KEY = 'multimodal_delay_prediction_history_v2';

function initPredictionHistory() {
  renderPredictionHistoryTable();

  const clearBtn = document.getElementById('btn-clear-history');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      localStorage.removeItem(HISTORY_KEY);
      renderPredictionHistoryTable();
    });
  }
}

function savePredictionToHistory(res, payload) {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];

    const weatherLabel = (payload.weather_condition || 'clear').replace(/_/g, ' ');
    const tempLabel = payload.temperature_c !== undefined ? `${payload.temperature_c}°C` : '18°C';
    const occPct = Math.round(parseFloat(payload.occupancy_rate || 0.65) * 100);
    const headwayVal = res.transport_mode === 'AIRPLANE' ? 'N/A' : (payload.headway_min !== undefined && payload.headway_min !== '' ? `${payload.headway_min}m` : 'N/A');

    const record = {
      timestamp: res.inference_timestamp || res.prediction_timestamp || new Date().toUTCString(),
      mode: res.transport_mode,
      carrier_or_route: payload.carrier_or_route,
      origin: payload.origin,
      destination: payload.destination,
      weather_temp: `${weatherLabel} (${tempLabel})`,
      ops_summary: `${payload.hour}:00 | HW: ${headwayVal} | Occ: ${occPct}%`,
      severity: res.predicted_category,
      estimated_delay: `${res.estimated_delay_min} min (P90: ${res.high_end_risk_p90_min || res.high_end_p90_delay_min || 0}m)`,
      confidence: `${res.confidence_pct}%`,
      risk_level: res.risk_level || 'Low',
      risk_badge: res.risk_badge || 'badge-ontime',
      model_version: `${res.classifier_name || 'Random Forest'} + ${res.regressor_name || 'CoherentTwoPartModel'}`
    };

    // Keep up to 10 latest predictions
    list.unshift(record);
    if (list.length > 10) list.pop();

    localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
  } catch (e) {
    console.warn('Unable to persist prediction to localStorage:', e);
  }
}

function renderPredictionHistoryTable() {
  const tbody = document.getElementById('history-table-body');
  if (!tbody) return;

  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];

    if (!list.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">
            No predictions recorded in this session yet. Submit a prediction above to view history.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = '';
    list.forEach(item => {
      const tr = document.createElement('tr');
      const sevClass = item.severity === 'ON TIME' ? 'badge-ontime' : (item.severity === 'MODERATELY DELAYED' ? 'badge-moderate' : 'badge-severe');

      tr.innerHTML = `
        <td style="font-size: 0.8rem; color: var(--text-muted);">${item.timestamp}</td>
        <td><strong>${item.mode}</strong></td>
        <td><code>${item.carrier_or_route}</code></td>
        <td style="font-size: 0.82rem;">${item.origin} ➔ ${item.destination}</td>
        <td style="font-size: 0.82rem;">${item.weather_temp || 'Clear (18°C)'}</td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${item.ops_summary || '08:00'}</td>
        <td><span class="badge ${sevClass}">${item.severity}</span></td>
        <td><strong style="color: var(--accent-cyan); font-size: 0.85rem;">${item.estimated_delay}</strong></td>
        <td>${item.confidence}</td>
        <td><span class="badge ${item.risk_badge}">${item.risk_level} Risk</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.warn('Unable to render prediction history:', e);
  }
}
