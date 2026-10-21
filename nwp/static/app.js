(() => {
  const els = {
    model: document.getElementById('model-select'),
    modelResolution: document.getElementById('model-resolution'),
    modelSource: document.getElementById('model-source'),
    modelNote: document.getElementById('model-note'),
    field: document.getElementById('field-select'),
    fieldNote: document.getElementById('field-note'),
    forecastControls: document.getElementById('forecast-controls'),
    regularControls: document.getElementById('regular-layer-controls'),
    trackControls: document.getElementById('track-model-controls'),
    trackModels: [...document.querySelectorAll('.track-model-toggle')],
    lead: document.getElementById('lead-range'),
    leadLabel: document.getElementById('lead-label'),
    leadEnd: document.getElementById('lead-end'),
    prev: document.getElementById('prev-step'),
    next: document.getElementById('next-step'),
    play: document.getElementById('play-button'),
    speed: document.getElementById('speed-toggle'),
    stream: document.getElementById('stream-toggle'),
    units: document.getElementById('unit-select'),
    image: document.getElementById('forecast-image'),
    placeholder: document.getElementById('map-placeholder'),
    loading: document.getElementById('map-loading'),
    error: document.getElementById('map-error'),
    errorMessage: document.getElementById('error-message'),
    retry: document.getElementById('retry-button'),
    state: document.getElementById('render-state'),
    download: document.getElementById('download-button'),
    mapModel: document.getElementById('map-model-label'),
    mapStep: document.getElementById('map-step-label'),
    valid: document.getElementById('valid-time'),
    init: document.getElementById('init-time'),
    clock: document.getElementById('utc-clock'),
    trackStatus: document.getElementById('track-status'),
  };

  const TRACK_FIELD = 'cyclone_track';
  let catalog = [];
  let currentBlobUrl = null;
  let currentDownload = null;
  let hdDownloadInProgress = false;
  let animationTimer = null;
  let requestController = null;
  let trackController = null;
  let requestSequence = 0;
  let trackSequence = 0;
  let leadDebounce = null;
  let lastSelectedStep = 24;

  const pad = (value, length = 3) => String(value).padStart(length, '0');
  const currentModel = () => catalog.find(model => model.id === els.model.value);
  const currentField = () => currentModel()?.fields?.find(field => field.id === els.field.value);
  const isTrackMode = () => els.field.value === TRACK_FIELD;
  const availableSteps = () => {
    const model = currentModel();
    const field = currentField();
    return field?.steps?.length ? field.steps : (model?.steps || []);
  };
  const selectedStep = () => availableSteps()[Number(els.lead.value)] ?? 24;

  function setState(text, stateClass = '') {
    els.state.className = `render-state ${stateClass}`.trim();
    els.state.lastElementChild.textContent = text;
  }

  function updateClock() {
    els.clock.textContent = `UTC ${new Intl.DateTimeFormat('en-GB', {hour: '2-digit', minute: '2-digit', timeZone: 'UTC'}).format(new Date())}`;
  }

  function clearCurrentImage() {
    currentDownload = null;
    els.download.disabled = true;
  }

  function showLoading(label = 'Building forecast image') {
    clearCurrentImage();
    els.error.hidden = true;
    els.placeholder.hidden = true;
    els.loading.hidden = false;
    els.loading.querySelector('strong').textContent = label;
    els.image.hidden = true;
    setState('Rendering…', 'is-loading');
  }

  function showError(message) {
    clearCurrentImage();
    els.loading.hidden = true;
    els.image.hidden = true;
    els.error.hidden = false;
    els.errorMessage.textContent = message;
    setState('Unavailable', 'is-error');
  }

  function populateFields(model, preferred = els.field.value) {
    const fields = [...(model.fields || [])];
    fields.push({id: TRACK_FIELD, label: 'Cyclone Track'});
    els.field.replaceChildren();
    for (const field of fields) {
      const option = document.createElement('option');
      option.value = field.id;
      option.textContent = field.label;
      els.field.appendChild(option);
    }
    const target = fields.some(field => field.id === preferred) ? preferred : 'wind';
    els.field.value = fields.some(field => field.id === target) ? target : fields[0]?.id;
  }

  function fitStepToField() {
    const steps = availableSteps();
    if (!steps.length) return;
    const current = lastSelectedStep;
    const nearest = steps.includes(current) ? current : steps.reduce((best, value) =>
      Math.abs(value - current) < Math.abs(best - current) ? value : best, steps[0]);
    els.lead.min = '0';
    els.lead.max = String(steps.length - 1);
    els.lead.value = String(steps.indexOf(nearest));
    els.leadEnd.textContent = `+${steps[steps.length - 1]} h`;
  }

  function updateFieldControls() {
    const field = currentField();
    if (!field || isTrackMode()) return;
    const isWind = field.id === 'wind' || field.id === 'wind_200';
    const speedRow = els.speed.closest('.layer-row');
    const streamRow = els.stream.closest('.layer-row');
    els.speed.disabled = !isWind;
    els.stream.disabled = !isWind;
    speedRow.classList.toggle('is-inactive', !isWind);
    streamRow.classList.toggle('is-inactive', !isWind);

    const oldUnits = els.units.value;
    els.units.replaceChildren();
    for (const unit of field.units || []) {
      const option = document.createElement('option');
      option.value = unit.id;
      option.textContent = unit.label;
      els.units.appendChild(option);
    }
    const unitIds = (field.units || []).map(unit => unit.id);
    els.units.value = unitIds.includes(oldUnits) ? oldUnits : (unitIds[0] || 'knots');
    els.units.disabled = unitIds.length < 2;

    const notes = {
      wind: 'Only the selected model, forecast hour, and 10 m U/V wind fields are fetched; maps and fields remain cached.',
      mslp: 'Shaded sea-level pressure with labelled isobars. Only the selected model and lead are fetched.',
      temperature: '2 m air temperature in °C or °F. The map uses a fixed, readable temperature scale.',
      precipitation: 'Accumulation units are mm; the chart also labels the source GRIB accumulation interval.',
      precipitation_6h: 'A verified six-hour total in mm; source intervals are checked and combined only when adjacent.',
      gust: 'Peak 10 m gust, not sustained wind. ECMWF gust leads are limited to the published range.',
      wind_200: '200 hPa vector wind, shown with speed shading and/or streamlines. Only published pressure-level fields are offered.',
      shear_200_850: 'Vector difference between 200 and 850 hPa winds, shaded as shear magnitude. Available only where both official levels are published.',
    };
    els.fieldNote.textContent = notes[field.id] || 'Only the selected official model, lead, and parameter are fetched.';
    els.image.alt = `${field.label} forecast over the Bay of Bengal`;
  }

  function updateModelDetails() {
    const model = currentModel();
    if (!model) return;
    const preferredField = els.field.value || 'wind';
    els.modelResolution.textContent = model.resolution;
    els.modelSource.textContent = model.source;
    els.modelNote.textContent = model.note || '';
    populateFields(model, preferredField);
    if (isTrackMode()) return;
    const steps = availableSteps();
    const initialStep = steps.includes(24) ? 24 : steps[0];
    els.lead.min = '0';
    els.lead.max = String(Math.max(0, steps.length - 1));
    els.lead.value = String(Math.max(0, steps.indexOf(initialStep)));
    els.leadEnd.textContent = `+${steps[steps.length - 1]} h`;
    updateFieldControls();
    updateLeadLabel();
    els.mapModel.textContent = model.name;
  }

  function updateLeadLabel() {
    const step = selectedStep();
    lastSelectedStep = step;
    els.leadLabel.textContent = `+${pad(step)} h`;
    els.mapStep.textContent = `F${pad(step)}`;
  }

  function setParameterMode() {
    const trackMode = isTrackMode();
    els.forecastControls.hidden = trackMode;
    els.regularControls.hidden = trackMode;
    els.trackControls.hidden = !trackMode;
    stopAnimation();
    if (trackMode) {
      requestSequence += 1;
      if (requestController) requestController.abort();
      els.mapModel.textContent = 'Cyclone Track';
      els.mapStep.textContent = 'Full selected runs';
      els.valid.textContent = 'Full available forecast runs';
      els.init.textContent = 'Run stamps shown on map';
      els.image.alt = 'Standalone cyclone tracks over the Bay of Bengal';
      loadTrackMap();
    } else {
      trackSequence += 1;
      if (trackController) trackController.abort();
      els.trackStatus.hidden = true;
      els.mapModel.textContent = currentModel()?.name || 'Forecast';
      updateFieldControls();
      updateLeadLabel();
      loadMap();
    }
  }

  async function loadMap() {
    const model = currentModel();
    if (!model || isTrackMode()) return;
    const vectorField = els.field.value === 'wind' || els.field.value === 'wind_200';
    if (vectorField && !els.speed.checked && !els.stream.checked) {
      showError('Turn on Wind speed or Streamlines to display a map.');
      setState('Choose a layer', 'is-error');
      return;
    }
    if (trackController) trackController.abort();
    if (requestController) requestController.abort();
    requestController = new AbortController();
    const requestId = ++requestSequence;
    showLoading('Building forecast image');
    updateLeadLabel();
    const query = new URLSearchParams({
      model: model.id,
      step: String(selectedStep()),
      units: els.units.value,
      field: els.field.value,
      speed: vectorField && els.speed.checked ? '1' : '0',
      streamlines: vectorField && els.stream.checked ? '1' : '0',
      dpi: '150',
    });
    try {
      const response = await fetch(`/api/map.png?${query}`, {signal: requestController.signal, cache: 'no-store'});
      if (!response.ok) throw new Error(await responseError(response));
      const blob = await response.blob();
      if (requestId !== requestSequence) return;
      displayBlob(blob, `${model.name}, ${model.resolution}, ${currentField()?.label || 'weather'} forecast at hour ${selectedStep()}`);
      currentDownload = {path: '/api/map.png', query};
      els.valid.textContent = response.headers.get('X-Valid-Time') || '—';
      const run = response.headers.get('X-Model-Run');
      els.init.textContent = run ? run.replace('T', ' ').replace('Z', ' UTC') : 'Latest available';
      setState('Map ready', 'is-loaded');
    } catch (error) {
      if (error.name === 'AbortError' || requestId !== requestSequence) return;
      showError(error.message || 'The selected forecast could not be generated.');
    }
  }

  async function loadTrackMap() {
    if (!isTrackMode()) return;
    if (requestController) requestController.abort();
    if (trackController) trackController.abort();
    const selected = els.trackModels.filter(input => input.checked).map(input => input.value);
    const sequence = ++trackSequence;
    if (!selected.length) {
      showError('Select at least one model to display a cyclone track.');
      els.trackStatus.hidden = true;
      return;
    }
    trackController = new AbortController();
    showLoading('Building dedicated cyclone track map');
    els.trackStatus.hidden = false;
    els.trackStatus.className = 'track-status is-loading';
    els.trackStatus.textContent = `Loading full forecast runs for ${selected.length} selected model${selected.length === 1 ? '' : 's'}…`;
    const query = new URLSearchParams({models: selected.join(','), dpi: '150'});
    try {
      const response = await fetch(`/api/track.png?${query}`, {signal: trackController.signal, cache: 'no-store'});
      if (!response.ok) throw new Error(await responseError(response));
      const blob = await response.blob();
      if (sequence !== trackSequence || !isTrackMode()) return;
      displayBlob(blob, `Standalone cyclone track map for ${selected.join(', ')}`);
      currentDownload = {path: '/api/track.png', query};
      const horizons = response.headers.get('X-Track-Horizons') || 'Full available runs';
      const unavailable = response.headers.get('X-Track-Unavailable');
      const trackNotes = response.headers.get('X-Track-Notes');
      els.trackStatus.className = (unavailable || trackNotes) ? 'track-status is-warning' : 'track-status';
      els.trackStatus.textContent = `12 h samples · ${horizons}${unavailable ? ` · Unavailable now: ${unavailable}` : ''}${trackNotes ? ` · ${trackNotes}` : ''}`;
      els.trackStatus.title = 'Each track uses one coherent model run. It stops if no continuous pressure center can be followed; IFS and GFS track caps are +240 h.';
      els.valid.textContent = 'Full available forecast runs';
      els.init.textContent = response.headers.get('X-Model-Run') === 'Multiple model cycles'
        ? 'Run stamps shown on map' : (response.headers.get('X-Model-Run') || 'Latest available');
      els.mapStep.textContent = horizons.replaceAll('; ', ' · ');
      setState('Track map ready', 'is-loaded');
    } catch (error) {
      if (error.name === 'AbortError' || sequence !== trackSequence) return;
      els.trackStatus.className = 'track-status is-error';
      els.trackStatus.textContent = error.message || 'The selected cyclone track map is unavailable.';
      els.trackStatus.title = 'Try fewer models or retry when a new official model cycle is available.';
      showError(error.message || 'The selected cyclone track map is unavailable.');
    }
  }

  function displayBlob(blob, alt) {
    if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = URL.createObjectURL(blob);
    els.image.src = currentBlobUrl;
    els.image.alt = alt;
    els.image.hidden = false;
    els.loading.hidden = true;
    els.error.hidden = true;
    els.placeholder.hidden = true;
    els.download.disabled = false;
  }

  async function responseError(response) {
    let message = `Server returned ${response.status}.`;
    try { message = (await response.json()).error || message; } catch (_) {}
    return message;
  }

  function stopAnimation() {
    if (animationTimer) window.clearInterval(animationTimer);
    animationTimer = null;
    els.play.classList.remove('is-playing');
    els.play.querySelector('span').textContent = 'Animate forecast';
    els.play.querySelector('.play-icon').innerHTML = '<path d="m6 4 10 6-10 6z"/>';
  }

  function startAnimation() {
    const steps = availableSteps();
    if (isTrackMode() || steps.length < 2) return;
    els.play.classList.add('is-playing');
    els.play.querySelector('span').textContent = 'Pause animation';
    els.play.querySelector('.play-icon').innerHTML = '<path d="M6 4h3v12H6zM12 4h3v12h-3z"/>';
    let index = Number(els.lead.value);
    animationTimer = window.setInterval(() => {
      index = (index + 1) % steps.length;
      els.lead.value = String(index);
      updateLeadLabel();
      loadMap();
    }, 4200);
  }

  function moveStep(direction) {
    const steps = availableSteps();
    if (isTrackMode() || !steps.length) return;
    const index = Math.min(steps.length - 1, Math.max(0, Number(els.lead.value) + direction));
    els.lead.value = String(index);
    updateLeadLabel();
    loadMap();
  }

  async function downloadCurrent() {
    if (!currentDownload || hdDownloadInProgress) return;
    hdDownloadInProgress = true;
    const label = els.download.querySelector('span');
    const previousLabel = label.textContent;
    const query = new URLSearchParams(currentDownload.query);
    query.set('dpi', '300');
    const path = currentDownload.path;
    els.download.disabled = true;
    label.textContent = 'Making HD…';
    setState('Preparing 300 DPI PNG…', 'is-loading');
    try {
      const response = await fetch(`${path}?${query}`, {cache: 'no-store'});
      if (!response.ok) throw new Error(await responseError(response));
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      if (path === '/api/track.png') {
        link.download = `baywind-cyclone-track-${query.get('models') || 'ifs_025'}-300dpi.png`;
      } else {
        link.download = `baywind-${query.get('model')}-${query.get('field') || 'wind'}-f${pad(query.get('step') || 0)}-300dpi.png`;
      }
      link.href = downloadUrl;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1500);
      setState(isTrackMode() ? 'Track map ready' : 'Map ready', 'is-loaded');
    } catch (error) {
      els.download.title = error.message || 'High-resolution export failed';
      setState('HD export failed', 'is-error');
      window.setTimeout(() => {
        if (!els.image.hidden) setState(isTrackMode() ? 'Track map ready' : 'Map ready', 'is-loaded');
        els.download.title = 'Download 300 DPI PNG';
      }, 4500);
    } finally {
      hdDownloadInProgress = false;
      label.textContent = previousLabel;
      els.download.disabled = els.image.hidden || !currentDownload;
    }
  }

  async function init() {
    updateClock();
    window.setInterval(updateClock, 30000);
    try {
      const response = await fetch('/api/models');
      if (!response.ok) throw new Error('Could not load the model list.');
      const data = await response.json();
      catalog = data.models || [];
      els.model.replaceChildren();
      for (const model of catalog) {
        const option = document.createElement('option');
        option.value = model.id;
        option.disabled = model.enabled === false;
        option.textContent = `${model.name} — ${model.resolution}${model.enabled === false ? ' (API key required)' : ''}`;
        els.model.appendChild(option);
      }
      els.model.value = 'ifs_025';
      updateModelDetails();
      setParameterMode();
    } catch (error) {
      showError(error.message || 'Unable to load the model catalog.');
    }
  }

  els.model.addEventListener('change', () => {
    stopAnimation();
    updateModelDetails();
    if (isTrackMode()) loadTrackMap(); else loadMap();
  });
  els.field.addEventListener('change', () => {
    stopAnimation();
    if (!isTrackMode()) fitStepToField();
    setParameterMode();
  });
  els.lead.addEventListener('input', () => {
    updateLeadLabel();
    window.clearTimeout(leadDebounce);
    leadDebounce = window.setTimeout(loadMap, 300);
  });
  els.prev.addEventListener('click', () => moveStep(-1));
  els.next.addEventListener('click', () => moveStep(1));
  els.play.addEventListener('click', () => animationTimer ? stopAnimation() : startAnimation());
  els.speed.addEventListener('change', loadMap);
  els.stream.addEventListener('change', loadMap);
  els.units.addEventListener('change', loadMap);
  els.trackModels.forEach(input => input.addEventListener('change', () => { if (isTrackMode()) loadTrackMap(); }));
  els.retry.addEventListener('click', () => isTrackMode() ? loadTrackMap() : loadMap());
  els.download.addEventListener('click', downloadCurrent);
  window.addEventListener('beforeunload', () => { if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl); });

  init();
})();
