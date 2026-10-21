(() => {
  const els = {
    model: document.getElementById('model-select'),
    modelResolution: document.getElementById('model-resolution'),
    modelSource: document.getElementById('model-source'),
    modelNote: document.getElementById('model-note'),
    field: document.getElementById('field-select'),
    fieldNote: document.getElementById('field-note'),
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
  };

  let catalog = [];
  let currentBlobUrl = null;
  let currentDownloadQuery = null;
  let hdDownloadInProgress = false;
  let animationTimer = null;
  let requestController = null;
  let requestSequence = 0;
  let leadDebounce = null;

  const pad = (value, length = 3) => String(value).padStart(length, '0');
  const currentModel = () => catalog.find(model => model.id === els.model.value);
  const currentField = () => currentModel()?.fields?.find(field => field.id === els.field.value);
  const availableSteps = () => {
    const model = currentModel();
    const field = currentField();
    return field?.steps?.length ? field.steps : (model?.steps || []);
  };
  const selectedStep = () => availableSteps()[Number(els.lead.value)] ?? 24;
  let lastSelectedStep = 24;

  function setState(text, stateClass = '') {
    els.state.className = `render-state ${stateClass}`.trim();
    els.state.lastElementChild.textContent = text;
  }

  function updateClock() {
    const now = new Date();
    els.clock.textContent = `UTC ${new Intl.DateTimeFormat('en-GB', {hour: '2-digit', minute: '2-digit', timeZone: 'UTC'}).format(now)}`;
  }

  function populateFields(model, preferred = els.field.value) {
    const available = model.fields || [{id: 'wind', label: '10 m Wind', units: [{id: 'knots', label: 'Knots'}], steps: model.steps}];
    els.field.innerHTML = '';
    for (const field of available) {
      const option = document.createElement('option');
      option.value = field.id;
      option.textContent = field.label;
      els.field.appendChild(option);
    }
    const target = available.some(field => field.id === preferred) ? preferred : 'wind';
    els.field.value = available.some(field => field.id === target) ? target : available[0].id;
  }

  function fitStepToField() {
    const steps = availableSteps();
    if (!steps.length) return;
    const current = lastSelectedStep;
    const nearest = steps.includes(current) ? current : steps.reduce((best, value) => Math.abs(value - current) < Math.abs(best - current) ? value : best, steps[0]);
    els.lead.min = '0';
    els.lead.max = String(steps.length - 1);
    els.lead.value = String(steps.indexOf(nearest));
    els.leadEnd.textContent = `+${steps[steps.length - 1]} h`;
  }

  function updateFieldControls() {
    const field = currentField();
    if (!field) return;
    const isWind = field.id === 'wind';
    const speedRow = els.speed.closest('.layer-row');
    const streamRow = els.stream.closest('.layer-row');
    els.speed.disabled = !isWind;
    els.stream.disabled = !isWind;
    speedRow.classList.toggle('is-inactive', !isWind);
    streamRow.classList.toggle('is-inactive', !isWind);

    const oldUnits = els.units.value;
    els.units.innerHTML = '';
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
    els.mapModel.textContent = model.name;
    populateFields(model, preferredField);
    const steps = availableSteps();
    const initialStep = steps.includes(24) ? 24 : steps[0];
    els.lead.min = '0';
    els.lead.max = String(Math.max(0, steps.length - 1));
    els.lead.value = String(Math.max(0, steps.indexOf(initialStep)));
    els.leadEnd.textContent = `+${steps[steps.length - 1]} h`;
    updateFieldControls();
    updateLeadLabel();
  }

  function updateLeadLabel() {
    const step = selectedStep();
    lastSelectedStep = step;
    els.leadLabel.textContent = `+${pad(step)} h`;
    els.mapStep.textContent = `F${pad(step)}`;
  }

  function showLoading() {
    currentDownloadQuery = null;
    els.error.hidden = true;
    els.placeholder.hidden = true;
    els.loading.hidden = false;
    els.image.hidden = true;
    els.download.disabled = true;
    setState('Rendering…', 'is-loading');
  }

  function showError(message) {
    currentDownloadQuery = null;
    els.download.disabled = true;
    els.loading.hidden = true;
    els.image.hidden = true;
    els.error.hidden = false;
    els.errorMessage.textContent = message;
    setState('Unavailable', 'is-error');
  }

  async function loadMap() {
    const model = currentModel();
    if (!model) return;
    if (els.field.value === 'wind' && !els.speed.checked && !els.stream.checked) {
      currentDownloadQuery = null;
      els.download.disabled = true;
      els.error.hidden = false;
      els.loading.hidden = true;
      els.placeholder.hidden = true;
      els.image.hidden = true;
      els.errorMessage.textContent = 'Turn on Wind speed or Streamlines to display a map.';
      setState('Choose a layer', 'is-error');
      return;
    }

    if (requestController) requestController.abort();
    requestController = new AbortController();
    const requestId = ++requestSequence;
    showLoading();
    updateLeadLabel();

    const query = new URLSearchParams({
      model: model.id,
      step: String(selectedStep()),
      units: els.units.value,
      field: els.field.value,
      speed: els.field.value === 'wind' && els.speed.checked ? '1' : '0',
      streamlines: els.field.value === 'wind' && els.stream.checked ? '1' : '0',
      dpi: '150',
    });

    try {
      const response = await fetch(`/api/map.png?${query.toString()}`, {signal: requestController.signal, cache: 'no-store'});
      if (!response.ok) {
        let message = `Server returned ${response.status}.`;
        try { message = (await response.json()).error || message; } catch (_) {}
        throw new Error(message);
      }
      const blob = await response.blob();
      if (requestId !== requestSequence) return;
      if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = URL.createObjectURL(blob);
      els.image.src = currentBlobUrl;
      const selectedField = currentField();
      els.image.alt = `${model.name}, ${model.resolution}, ${selectedField?.label || 'weather'} forecast at hour ${selectedStep()}`;
      els.image.hidden = false;
      els.loading.hidden = true;
      els.error.hidden = true;
      els.placeholder.hidden = true;
      currentDownloadQuery = query.toString();
      els.download.disabled = false;
      els.valid.textContent = response.headers.get('X-Valid-Time') || '—';
      const run = response.headers.get('X-Model-Run');
      els.init.textContent = run ? `${run.replace('T', ' ').replace('Z', ' UTC')}` : 'Latest available';
      setState('Map ready', 'is-loaded');
    } catch (error) {
      if (error.name === 'AbortError') return;
      if (requestId !== requestSequence) return;
      showError(error.message || 'Unable to retrieve this model run. Try another forecast hour.');
    }
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
    if (steps.length < 2) return;
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
    if (!steps.length) return;
    const next = Math.min(steps.length - 1, Math.max(0, Number(els.lead.value) + direction));
    els.lead.value = String(next);
    updateLeadLabel();
    loadMap();
  }

  async function downloadCurrent() {
    if (!currentBlobUrl || !currentDownloadQuery || hdDownloadInProgress) return;
    hdDownloadInProgress = true;
    const label = els.download.querySelector('span');
    const previousLabel = label.textContent;
    const query = new URLSearchParams(currentDownloadQuery);
    query.set('dpi', '300');
    els.download.disabled = true;
    label.textContent = 'Making HD…';
    setState('Preparing 300 DPI PNG…', 'is-loading');

    try {
      const response = await fetch(`/api/map.png?${query.toString()}`, {cache: 'no-store'});
      if (!response.ok) {
        let message = `Server returned ${response.status}.`;
        try { message = (await response.json()).error || message; } catch (_) {}
        throw new Error(message);
      }
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const step = query.get('step') || '000';
      link.href = downloadUrl;
      link.download = `baywind-${query.get('model')}-${query.get('field') || 'wind'}-f${pad(step)}-300dpi.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1500);
      setState('Map ready', 'is-loaded');
    } catch (error) {
      els.download.title = error.message || 'High-resolution export failed';
      setState('HD export failed', 'is-error');
      window.setTimeout(() => {
        if (!els.image.hidden) setState('Map ready', 'is-loaded');
        els.download.title = 'Download 300 DPI PNG';
      }, 4500);
    } finally {
      hdDownloadInProgress = false;
      label.textContent = previousLabel;
      els.download.disabled = els.image.hidden || !currentBlobUrl;
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
      els.model.innerHTML = '';
      for (const model of catalog) {
        const option = document.createElement('option');
        option.value = model.id;
        option.disabled = model.enabled === false;
        option.textContent = `${model.name} — ${model.resolution}${model.enabled === false ? ' (API key required)' : ''}`;
        els.model.appendChild(option);
      }
      // Default to the lighter 0.25° IFS file for faster uncached loads;
      // the 0.1° option remains available in the model selector.
      els.model.value = 'ifs_025';
      updateModelDetails();
      loadMap();
    } catch (error) {
      showError(error.message || 'Unable to load the model catalog.');
    }
  }

  els.model.addEventListener('change', () => { stopAnimation(); updateModelDetails(); loadMap(); });
  els.lead.addEventListener('input', () => {
    updateLeadLabel();
    window.clearTimeout(leadDebounce);
    leadDebounce = window.setTimeout(loadMap, 300);
  });
  els.prev.addEventListener('click', () => moveStep(-1));
  els.next.addEventListener('click', () => moveStep(1));
  els.play.addEventListener('click', () => animationTimer ? stopAnimation() : startAnimation());
  els.field.addEventListener('change', () => { stopAnimation(); fitStepToField(); updateFieldControls(); updateLeadLabel(); loadMap(); });
  els.speed.addEventListener('change', loadMap);
  els.stream.addEventListener('change', loadMap);
  els.units.addEventListener('change', loadMap);
  els.retry.addEventListener('click', loadMap);
  els.download.addEventListener('click', downloadCurrent);
  window.addEventListener('beforeunload', () => { if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl); });

  init();
})();
