(() => {
  const els = {
    model: document.getElementById('model-select'),
    modelResolution: document.getElementById('model-resolution'),
    modelSource: document.getElementById('model-source'),
    modelNote: document.getElementById('model-note'),
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
  let animationTimer = null;
  let requestController = null;
  let requestSequence = 0;
  let leadDebounce = null;

  const pad = (value, length = 3) => String(value).padStart(length, '0');
  const currentModel = () => catalog.find(model => model.id === els.model.value);
  const selectedStep = () => {
    const model = currentModel();
    return model ? model.steps[Number(els.lead.value)] : 24;
  };

  function setState(text, stateClass = '') {
    els.state.className = `render-state ${stateClass}`.trim();
    els.state.lastElementChild.textContent = text;
  }

  function updateClock() {
    const now = new Date();
    els.clock.textContent = `UTC ${new Intl.DateTimeFormat('en-GB', {hour: '2-digit', minute: '2-digit', timeZone: 'UTC'}).format(now)}`;
  }

  function updateModelDetails() {
    const model = currentModel();
    if (!model) return;
    els.modelResolution.textContent = model.resolution;
    els.modelSource.textContent = model.source;
    els.modelNote.textContent = model.note || '';
    els.mapModel.textContent = model.name;
    els.lead.min = '0';
    els.lead.max = String(Math.max(0, model.steps.length - 1));
    els.lead.value = String(Math.max(0, model.steps.indexOf(24) >= 0 ? model.steps.indexOf(24) : 0));
    els.leadEnd.textContent = `+${model.steps[model.steps.length - 1]} h`;
    updateLeadLabel();
  }

  function updateLeadLabel() {
    const step = selectedStep();
    els.leadLabel.textContent = `+${pad(step)} h`;
    els.mapStep.textContent = `F${pad(step)}`;
  }

  function showLoading() {
    els.error.hidden = true;
    els.placeholder.hidden = true;
    els.loading.hidden = false;
    els.image.hidden = true;
    els.download.disabled = true;
    setState('Rendering…', 'is-loading');
  }

  function showError(message) {
    els.loading.hidden = true;
    els.image.hidden = true;
    els.error.hidden = false;
    els.errorMessage.textContent = message;
    setState('Unavailable', 'is-error');
  }

  async function loadMap() {
    const model = currentModel();
    if (!model) return;
    if (!els.speed.checked && !els.stream.checked) {
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
      speed: els.speed.checked ? '1' : '0',
      streamlines: els.stream.checked ? '1' : '0',
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
      els.image.alt = `${model.name}, ${model.resolution}, 10 metre wind forecast at hour ${selectedStep()}`;
      els.image.hidden = false;
      els.loading.hidden = true;
      els.error.hidden = true;
      els.placeholder.hidden = true;
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
    const model = currentModel();
    if (!model || model.steps.length < 2) return;
    els.play.classList.add('is-playing');
    els.play.querySelector('span').textContent = 'Pause animation';
    els.play.querySelector('.play-icon').innerHTML = '<path d="M6 4h3v12H6zM12 4h3v12h-3z"/>';
    let index = Number(els.lead.value);
    animationTimer = window.setInterval(() => {
      index = (index + 1) % model.steps.length;
      els.lead.value = String(index);
      updateLeadLabel();
      loadMap();
    }, 4200);
  }

  function moveStep(direction) {
    const model = currentModel();
    if (!model) return;
    const next = Math.min(model.steps.length - 1, Math.max(0, Number(els.lead.value) + direction));
    els.lead.value = String(next);
    updateLeadLabel();
    loadMap();
  }

  function downloadCurrent() {
    if (!currentBlobUrl) return;
    const link = document.createElement('a');
    link.href = currentBlobUrl;
    link.download = `baywind-${els.model.value}-f${pad(selectedStep())}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
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
        option.textContent = `${model.name} — ${model.resolution}`;
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
  els.speed.addEventListener('change', loadMap);
  els.stream.addEventListener('change', loadMap);
  els.units.addEventListener('change', loadMap);
  els.retry.addEventListener('click', loadMap);
  els.download.addEventListener('click', downloadCurrent);
  window.addEventListener('beforeunload', () => { if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl); });

  init();
})();
