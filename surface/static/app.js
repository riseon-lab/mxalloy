const state = {
  models: [],
  assets: [],
  loras: [],
  activeRefs: new Set(),
  activeLoras: new Map(),
  selectedModelId: '',
  lastModelDefaultsApplied: '',
  currentOutputId: '',
  quant: 'auto',
  memoryMode: 'auto',
  currentView: 'generate',
  generating: false,
  engine: {},
  memorySamples: [],
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const api = {
  get: async (url) => (await fetch(url)).json(),
  post: async (url, body) => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  postForm: async (url, form) => {
    const res = await fetch(url, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  del: async (url) => (await fetch(url, { method: 'DELETE' })).json(),
};

function init() {
  bindUi();
  connectEvents();
  refreshAll();
}

function bindUi() {
  $$('.tab').forEach((btn) => btn.addEventListener('click', () => setView(btn.dataset.view)));
  $$('[data-view-jump]').forEach((btn) => btn.addEventListener('click', () => setView(btn.dataset.viewJump)));
  bindSegment($('#quantGroup'), 'quant');
  bindSegment($('#memoryGroup'), 'memoryMode');
  $('#modelSelect').addEventListener('change', () => {
    state.selectedModelId = $('#modelSelect').value;
    applySelectedModelDefaults();
    renderModelControls();
    renderLoras();
    syncActionAvailability();
  });

  $('#prompt').addEventListener('input', () => {
    $('#promptCount').textContent = $('#prompt').value.length;
  });
  $('#steps').addEventListener('input', () => $('#stepsValue').textContent = $('#steps').value);
  $('#guidance').addEventListener('input', () => $('#guidanceValue').textContent = $('#guidance').value);
  $('#swapSize').addEventListener('click', () => {
    const w = $('#width').value;
    $('#width').value = $('#height').value;
    $('#height').value = w;
    syncPreset();
  });
  $$('#sizePresets button').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('#width').value = btn.dataset.w;
      $('#height').value = btn.dataset.h;
      syncPreset();
    });
  });
  $('#width').addEventListener('input', syncPreset);
  $('#height').addEventListener('input', syncPreset);

  $('#generateBtn').addEventListener('click', generate);
  $('#cancelBtn').addEventListener('click', cancel);
  $('#refUpload').addEventListener('change', (event) => uploadFile(event, '/api/assets/upload'));
  $('#loraUpload').addEventListener('change', (event) => uploadFile(event, '/api/loras'));
  $('#loraUploadQuick').addEventListener('change', (event) => uploadFile(event, '/api/loras'));
  $('#settingsForm').addEventListener('submit', saveSettings);
  $('#testHf').addEventListener('click', testHfToken);
  $('#clearHf').addEventListener('click', clearHfToken);
}

function bindSegment(root, key) {
  root.addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-value]');
    if (!btn || !root.contains(btn)) return;
    state[key] = btn.dataset.value;
    renderModelControls();
  });
}

async function refreshAll() {
  const [models, assets, loras, settings, status] = await Promise.all([
    api.get('/api/models'),
    api.get('/api/assets'),
    api.get('/api/loras'),
    api.get('/api/settings'),
    api.get('/api/status'),
  ]);
  state.models = models.models || [];
  state.assets = assets.assets || [];
  state.loras = loras.loras || [];
  renderModels();
  renderAssets();
  renderLoras();
  renderSettings(settings);
  renderStatus(status);
  renderLogs(status.logs || []);
}

function setView(view) {
  state.currentView = view;
  $$('.tab').forEach((btn) => btn.classList.toggle('active', btn.dataset.view === view));
  $$('.side-view').forEach((panel) => panel.classList.toggle('active', panel.dataset.viewPanel === view));
}

function renderModels() {
  const select = $('#modelSelect');
  const previous = state.selectedModelId || select.value;
  select.innerHTML = state.models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('');
  if (state.models.some((m) => m.id === previous)) select.value = previous;
  else if (state.models[0]) select.value = state.models[0].id;
  state.selectedModelId = select.value;
  applySelectedModelDefaults();
  renderModelControls();

  const list = $('#modelsList');
  list.innerHTML = state.models.map((m) => `
    <div class="model-row" data-model-id="${esc(m.id)}">
      <div class="row-title">
        <span>${esc(m.name)}</span>
        <span class="pill">${m.available ? 'local' : 'missing'}</span>
      </div>
      <div class="row-meta">${esc(m.description)}</div>
      <div class="row-meta">quants: ${esc((m.quants || []).join(', '))} · modes: ${esc((m.memory_modes || []).join(', '))}</div>
      <div class="row-meta">LoRA: ${m.supports_lora ? esc((m.lora_formats || ['supported']).join(', ')) : 'not supported'}</div>
      <div class="row-meta">${esc(strategySummary(m.recommended_strategy))}</div>
      <div class="row-meta">${m.local_path ? esc(m.local_path) : 'Set the Hugging Face cache path in Settings once the model is downloaded.'}</div>
      <div class="row-actions">
        <button class="small-btn" data-select-model="${esc(m.id)}">Select</button>
        <button class="small-btn" data-load-model="${esc(m.id)}" ${m.available ? '' : 'disabled'}>Load</button>
      </div>
    </div>
  `).join('');
  $$('[data-select-model]').forEach((btn) => btn.addEventListener('click', () => {
    select.value = btn.dataset.selectModel;
    state.selectedModelId = select.value;
    applySelectedModelDefaults();
    renderModelControls();
    renderLoras();
    syncActionAvailability();
    setView('generate');
  }));
  $$('[data-load-model]').forEach((btn) => btn.addEventListener('click', async () => {
    await api.post('/api/load', currentLoadBody(btn.dataset.loadModel));
    await refreshAll();
  }));
}

function renderAssets() {
  const refs = state.assets.filter((a) => a.role === 'reference');
  const outputs = state.assets.filter((a) => a.role === 'output');
  $('#referenceGrid').innerHTML = refs.length ? refs.map(assetCard).join('') : '<p class="hint">No references yet.</p>';
  $('#recentOutputs').innerHTML = outputs.slice(0, 6).map(assetCard).join('') || '<p class="hint">No outputs yet.</p>';
  $('#outputsGrid').innerHTML = outputs.map(assetCard).join('') || '<p class="hint">No outputs yet.</p>';
  bindAssetCards();
  const selectedOutput = outputs.find((asset) => asset.id === state.currentOutputId) || outputs[0];
  if (selectedOutput) showOutput(selectedOutput);
  else clearOutput();
}

function assetCard(asset) {
  const active = state.activeRefs.has(asset.id) || state.currentOutputId === asset.id ? ' active' : '';
  return `
    <div class="asset-card${active}" data-asset-id="${esc(asset.id)}" data-role="${esc(asset.role)}">
      <button class="asset-delete" data-delete-asset="${esc(asset.id)}" type="button" aria-label="Delete ${esc(asset.name)}">&times;</button>
      <img src="${esc(asset.url)}" alt="">
      <div class="asset-name">${esc(asset.name)}</div>
    </div>
  `;
}

function bindAssetCards() {
  $$('.asset-card').forEach((card) => {
    card.addEventListener('click', () => {
      const asset = state.assets.find((item) => item.id === card.dataset.assetId);
      if (!asset) return;
      if (asset.role === 'reference') {
        if (state.activeRefs.has(asset.id)) state.activeRefs.delete(asset.id);
        else state.activeRefs.add(asset.id);
        renderAssets();
      } else {
        showOutput(asset);
      }
    });
  });
  $$('[data-delete-asset]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      event.stopPropagation();
      await deleteAsset(btn.dataset.deleteAsset);
    });
  });
}

async function deleteAsset(id) {
  const [scope, ...pathParts] = String(id || '').split('/');
  if (!scope || !pathParts.length) return;
  const filename = pathParts.map(encodeURIComponent).join('/');
  await api.del(`/api/assets/${encodeURIComponent(scope)}/${filename}`);
  state.activeRefs.delete(id);
  if (state.currentOutputId === id) state.currentOutputId = '';
  await refreshAll();
}

function renderLoras() {
  const loraSupported = selectedModel()?.supports_lora !== false;
  const renderRow = (lora, libraryOnly = false) => {
    const active = state.activeLoras.get(lora.id);
    const strength = active?.strength ?? 1;
    return `
      <div class="lora-row${active ? ' active' : ''}" data-lora-id="${esc(lora.id)}">
        <div class="row-title">
          <span>${esc(lora.name)}</span>
          <button class="small-btn" data-toggle-lora="${esc(lora.id)}" ${loraSupported ? '' : 'disabled'}>${active ? 'On' : 'Off'}</button>
        </div>
        <div class="row-meta">${esc(lora.id)} · ${lora.size_mb || 0} MB</div>
        ${active ? `<label class="slider-row"><span>Strength <b>${strength}</b></span><input type="range" min="0" max="2" step="0.05" value="${strength}" data-lora-strength="${esc(lora.id)}"></label>` : ''}
        ${libraryOnly ? `<button class="small-btn" data-delete-lora="${esc(lora.id)}">Remove file</button>` : ''}
      </div>
    `;
  };
  $('#activeLoras').innerHTML = state.loras.map((lora) => renderRow(lora)).join('') || '<p class="hint">No LoRAs added.</p>';
  $('#loraLibrary').innerHTML = state.loras.map((lora) => renderRow(lora, true)).join('') || '<p class="hint">No LoRA files in the folder.</p>';
  $$('[data-toggle-lora]').forEach((btn) => btn.addEventListener('click', () => toggleLora(btn.dataset.toggleLora)));
  $$('[data-lora-strength]').forEach((input) => input.addEventListener('input', () => {
    const active = state.activeLoras.get(input.dataset.loraStrength);
    if (!active) return;
    active.strength = Number(input.value);
    renderLoras();
  }));
  $$('[data-delete-lora]').forEach((btn) => btn.addEventListener('click', async () => {
    await api.del(`/api/loras/${encodeURIComponent(btn.dataset.deleteLora)}`);
    state.activeLoras.delete(btn.dataset.deleteLora);
    await refreshAll();
  }));
}

function toggleLora(id) {
  if (selectedModel()?.supports_lora === false) return;
  if (state.activeLoras.has(id)) state.activeLoras.delete(id);
  else state.activeLoras.set(id, { id, strength: 1, enabled: true });
  renderLoras();
}

function renderSettings(settings) {
  $('#modelCachePath').value = settings.model_cache_path || '';
  $('#loraFolderPath').value = settings.lora_folder_path || '';
  $('#outputFolderPath').value = settings.output_folder_path || '';
  $('#referenceFolderPath').value = settings.reference_folder_path || '';
  $('#settingsHint').textContent = settings.hf_token_set
    ? 'HF token saved by the local backend.'
    : 'Tokens are stored by the local backend, never browser localStorage.';
}

function renderStatus(status) {
  const engine = status.engine || {};
  state.engine = engine;
  state.generating = Boolean(engine.running);
  $('#engineMode').textContent = engine.mode || 'mock';
  $('#loadedModel').textContent = engine.loaded_model_id || 'idle';
  renderMemory(engine.memory);
  syncActionAvailability();
  $('#cancelBtn').disabled = !state.generating;
}

function strategySummary(strategy) {
  if (!strategy) return 'planner: unavailable';
  const fit = strategy.fits ? 'fits' : 'does not fit';
  return `planner: ${strategy.precision}/${strategy.memory_mode} · est ${strategy.estimated_peak_gb} GB · ${fit}`;
}

function renderLogs(items) {
  $('#logStrip').innerHTML = items.slice(-80).map((event) => {
    const t = new Date((event.ts || Date.now()) * 1000).toLocaleTimeString();
    return `<div class="log-line"><span>${esc(t)}</span><span>${esc(event.message || '')}</span></div>`;
  }).join('');
  $('#logStrip').scrollTop = $('#logStrip').scrollHeight;
}

function showOutput(asset) {
  state.currentOutputId = asset.id;
  $('#viewerTitle').textContent = asset.name;
  const meta = asset.meta || {};
  $('#viewerSub').textContent = meta.prompt || 'Output';
  $('#imageStage').innerHTML = `<img src="${esc(asset.url)}" alt="${esc(asset.name)}">`;
}

function clearOutput() {
  state.currentOutputId = '';
  $('#viewerTitle').textContent = 'Ready';
  $('#viewerSub').textContent = 'FLUX.2 klein 4B is ready for local MLX testing.';
  $('#imageStage').innerHTML = `
    <div class="empty-state">
      <div class="empty-mark"></div>
      <p>Generated images will appear here.</p>
    </div>
  `;
}

async function generate() {
  const body = {
    ...currentLoadBody($('#modelSelect').value),
    prompt: $('#prompt').value,
    negative_prompt: $('#negativePrompt').value,
    width: Number($('#width').value),
    height: Number($('#height').value),
    steps: Number($('#steps').value),
    guidance: Number($('#guidance').value),
    seed: $('#seed').value ? Number($('#seed').value) : null,
    refs: Array.from(state.activeRefs),
    loras: selectedModel()?.supports_lora === false ? [] : Array.from(state.activeLoras.values()),
  };
  try {
    await api.post('/api/generate', body);
    state.generating = true;
    $('#generateBtn').disabled = true;
    $('#cancelBtn').disabled = false;
  } catch (err) {
    addLocalLog(`Generate failed: ${err.message}`);
  }
}

async function cancel() {
  await api.post('/api/cancel');
  await refreshAll();
}

async function uploadFile(event, url) {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  try {
    await api.postForm(url, form);
    await refreshAll();
  } catch (err) {
    addLocalLog(`Upload failed: ${err.message}`);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const body = {
    model_cache_path: $('#modelCachePath').value,
    lora_folder_path: $('#loraFolderPath').value,
    output_folder_path: $('#outputFolderPath').value,
    reference_folder_path: $('#referenceFolderPath').value,
  };
  if ($('#hfToken').value.trim()) body.hf_token = $('#hfToken').value.trim();
  const res = await api.post('/api/settings', body);
  $('#hfToken').value = '';
  renderSettings(res);
  await refreshAll();
}

async function testHfToken() {
  const res = await api.post('/api/settings/hf-token/test', {});
  $('#settingsHint').textContent = res.message;
}

async function clearHfToken() {
  const res = await api.post('/api/settings', { clear_hf_token: true });
  renderSettings(res);
}

function connectEvents() {
  const es = new EventSource('/api/events');
  es.onmessage = () => {};
  ['load', 'ready', 'download', 'generate', 'progress', 'complete', 'asset', 'cancel', 'cancelled', 'error', 'lora', 'memory'].forEach((kind) => {
    es.addEventListener(kind, async (event) => {
      const data = JSON.parse(event.data);
      addLogEvent(data);
      if (data.payload?.memory) renderMemory(data.payload.memory);
      else if (data.payload?.active_memory) $('#memoryValue').textContent = data.payload.active_memory;
      if (['complete', 'asset', 'cancelled', 'error'].includes(kind)) {
        await refreshAll();
      }
    });
  });
}

function addLogEvent(event) {
  const log = document.createElement('div');
  log.className = 'log-line';
  const t = new Date((event.ts || Date.now()) * 1000).toLocaleTimeString();
  log.innerHTML = `<span>${esc(t)}</span><span>${esc(event.message || '')}</span>`;
  $('#logStrip').append(log);
  $('#logStrip').scrollTop = $('#logStrip').scrollHeight;
}

function addLocalLog(message) {
  addLogEvent({ ts: Date.now() / 1000, message });
}

function currentLoadBody(modelId) {
  return {
    model_id: modelId || $('#modelSelect').value,
    quant: state.quant,
    memory_mode: state.memoryMode,
  };
}

function selectedModel() {
  return state.models.find((model) => model.id === ($('#modelSelect').value || state.selectedModelId));
}

function applySelectedModelDefaults() {
  const model = selectedModel();
  if (!model) return;
  if (!(model.quants || []).includes(state.quant)) state.quant = (model.quants || ['int4'])[0];
  if (!(model.memory_modes || []).includes(state.memoryMode)) {
    state.memoryMode = (model.memory_modes || ['resident'])[0];
  }
  if (model.supports_lora === false && state.activeLoras.size) {
    state.activeLoras.clear();
    renderLoras();
  }
  if (state.lastModelDefaultsApplied !== model.id) {
    $('#width').value = model.default_width || $('#width').value;
    $('#height').value = model.default_height || $('#height').value;
    $('#steps').value = model.default_steps || $('#steps').value;
    $('#guidance').value = model.default_guidance ?? $('#guidance').value;
    $('#stepsValue').textContent = $('#steps').value;
    $('#guidanceValue').textContent = $('#guidance').value;
    state.lastModelDefaultsApplied = model.id;
    syncPreset();
  }
}

function renderModelControls() {
  const model = selectedModel();
  if (!model) {
    $('#modelMeta').textContent = 'No models registered.';
    $('#quantGroup').innerHTML = '';
    $('#memoryGroup').innerHTML = '';
    return;
  }
  $('#modelMeta').textContent = model.available
    ? `Local cache ready · LoRA ${model.supports_lora === false ? 'off' : 'ready'}`
    : `Missing from the configured Hugging Face cache · LoRA ${model.supports_lora === false ? 'off' : 'ready'}`;
  $('#modelMeta').title = model.local_path || '';
  renderSegment($('#quantGroup'), model.quants || [], state.quant, quantLabel, model.notes || {});
  renderSegment($('#memoryGroup'), model.memory_modes || [], state.memoryMode, modeLabel, model.notes || {});
}

function renderSegment(root, values, activeValue, labeler, notes) {
  root.style.setProperty('--segment-count', Math.max(1, values.length));
  root.innerHTML = values.map((value) => `
    <button data-value="${esc(value)}" class="${value === activeValue ? 'active' : ''}" title="${esc(notes[value] || '')}">
      ${esc(labeler(value))}
    </button>
  `).join('');
}

function quantLabel(value) {
  return ({ auto: 'auto', int4: 'int4', int8: 'int8', bf16: 'bf16', fp16: 'fp16' })[value] || value;
}

function modeLabel(value) {
  return ({ auto: 'auto', resident: 'resident', staged: 'staged', survival: 'survival' })[value] || value;
}

function syncActionAvailability() {
  const model = selectedModel();
  const needsLocalModel = (state.engine.mode || 'real') !== 'mock';
  const canUseModel = Boolean(model) && (model.available || !needsLocalModel);
  $('#generateBtn').disabled = state.generating || !canUseModel;
}

function renderMemory(memory) {
  if (!memory) return;
  const active = Number(memory.active_gb || 0);
  const peak = Number(memory.peak_gb || 0);
  const cache = Number(memory.cache_gb || 0);
  const percent = Number.isFinite(Number(memory.percent))
    ? Number(memory.percent)
    : (peak > 0 ? Math.min(100, active / peak * 100) : 0);
  $('#memoryValue').textContent = memory.available === false
    ? 'unavailable'
    : `${active.toFixed(2)} GB`;
  $('#memoryFill').style.width = `${Math.max(0, Math.min(100, percent)).toFixed(1)}%`;
  $('#memoryPeak').textContent = `peak ${peak.toFixed(2)} GB`;
  $('#memoryCache').textContent = `cache ${cache.toFixed(2)} GB`;
  state.memorySamples.push(Math.max(0, Math.min(100, percent)));
  state.memorySamples = state.memorySamples.slice(-36);
  $('#memoryStream').innerHTML = state.memorySamples.map((sample) => (
    `<span style="height:${Math.max(4, sample).toFixed(1)}%"></span>`
  )).join('');
}

function syncPreset() {
  const w = String($('#width').value);
  const h = String($('#height').value);
  $$('#sizePresets button').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.w === w && btn.dataset.h === h);
  });
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

init();
