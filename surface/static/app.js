const state = {
  models: [],
  assets: [],
  loras: [],
  activeRefs: new Set(),
  activeLoras: new Map(),
  quant: 'int4',
  memoryMode: 'resident',
  currentView: 'generate',
  generating: false,
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
  $$('button', root).forEach((btn) => {
    btn.addEventListener('click', () => {
      state[key] = btn.dataset.value;
      $$('button', root).forEach((item) => item.classList.toggle('active', item === btn));
    });
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
  select.innerHTML = state.models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('');
  const list = $('#modelsList');
  list.innerHTML = state.models.map((m) => `
    <div class="model-row" data-model-id="${esc(m.id)}">
      <div class="row-title">
        <span>${esc(m.name)}</span>
        <span class="pill">${esc(m.status)}</span>
      </div>
      <div class="row-meta">${esc(m.description)}</div>
      <div class="row-meta">quants: ${esc((m.quants || []).join(', '))}</div>
      <div class="row-actions">
        <button class="small-btn" data-select-model="${esc(m.id)}">Select</button>
        <button class="small-btn" data-download-model="${esc(m.id)}">${m.downloaded ? 'Downloaded' : 'Mark downloaded'}</button>
        <button class="small-btn" data-load-model="${esc(m.id)}">Load</button>
      </div>
    </div>
  `).join('');
  $$('[data-select-model]').forEach((btn) => btn.addEventListener('click', () => {
    select.value = btn.dataset.selectModel;
    setView('generate');
  }));
  $$('[data-download-model]').forEach((btn) => btn.addEventListener('click', async () => {
    await api.post('/api/models/download', currentLoadBody(btn.dataset.downloadModel));
    await refreshAll();
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
  if (outputs[0]) showOutput(outputs[0]);
}

function assetCard(asset) {
  const active = state.activeRefs.has(asset.id) ? ' active' : '';
  return `
    <div class="asset-card${active}" data-asset-id="${esc(asset.id)}" data-role="${esc(asset.role)}">
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
}

function renderLoras() {
  const renderRow = (lora, libraryOnly = false) => {
    const active = state.activeLoras.get(lora.id);
    const strength = active?.strength ?? 1;
    return `
      <div class="lora-row${active ? ' active' : ''}" data-lora-id="${esc(lora.id)}">
        <div class="row-title">
          <span>${esc(lora.name)}</span>
          <button class="small-btn" data-toggle-lora="${esc(lora.id)}">${active ? 'On' : 'Off'}</button>
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
  state.generating = Boolean(engine.running);
  $('#engineMode').textContent = engine.mode || 'mock';
  $('#loadedModel').textContent = engine.loaded_model_id || 'idle';
  $('#generateBtn').disabled = state.generating;
  $('#cancelBtn').disabled = !state.generating;
}

function renderLogs(items) {
  $('#logStrip').innerHTML = items.slice(-80).map((event) => {
    const t = new Date((event.ts || Date.now()) * 1000).toLocaleTimeString();
    return `<div class="log-line"><span>${esc(t)}</span><span>${esc(event.message || '')}</span></div>`;
  }).join('');
  $('#logStrip').scrollTop = $('#logStrip').scrollHeight;
}

function showOutput(asset) {
  $('#viewerTitle').textContent = asset.name;
  const meta = asset.meta || {};
  $('#viewerSub').textContent = meta.prompt || 'Output';
  $('#imageStage').innerHTML = `<img src="${esc(asset.url)}" alt="${esc(asset.name)}">`;
  $('#memoryValue').textContent = meta.memory_mode || 'mock';
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
    loras: Array.from(state.activeLoras.values()),
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
  ['load', 'ready', 'download', 'generate', 'progress', 'complete', 'asset', 'cancel', 'cancelled', 'error', 'lora'].forEach((kind) => {
    es.addEventListener(kind, async (event) => {
      const data = JSON.parse(event.data);
      addLogEvent(data);
      if (data.payload?.active_memory) $('#memoryValue').textContent = data.payload.active_memory;
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

