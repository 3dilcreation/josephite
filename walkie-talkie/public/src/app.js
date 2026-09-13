// UI wiring: identity, the mesh, the PTT button and the message log.

import { Mesh } from './mesh.js';
import { PushToTalk } from './ptt.js';
import { Player } from './player.js';
import { TIER, profileFor } from './link-quality.js';
import { loadWhisper, sttSupport } from './stt.js';
import { voicesReady } from './tts.js';
import { TYPE, encode, randomId } from './protocol.js';

const $ = (id) => document.getElementById(id);
const el = {
  signalDot: $('signal-dot'),
  name: $('name'),
  channel: $('channel'),
  settingsToggle: $('settings-toggle'),
  settings: $('settings'),
  sttProvider: $('stt-provider'),
  alwaysTranscribe: $('always-transcribe'),
  downloadVoicePack: $('download-voice-pack'),
  voicePackStatus: $('voice-pack-status'),
  tierBadge: $('tier-badge'),
  tierLabel: $('tier-label'),
  tierDetail: $('tier-detail'),
  peers: $('peers'),
  log: $('log'),
  interim: $('interim'),
  ptt: $('ptt'),
  pttMode: $('ptt-mode'),
  compose: $('compose'),
  textInput: $('text-input'),
};

const settings = loadSettings();
const selfId = settings.selfId;
const player = new Player();
let mesh = null;
let ptt = null;
let announcedForcedTier = false;

const TIER_COPY = {
  [TIER.GREEN]: ['CLEAR', 'streaming voice'],
  [TIER.AMBER]: ['WEAK', 'buffered low-rate voice'],
  [TIER.RED]: ['POOR', 'voice will be sent as text'],
};

const MODE_COPY = {
  stream: 'live voice',
  clip: 'buffered voice',
  transcribe: 'transcribing — your words go as text',
};

function loadSettings() {
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem('nearby-ptt') || '{}'); } catch { /* first run */ }
  return {
    selfId: stored.selfId || crypto.randomUUID(),
    name: stored.name || `radio-${Math.floor(Math.random() * 900 + 100)}`,
    channel: stored.channel || 'main',
    sttProvider: stored.sttProvider || 'webspeech',
    alwaysTranscribe: stored.alwaysTranscribe !== false,
  };
}

function saveSettings() {
  try { localStorage.setItem('nearby-ptt', JSON.stringify(settings)); } catch { /* private mode */ }
}

// ---- log ----------------------------------------------------------------

function addEntry({ kind, name, text, badge, mine = false }) {
  if (kind !== 'system') lastSystemText = null;
  const node = document.createElement('article');
  node.className = 'entry';
  node.dataset.kind = kind;
  node.dataset.mine = String(mine);

  const head = document.createElement('div');
  head.className = 'entry-head';
  head.innerHTML = `<b></b><span class="badge"></span>`;
  head.querySelector('b').textContent = name;
  head.querySelector('.badge').textContent = badge || kind;

  const body = document.createElement('div');
  body.className = 'entry-body';
  body.textContent = text;

  node.append(head, body);
  el.log.append(node);
  el.log.scrollTop = el.log.scrollHeight;
  while (el.log.children.length > 200) el.log.firstChild.remove();
  return node;
}

let lastSystemText = null;

function system(text) {
  // A missing transcription engine warns on every single transmission. Saying it
  // once is information; saying it twenty times buries the conversation.
  if (text === lastSystemText) return;
  lastSystemText = text;
  addEntry({ kind: 'system', name: 'system', text, badge: 'info' });
}

// ---- mesh lifecycle -----------------------------------------------------

function start() {
  mesh?.close();
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/signal`;
  mesh = new Mesh({
    signalUrl: url,
    selfId,
    displayName: settings.name,
    channel: settings.channel,
  });

  mesh.addEventListener('signal-state', (e) => {
    el.signalDot.dataset.state = e.detail.state;
  });
  mesh.addEventListener('peers', () => renderPeers());
  mesh.addEventListener('quality', () => renderPeers());

  mesh.addEventListener('message', (e) => {
    const { kind, from, name, text, meta } = e.detail;
    const badge = kind === 'voice-as-text'
      ? (meta.collapsed ? 'link dropped → text' : 'spoken → text')
      : 'text';
    addEntry({ kind, name, text, badge });
    // Spoken messages are read aloud; typed ones are not, so the channel does
    // not start narrating itself at you.
    if (kind === 'voice-as-text') player.enqueueSpoken(text, from);
  });

  mesh.addEventListener('audio-start', (e) => {
    player.beginLive(e.detail.id, e.detail.meta);
  });
  mesh.addEventListener('audio-chunk', (e) => {
    player.feedLive(e.detail.id, e.detail.seq, e.detail.chunk);
  });
  mesh.addEventListener('audio-end', (e) => {
    const { name, meta, blob, id } = e.detail;
    const wasLive = player.endLive(id);
    const seconds = ((meta.durationMs || 0) / 1000).toFixed(1);
    const kb = Math.round((meta.bytesReceived || 0) / 1024);
    addEntry({
      kind: 'audio',
      name,
      text: meta.transcript || `${seconds}s of voice`,
      badge: meta.transcript
        ? `voice ${seconds}s · ${kb} KB + transcript`
        : `voice ${seconds}s · ${kb} KB`,
    });
    if (!wasLive) player.enqueueClip(blob);
  });

  const forced = new URLSearchParams(location.search).get('tier');
  if (forced && Object.values(TIER).includes(forced)) {
    mesh.forcedTier = forced;
    // Changing your name or channel restarts the mesh; the override note is a
    // property of the page load, not of each restart.
    if (!announcedForcedTier) {
      announcedForcedTier = true;
      system(`link grade pinned to ${forced.toUpperCase()} by ?tier= — measurements ignored`);
    }
  }

  mesh.connect();
  system(`joined #${settings.channel} as ${settings.name}`);

  ptt = new PushToTalk({
    mesh,
    selfId,
    displayName: settings.name,
    sttProvider: settings.sttProvider,
    alwaysTranscribe: settings.alwaysTranscribe,
  });

  ptt.addEventListener('interim', (e) => {
    el.interim.hidden = !e.detail.text;
    el.interim.textContent = e.detail.text;
  });
  ptt.addEventListener('state', (e) => {
    const { transmitting, mode } = e.detail;
    el.ptt.dataset.live = String(transmitting);
    if (transmitting) {
      el.ptt.dataset.mode = mode;
      el.pttMode.textContent = MODE_COPY[mode];
    } else {
      el.interim.hidden = true;
      renderTier();
    }
  });
  ptt.addEventListener('warn', (e) => system(e.detail.message));
  ptt.addEventListener('sent', (e) => {
    const { kind, text, transcript, durationMs, delivered } = e.detail;
    if (kind === 'voice-as-text') {
      addEntry({
        kind: 'voice-as-text',
        name: 'you',
        text,
        badge: delivered ? `spoken → text → ${delivered} peer(s)` : 'spoken → text (nobody in range)',
        mine: true,
      });
    } else {
      const seconds = (durationMs / 1000).toFixed(1);
      addEntry({
        kind: 'audio',
        name: 'you',
        text: transcript || `${seconds}s of voice`,
        badge: `voice ${seconds}s · ${Math.round(e.detail.bytes / 1024)} KB`,
        mine: true,
      });
    }
  });

  renderTier();
}

function renderPeers() {
  const peers = mesh?.peerList || [];
  el.peers.replaceChildren(...peers.map((p) => {
    const li = document.createElement('li');
    li.dataset.tier = p.tier;
    const rtt = p.rtt === null ? '—' : `${Math.round(p.rtt)}ms`;
    li.textContent = `${p.name} · ${rtt}`;
    return li;
  }));
  renderTier();
}

function renderTier() {
  if (ptt?.isTransmitting) return;
  const peers = mesh?.peerList || [];
  const tier = peers.length ? mesh.worstTier() : TIER.RED;
  const [label, detail] = TIER_COPY[tier];
  el.tierBadge.dataset.tier = tier;
  // With nobody connected the grade is not "poor", it is undefined -- saying so
  // keeps the badge from crying wolf about a link that does not exist yet.
  el.tierLabel.textContent = peers.length ? label : 'NO PEERS';
  el.tierDetail.textContent = peers.length
    ? `${peers.length} peer${peers.length > 1 ? 's' : ''} · ${detail}`
    : 'nobody in range yet';
  el.pttMode.textContent = peers.length ? MODE_COPY[profileFor(tier).mode] : 'nobody to talk to yet';
}

// ---- controls -----------------------------------------------------------

let pressed = false;

async function beginTalk(event) {
  event.preventDefault();
  if (pressed) return;
  pressed = true;
  try {
    await ptt.press();
  } catch (err) {
    pressed = false;
    system(`microphone unavailable: ${err.message}`);
  }
}

async function endTalk(event) {
  event?.preventDefault();
  if (!pressed) return;
  pressed = false;
  await ptt.release();
}

el.ptt.addEventListener('pointerdown', beginTalk);
el.ptt.addEventListener('pointerup', endTalk);
el.ptt.addEventListener('pointercancel', endTalk);
el.ptt.addEventListener('pointerleave', endTalk);
// Space bar is the desktop equivalent of holding the button.
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && document.activeElement?.tagName !== 'INPUT') beginTalk(e);
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && document.activeElement?.tagName !== 'INPUT') endTalk(e);
});

el.compose.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = el.textInput.value.trim();
  if (!text || !mesh) return;
  const delivered = mesh.broadcast(encode({
    type: TYPE.TEXT,
    id: randomId(),
    meta: { from: mesh.selfId, name: settings.name },
    payload: text,
  }));
  addEntry({ kind: 'text', name: 'you', text, badge: `text → ${delivered} peer(s)`, mine: true });
  el.textInput.value = '';
});

el.settingsToggle.addEventListener('click', () => {
  const open = el.settings.hidden;
  el.settings.hidden = !open;
  el.settingsToggle.setAttribute('aria-expanded', String(open));
});

el.name.value = settings.name;
el.channel.value = settings.channel;
el.sttProvider.value = settings.sttProvider;
el.alwaysTranscribe.checked = settings.alwaysTranscribe;

el.name.addEventListener('change', () => {
  settings.name = el.name.value.trim() || settings.name;
  el.name.value = settings.name;
  saveSettings();
  start();
});
el.channel.addEventListener('change', () => {
  settings.channel = el.channel.value.trim() || 'main';
  el.channel.value = settings.channel;
  saveSettings();
  start();
});
el.sttProvider.addEventListener('change', () => {
  settings.sttProvider = el.sttProvider.value;
  saveSettings();
  if (ptt) ptt.sttProvider = settings.sttProvider;
});
el.alwaysTranscribe.addEventListener('change', () => {
  settings.alwaysTranscribe = el.alwaysTranscribe.checked;
  saveSettings();
  if (ptt) ptt.alwaysTranscribe = settings.alwaysTranscribe;
});

el.downloadVoicePack.addEventListener('click', async () => {
  el.downloadVoicePack.disabled = true;
  el.voicePackStatus.textContent = 'downloading whisper-tiny…';
  try {
    await loadWhisper();
    el.voicePackStatus.textContent = 'Voice pack ready. Transcription now runs on this device.';
    settings.sttProvider = 'whisper';
    el.sttProvider.value = 'whisper';
    saveSettings();
    if (ptt) ptt.sttProvider = 'whisper';
  } catch (err) {
    el.voicePackStatus.textContent = `download failed: ${err.message}`;
    el.downloadVoicePack.disabled = false;
  }
});

// ---- boot ---------------------------------------------------------------

const support = sttSupport();
if (!support.webspeech) {
  el.sttProvider.querySelector('option[value="webspeech"]').disabled = true;
  settings.sttProvider = 'whisper';
  el.sttProvider.value = 'whisper';
}

voicesReady();
start();
setInterval(renderTier, 2000);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {
    system('offline caching unavailable (needs HTTPS)');
  });
}
