const $ = selector => document.querySelector(selector);
const input = $('#video');
const button = $('#upload');
const status = $('#status');
const memories = $('#memories');
const recordCount = $('#record-count');
const questionInput = $('#question');
const askButton = $('#ask');
const answer = $('#answer');
const startButton = $('#start-observation');
const stopButton = $('#stop-observation');
const timer = $('#timer');
const liveStatus = $('#live-status');
const systemState = $('#system-state');
const liveFeed = $('#live-feed');
const cameraPlaceholder = $('#camera-placeholder');
const responseText = $('#response-text');
const sendResponseButton = $('#send-response');
const listenButton = $('#listen');
const listeningIndicator = $('#listening-indicator');
const caregiverEnabled = $('#caregiver-enabled');
const caregiverRecipient = $('#caregiver-recipient');
const caregiverDelay = $('#caregiver-delay');
const caregiverLimit = $('#caregiver-limit');

let currentSessionId = null;
let recognition = null;

function esc(value) {
  const node = document.createElement('div');
  node.textContent = value ?? '';
  return node.innerHTML;
}

function formatTime(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function labelState(value) {
  return (value || 'unknown').replaceAll('_', ' ');
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function renderLive(payload) {
  const running = Boolean(payload.running);
  startButton.disabled = running;
  stopButton.disabled = !running;
  timer.disabled = running;
  caregiverEnabled.disabled = running;
  caregiverRecipient.disabled = running || !caregiverEnabled.checked;
  caregiverDelay.disabled = running || !caregiverEnabled.checked;
  caregiverLimit.disabled = running || !caregiverEnabled.checked;
  systemState.textContent = running ? 'Observation running' : 'Observation stopped';
  systemState.classList.toggle('active', running);
  liveStatus.textContent = payload.error
    ? payload.error
    : running
      ? `Camera ${payload.camera_source} is live. Temporal perception samples every few frames.`
      : 'Camera is not running.';

  const cameraHealth = $('#camera-health');
  cameraHealth.textContent = labelState(payload.camera_state);
  cameraHealth.dataset.state = payload.camera_state;
  $('#processed-count').textContent = payload.processed_frames ?? 0;
  $('#dropped-count').textContent = payload.dropped_frames ?? 0;
  $('#latency').textContent = payload.last_perception_ms == null
    ? '—'
    : `${payload.last_perception_ms} ms`;

  if (payload.session_id !== currentSessionId) {
    currentSessionId = payload.session_id;
    if (running) {
      liveFeed.src = `/api/live/feed?session=${encodeURIComponent(currentSessionId)}&t=${Date.now()}`;
    }
  }
  liveFeed.hidden = !running;
  cameraPlaceholder.hidden = running;

  renderEpisode(payload.episode);
  renderTimeline(payload.timeline || []);
  renderEvidence(payload.evidence || []);
  renderConversation(payload.checkin, payload.session_id);
  renderCaregiverOutbox(payload.caregiver_preview, payload.caregiver_outbox || []);
}

function renderEpisode(episode) {
  if (!episode) {
    $('#activity-name').textContent = 'Waiting for activity';
    $('#episode-state').textContent = 'No episode';
    $('#episode-copy').textContent = 'Tea preparation requires repeated observations of a person, cup, preparation object, and motion. A single detection will not create an episode.';
    $('#episode-started').textContent = '—';
    $('#episode-confidence').textContent = '—';
    $('#completion-source').textContent = '—';
    $('#reminder-count').textContent = '0';
    return;
  }
  $('#activity-name').textContent = episode.display_name;
  $('#episode-state').textContent = labelState(episode.state);
  $('#episode-state').dataset.state = episode.state;
  $('#episode-copy').textContent = episode.state === 'completed'
    ? 'This episode is closed. Its completion source remains visible below.'
    : 'This memory updates as new camera evidence arrives; absence is counted only while the camera view is usable.';
  $('#episode-started').textContent = formatTime(episode.started_at);
  $('#episode-confidence').textContent = `${Math.round((episode.confidence || 0) * 100)}%`;
  $('#completion-source').textContent = episode.completion_source
    ? labelState(episode.completion_source)
    : 'Not completed';
  $('#reminder-count').textContent = episode.reminder_count ?? 0;
}

function renderTimeline(items) {
  const timeline = $('#timeline');
  if (!items.length) {
    timeline.innerHTML = '<li class="empty">No episode transitions yet.</li>';
    return;
  }
  timeline.innerHTML = items.map(item => `
    <li>
      <time>${esc(formatTime(item.created_at))}</time>
      <div><strong>${esc(labelState(item.to_state))}</strong><p>${esc(item.reason)}</p></div>
    </li>`).join('');
}

function renderEvidence(items) {
  const evidence = $('#evidence');
  if (!items.length) {
    evidence.innerHTML = '<p class="empty">No episode evidence yet.</p>';
    return;
  }
  evidence.innerHTML = items.slice().reverse().map(item => {
    const facts = item.facts?.observed || {};
    const labels = (facts.object_labels || []).join(', ') || 'No recognised objects';
    const image = item.frame_url
      ? `<img loading="lazy" src="${esc(item.frame_url)}" alt="Evidence captured at ${esc(formatTime(item.observed_at))}">`
      : '<div class="evidence-unavailable">No image</div>';
    const person = facts.person_present === true
      ? 'visible'
      : facts.person_present === false ? 'not detected' : 'unknown';
    return `<article>${image}<div><time>${esc(formatTime(item.observed_at))}</time><p>${esc(labels)}</p><span>${esc(labelState(item.role))} · person ${person}</span></div></article>`;
  }).join('');
}

function renderConversation(checkin, sessionId) {
  const conversation = $('#conversation');
  const quickButtons = document.querySelectorAll('[data-response]');
  const awaiting = checkin?.status === 'awaiting_response';
  quickButtons.forEach(item => { item.disabled = !awaiting; });
  sendResponseButton.disabled = !awaiting;
  listenButton.disabled = !awaiting;
  if (!checkin) {
    conversation.innerHTML = '<p class="empty">The simulated assistant will speak here when a check-in is due.</p>';
    return;
  }
  const response = checkin.response_text
    ? `<div class="message user"><span>You</span><p>${esc(checkin.response_text)}</p></div>`
    : '';
  conversation.innerHTML = `<div class="message assistant"><span>Alexa-style simulation</span><p>${esc(checkin.prompt)}</p></div>${response}`;
  if (awaiting && checkin.session_id === sessionId) speakOnce(checkin);
}

function renderCaregiverOutbox(config, items) {
  const outbox = $('#caregiver-outbox');
  if (!config?.enabled) {
    outbox.innerHTML = '<p class="empty">Caregiver preview is off. Enable it before starting observation to test escalation safely.</p>';
    return;
  }
  if (!items.length) {
    outbox.innerHTML = `<p class="empty">Local preview armed for ${esc(config.recipient_label)} after ${config.escalation_delay_seconds} seconds. Nothing is sent externally.</p>`;
    return;
  }
  outbox.innerHTML = items.map(item => `
    <article class="outbox-item">
      <div><strong>Preview for ${esc(item.recipient_label)}</strong><time>${esc(formatTime(item.created_at))}</time></div>
      <p>${esc(item.message)}</p>
      <small>${esc(item.uncertainty_note)}</small>
    </article>`).join('');
}

function speakOnce(checkin) {
  const key = `digital-hippocampus-spoken-${checkin.id}`;
  if (sessionStorage.getItem(key) || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(checkin.prompt);
  utterance.rate = 0.96;
  window.speechSynthesis.speak(utterance);
  sessionStorage.setItem(key, 'true');
}

async function refreshLive() {
  try {
    renderLive(await api('/api/live/status'));
  } catch (error) {
    liveStatus.textContent = error.message;
  }
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  liveStatus.textContent = 'Opening camera and starting asynchronous perception…';
  try {
    const seconds = Number(timer.value);
    const payload = await api('/api/live/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        checkin_after_seconds: seconds,
        departure_confirm_seconds: Math.min(3, seconds),
        snooze_seconds: 60,
        cooldown_seconds: 60,
        caregiver_preview_enabled: caregiverEnabled.checked,
        caregiver_recipient: caregiverRecipient.value,
        caregiver_escalation_delay_seconds: Number(caregiverDelay.value),
        caregiver_notification_limit: Number(caregiverLimit.value),
      }),
    });
    renderLive(payload);
  } catch (error) {
    liveStatus.textContent = error.message;
    startButton.disabled = false;
  }
});

stopButton.addEventListener('click', async () => {
  stopButton.disabled = true;
  try {
    renderLive(await api('/api/live/stop', {method: 'POST'}));
    liveFeed.removeAttribute('src');
  } catch (error) {
    liveStatus.textContent = error.message;
  }
});

async function sendResponse(text) {
  const normalized = text.trim();
  if (!normalized) return;
  sendResponseButton.disabled = true;
  try {
    const result = await api('/api/live/respond', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: normalized}),
    });
    responseText.value = '';
    renderLive(result.status);
    if (!result.handled && result.intent === 'unknown') {
      liveStatus.textContent = 'Try “Give me another minute”, “I’m done”, or “Cancel”.';
    }
  } catch (error) {
    liveStatus.textContent = error.message;
  } finally {
    await refreshLive();
  }
}

sendResponseButton.addEventListener('click', () => sendResponse(responseText.value));
responseText.addEventListener('keydown', event => {
  if (event.key === 'Enter') sendResponse(responseText.value);
});
document.querySelectorAll('[data-response]').forEach(item => {
  item.addEventListener('click', () => sendResponse(item.dataset.response));
});

caregiverEnabled.addEventListener('change', () => {
  caregiverRecipient.disabled = !caregiverEnabled.checked;
  caregiverDelay.disabled = !caregiverEnabled.checked;
  caregiverLimit.disabled = !caregiverEnabled.checked;
});

function configureSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    listenButton.title = 'Speech recognition is unavailable in this browser; type a response instead.';
    return;
  }
  recognition = new Recognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.onstart = () => {
    listeningIndicator.textContent = 'Listening…';
    listeningIndicator.classList.add('active');
  };
  recognition.onresult = event => {
    const transcript = Array.from(event.results)
      .map(item => item[0].transcript)
      .join('');
    responseText.value = transcript;
    if (event.results[event.results.length - 1].isFinal) sendResponse(transcript);
  };
  recognition.onend = () => {
    listeningIndicator.textContent = 'Not listening';
    listeningIndicator.classList.remove('active');
  };
  recognition.onerror = event => {
    liveStatus.textContent = `Speech recognition: ${event.error}. Type a response if needed.`;
  };
  listenButton.addEventListener('click', () => recognition.start());
}

async function loadMemories() {
  const videos = await api('/api/videos');
  recordCount.textContent = `${videos.length} ${videos.length === 1 ? 'record' : 'records'}`;
  if (!videos.length) {
    memories.innerHTML = '<p class="empty">No uploaded-video observations recorded yet.</p>';
    return;
  }
  const details = await Promise.all(
    videos.map(video => api(`/api/videos/${video.id}`))
  );
  memories.innerHTML = details.map(video => {
    const frames = (video.frames || []).slice(0, 12).map(frame => {
      const summary = frame.observations.find(observation => observation.kind === 'summary');
      const imageName = frame.image_path.split('/').pop();
      return `<article class="frame"><img loading="lazy" alt="Video frame at ${frame.timestamp_seconds.toFixed(1)} seconds" src="/frames/${video.id}/${imageName}"><p><b>T+${frame.timestamp_seconds.toFixed(1)} SEC</b><br>${esc(summary?.label || 'No observation')}</p></article>`;
    }).join('');
    return `<article class="video"><div class="record-heading"><div><h3>${esc(video.original_name)}</h3><div class="meta">${(video.duration_seconds || 0).toFixed(1)} seconds / ${video.frames.length} sampled frames / ${esc(video.uploaded_at || '')} UTC</div></div><span class="record-status">${esc(video.status)}</span></div><div class="frames">${frames}</div></article>`;
  }).join('');
}

button.addEventListener('click', async () => {
  const file = input.files[0];
  if (!file) {
    status.textContent = 'Choose a video first.';
    return;
  }
  button.disabled = true;
  status.textContent = 'Uploading and processing video. This may take a moment…';
  try {
    const response = await fetch('/api/videos', {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name),
      },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Upload failed');
    const extraction = payload.event_extraction;
    const eventMessage = extraction?.status === 'complete'
      ? ` ${payload.symbolic_events.length} symbolic events extracted.`
      : extraction?.status === 'failed'
        ? ` Perception succeeded, but event extraction failed: ${extraction.error}`
        : '';
    status.textContent = `Processing complete. ${payload.sampled_frames} frames added.${eventMessage}`;
    input.value = '';
    await loadMemories();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function askArchive() {
  const question = questionInput.value.trim();
  if (!question) return questionInput.focus();
  askButton.disabled = true;
  answer.hidden = false;
  answer.innerHTML = '<div class="answer-copy"><p>Searching the observation archive…</p></div>';
  try {
    const payload = await api('/api/questions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });
    if (!payload.found) {
      answer.innerHTML = `<div class="answer-copy"><p>${esc(payload.answer)}</p></div>`;
      return;
    }
    const rows = payload.detections.map(detection => {
      const box = detection.bbox_xywh;
      const confidence = detection.confidence == null
        ? '—'
        : `${(detection.confidence * 100).toFixed(1)}%`;
      const boxText = box
        ? `${box.x}, ${box.y}, ${box.width}, ${box.height}`
        : 'Unavailable';
      return `<tr><td>${esc(detection.class)}</td><td>${confidence}</td><td>${boxText}</td></tr>`;
    }).join('');
    answer.innerHTML = `<img src="${payload.frame_url}" alt="Frame containing the latest observation"><div class="answer-copy"><p>${esc(payload.answer)}</p><table class="detection-table"><thead><tr><th>Class</th><th>Confidence</th><th>x, y, width, height</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  } catch (error) {
    answer.innerHTML = `<div class="answer-copy"><p>${esc(error.message)}</p></div>`;
  } finally {
    askButton.disabled = false;
  }
}

askButton.addEventListener('click', askArchive);
questionInput.addEventListener('keydown', event => {
  if (event.key === 'Enter') askArchive();
});

configureSpeechRecognition();
refreshLive();
window.setInterval(refreshLive, 1000);
loadMemories().catch(error => { memories.textContent = error.message; });
