const input = document.querySelector('#video');
const button = document.querySelector('#upload');
const status = document.querySelector('#status');
const memories = document.querySelector('#memories');
const recordCount = document.querySelector('#record-count');
const questionInput = document.querySelector('#question');
const askButton = document.querySelector('#ask');
const answer = document.querySelector('#answer');

function esc(value) {
  const node = document.createElement('div');
  node.textContent = value ?? '';
  return node.innerHTML;
}

async function loadMemories() {
  const videos = await fetch('/api/videos').then(response => response.json());
  recordCount.textContent = `${videos.length} ${videos.length === 1 ? 'record' : 'records'}`;
  if (!videos.length) {
    memories.innerHTML = '<p class="empty">No observations recorded yet.</p>';
    return;
  }

  const details = await Promise.all(
    videos.map(video => fetch(`/api/videos/${video.id}`).then(response => response.json()))
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
  if (!question) {
    questionInput.focus();
    return;
  }

  askButton.disabled = true;
  answer.hidden = false;
  answer.innerHTML = '<div class="answer-copy"><p>Searching the observation archive…</p></div>';
  try {
    const response = await fetch('/api/questions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Question failed');
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
loadMemories().catch(error => {
  memories.textContent = error.message;
});
