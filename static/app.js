/* music-boris frontend */
let currentTrack = null;

async function loadNext() {
  document.getElementById('loading').style.display = 'none';
  const res = await fetch('/api/next');
  if (!res.ok) { showLoading(); return; }
  const track = await res.json();
  if (!track || !track.yt_id) { showLoading(); return; }
  currentTrack = track;
  document.getElementById('track-title').textContent  = track.title;
  document.getElementById('track-artist').textContent = track.artist;
  document.getElementById('track-year').textContent   = track.year || '';
  document.getElementById('track-seed').textContent   = track.seed ? `similar to ${track.seed}` : '';
  document.getElementById('yt-player').src =
    `https://www.youtube.com/embed/${track.yt_id}?autoplay=1&controls=1`;
}

function showLoading() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('track-title').textContent  = '';
  document.getElementById('track-artist').textContent = '';
  document.getElementById('yt-player').src = '';
  setTimeout(loadNext, 3000);
}

async function verdict(v) {
  if (!currentTrack) return;
  await fetch('/api/verdict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ track_id: currentTrack.id, verdict: v }),
  });
  loadNext();
}

document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'l' || e.key === 'L') verdict('LIKED');
  if (e.key === 'j' || e.key === 'J') verdict('REJECTED');
  if (e.key === 'k' || e.key === 'K') verdict('SKIPPED');
});

loadNext();
