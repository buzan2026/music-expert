/* music-boris frontend */
let currentTrack = null;
let isLoading = false;

async function loadNext() {
  if (isLoading) return;
  isLoading = true;
  hideLoading();

  try {
    const res = await fetch('/api/next');
    if (!res.ok) throw new Error(res.status);
    const track = await res.json();

    if (!track.ready || !track.yt_id) {
      showLoading();
      return;
    }

    currentTrack = track;
    document.getElementById('track-title').textContent  = track.title;
    document.getElementById('track-artist').textContent = track.artist;
    document.getElementById('track-year').textContent   = track.year ? `${track.year}` : '';
    document.getElementById('track-seed').textContent   = track.seed ? `similar to ${track.seed}` : '';

    // Audio-only player: autoplay=1, start=0
    document.getElementById('yt-player').src =
      `https://www.youtube.com/embed/${track.yt_id}?autoplay=1&controls=1&rel=0`;

    document.getElementById('buttons').style.display = 'flex';
  } catch (e) {
    showLoading();
  } finally {
    isLoading = false;
  }
}

function showLoading() {
  document.getElementById('loading').style.display  = 'block';
  document.getElementById('buttons').style.display  = 'none';
  document.getElementById('track-title').textContent  = '';
  document.getElementById('track-artist').textContent = '';
  document.getElementById('track-year').textContent   = '';
  document.getElementById('track-seed').textContent   = '';
  document.getElementById('yt-player').src = '';
  setTimeout(loadNext, 4000);
}

function hideLoading() {
  document.getElementById('loading').style.display = 'none';
}

async function verdict(v) {
  if (!currentTrack) return;
  const track = currentTrack;
  currentTrack = null;  // prevent double-submit

  await fetch('/api/verdict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      verdict:          v,
      artist:           track.artist,
      title:            track.title,
      yt_id:            track.yt_id  || null,
      year:             track.year   || null,
      playcount:        track.playcount || 0,
      tags_json:        track.tags_json || '[]',
      similarity_score: track.similarity_score || 0,
    }),
  });

  loadNext();
}

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'l' || e.key === 'L') verdict('LIKED');
  if (e.key === 'j' || e.key === 'J') verdict('REJECTED');
  if (e.key === 'k' || e.key === 'K') verdict('SKIPPED');
});

loadNext();
