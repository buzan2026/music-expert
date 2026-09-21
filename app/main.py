"""FastAPI app — routes + startup + static files."""
# GET  /           → static/index.html
# GET  /api/next   → next candidate from pool
# POST /api/verdict → {track_id, verdict: LIKED|REJECTED|SKIPPED}
# GET  /api/status  → pool size, verdict counts
#
# Startup: init DB, load Notion exclusion set, seed bootstrap verdicts if missing,
#          launch background pool-fill task.
