"""SQLite schema and access helpers."""
# Tables:
#   verdicts(id, artist, title, yt_id, verdict, ts)
#     verdict IN ('LIKED', 'REJECTED', 'SKIPPED')
#   candidate_queue(id, artist, title, yt_id, year, playcount, listeners,
#                   tags_json, seed_artist, similarity_score, score, added_at)
