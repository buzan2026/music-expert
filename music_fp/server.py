"""Flask web server — listening interface."""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort

from .discover import DiscoveryWorker
from .store import (
    get_track_by_id,
    pending_candidate_count,
    pop_next_candidate,
    record_vote,
    get_vote_counts,
)
from .model import get_model

app = Flask(__name__, template_folder="templates")
_worker = DiscoveryWorker()


def _track_json(track: dict) -> dict:
    return {
        "track_id": track["id"],
        "title": track.get("title") or "Titre inconnu",
        "artist": track.get("artist") or "",
        "thumbnail_url": track.get("thumbnail_url") or None,
        "duration": track.get("duration"),
    }


def _stats_json() -> dict:
    counts = get_vote_counts()
    total = sum(counts.values())
    return {
        "total_votes": total,
        "liked": counts.get("liked", 0),
        "disliked": counts.get("disliked", 0),
        "skipped": counts.get("skipped", 0),
        "queue_size": pending_candidate_count(),
        "model_version": get_model().version,
        "model_trained": get_model().is_trained,
        "discovering": _worker.is_running,
    }


@app.route("/")
def index():
    return render_template("player.html")


@app.route("/api/current")
def api_current():
    track = pop_next_candidate()
    if not track:
        return jsonify({}), 200
    return jsonify({**_track_json(track), "stats": _stats_json()})


@app.route("/api/vote", methods=["POST"])
def api_vote():
    data = request.get_json(force=True, silent=True) or {}
    track_id = data.get("track_id")
    vote = data.get("vote")
    if not track_id or vote not in ("liked", "disliked", "skipped"):
        return jsonify({"error": "invalid"}), 400

    record_vote(int(track_id), vote)
    get_model().maybe_retrain()

    # Trigger refill if queue is low
    if pending_candidate_count() < 5 and not _worker.is_running:
        _worker.start()

    next_track = pop_next_candidate()
    resp = {"stats": _stats_json()}
    if next_track:
        resp["next"] = _track_json(next_track)
    return jsonify(resp)


@app.route("/audio/<int:track_id>")
def audio(track_id: int):
    track = get_track_by_id(track_id)
    if not track:
        abort(404)
    audio_path = track.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        abort(404)
    return send_file(audio_path, mimetype="audio/mpeg", conditional=True)


@app.route("/api/stats")
def api_stats():
    return jsonify(_stats_json())


@app.route("/api/discover", methods=["POST"])
def api_discover():
    if not _worker.is_running:
        _worker.start()
    return jsonify({"discovering": True})


def create_app(start_worker: bool = True) -> Flask:
    if start_worker:
        _worker.start()
    return app
