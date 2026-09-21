"""CLI entry point for music-fp."""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from . import __version__
from .config import VALID_LABELS, LABEL_LIKED, LABEL_DISLIKED, LABEL_UNTAGGED

app = typer.Typer(
    name="music-fp",
    help="Audio fingerprint tool — ingest tracks and predict if you'll like them.",
    add_completion=False,
)
console = Console()

EXTENDED_LABELS = {"top10", "valide", "hors_sujet", *VALID_LABELS}

_YOUTUBE_SEARCH_RE = re.compile(r"youtube\.com/results\?search_query=(.+)")


def _resolve_source(source: str) -> str:
    """Convert a YouTube search URL to a yt-dlp search string."""
    m = _YOUTUBE_SEARCH_RE.search(source)
    if m:
        from urllib.parse import unquote_plus
        query = unquote_plus(m.group(1))
        return f"ytsearch1:{query}"
    return source


def _label_style(label: str) -> str:
    return {
        "top10": "[bold green]top10[/]",
        "valide": "[green]valide[/]",
        "hors_sujet": "[red]hors_sujet[/]",
        "liked": "[green]liked[/]",
        "disliked": "[red]disliked[/]",
    }.get(label, "[dim]untagged[/]")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="YouTube URL / search URL / local file"),
    label: str = typer.Option(LABEL_UNTAGGED, "--label", "-l"),
    artist: Optional[str] = typer.Option(None, "--artist", "-a"),
    year: Optional[int] = typer.Option(None, "--year", "-y"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n"),
    no_stems: bool = typer.Option(False, "--no-stems", help="Skip Demucs stem separation (faster)"),
) -> None:
    """Download/load a track, extract fingerprint (+ stems), store in DB."""
    from .download import fetch
    from .fingerprint import extract, FEATURE_VERSION
    from .store import upsert_track

    label = label.lower()
    if label not in EXTENDED_LABELS:
        console.print(f"[red]Invalid label. Choose from: {', '.join(sorted(EXTENDED_LABELS))}[/]")
        raise typer.Exit(1)

    resolved = _resolve_source(source)
    console.print(f"[bold]Source:[/] {source}")
    if resolved != source:
        console.print(f"[dim]→ yt-dlp search: {resolved}[/]")

    with console.status("Fetching audio…"):
        audio_path, title = fetch(resolved, progress=False)

    display_title = title or audio_path.stem
    console.print(f"[bold]Title:[/]  {display_title}")

    use_stems = not no_stems
    status_msg = "Extracting fingerprint + stems (Demucs, ~2–5 min)…" if use_stems else "Extracting mix fingerprint…"
    with console.status(status_msg):
        features = extract(audio_path, use_stems=use_stems)

    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
    except Exception:
        duration = None

    track_id = upsert_track(
        source=source,
        title=display_title,
        duration=duration,
        label=label,
        features=features,
        artist=artist,
        year=year,
        notes=notes,
        feature_version=FEATURE_VERSION,
    )

    dur_str = f"{int(duration or 0)//60}m{int(duration or 0)%60:02d}s" if duration else "?"
    console.print(
        f"[green]✓ Ingested[/] [bold]{display_title}[/] "
        f"(id={track_id}, label={_label_style(label)}, {dur_str}, {len(features)}d features)"
    )


@app.command("sanity-check")
def sanity_check(
    source: str = typer.Argument(..., help="YouTube URL or local file"),
) -> None:
    """Run Demucs + full feature extraction and print diagnostic values."""
    from .download import fetch
    from .fingerprint import sanity_report

    resolved = _resolve_source(source)
    console.print(f"[bold]Source:[/] {source}")

    with console.status("Downloading audio…"):
        audio_path, title = fetch(resolved, progress=False)
    display_title = title or audio_path.stem
    console.print(f"[bold]File:[/] {audio_path}")

    console.print("[dim]Running Demucs stem separation (first run downloads model ~200 MB)…[/]")
    with console.status("Demucs + feature extraction (2–5 min)…"):
        report = sanity_report(audio_path)

    def _v(val, unit="") -> str:
        if val is None:
            return "[dim]N/A (Demucs requis)[/]"
        return f"[cyan]{val}{unit}[/]"

    stem_section = ""
    if report["stems_available"]:
        stem_section = f"""
[bold]Range vocal (F0)[/]
  min                         : {_v(report['f0_min_hz'], ' Hz')}
  max                         : {_v(report['f0_max_hz'], ' Hz')}
  médian                      : {_v(report['f0_median_hz'], ' Hz')}
  std                         : {_v(report['f0_std_hz'], ' Hz')}
[bold]Dynamique vocale (std RMS)[/]  : {_v(report['vocal_dynamics_rms_std'])}

[bold]Bass harmonic ratio[/]         : {_v(report['bass_harmonic_ratio'])}  (1.0=pur ton, 0=bruit)
[bold]Spectral centroid bass[/]      : {_v(report['spectral_centroid_bass_hz'], ' Hz')}
[bold]Spectral centroid drums[/]     : {_v(report['spectral_centroid_drums_hz'], ' Hz')}

[bold]RMS relatif des stems[/]
  vocals                      : {_v(report['rms_rel_vocals'])}
  bass                        : {_v(report['rms_rel_bass'])}
  drums                       : {_v(report['rms_rel_drums'])}
  other                       : {_v(report['rms_rel_other'])}"""
    else:
        stem_section = "\n[yellow]⚠  Demucs indisponible (modèle non téléchargé) — features de stems absentes.[/]\n[dim]  Relance sur machine avec accès internet pour obtenir les valeurs stem.[/]"

    console.print(Panel(
        f"""[bold]BPM[/]                         : {_v(report['bpm'])}
[bold]Loudness intégré[/]           : {_v(report['lufs'], ' LUFS')}
[bold]Spectral centroid mix[/]      : {_v(report['spectral_centroid_mix_hz'], ' Hz')}
[bold]RMS mix[/]                    : {_v(report['rms_mix'])}{stem_section}""",
        title=f"[bold]{display_title}[/]",
        subtitle="music-fp sanity check",
    ))


@app.command()
def tag(
    track_id: int = typer.Argument(...),
    label: str = typer.Argument(...),
) -> None:
    """Update the label of an already-ingested track."""
    from .store import update_label

    label = label.lower()
    if label not in EXTENDED_LABELS:
        console.print(f"[red]Invalid label.[/]")
        raise typer.Exit(1)
    update_label(track_id, label)
    console.print(f"Track {track_id} → {_label_style(label)}")


@app.command("list")
def list_tracks(
    label: Optional[str] = typer.Option(None, "--label", "-l"),
) -> None:
    """List all ingested tracks."""
    from .store import get_all_tracks

    tracks = get_all_tracks()
    if label:
        tracks = [t for t in tracks if t["label"] == label.lower()]

    if not tracks:
        console.print("[dim]No tracks found.[/]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Artist / Title", min_width=35)
    table.add_column("Label", width=12)
    table.add_column("Dur", width=7)
    table.add_column("Added", width=11)

    for t in tracks:
        dur = t["duration"]
        dur_str = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "—"
        name = f"{t['artist']} — {t['title']}" if t.get("artist") else (t["title"] or t["source"])
        table.add_row(str(t["id"]), name, _label_style(t["label"]), dur_str, (t["added_at"] or "")[:10])

    console.print(table)
    console.print(f"[dim]{len(tracks)} track(s)[/]")


@app.command()
def match(
    source: str = typer.Argument(..., help="YouTube URL or local file to match"),
    no_stems: bool = typer.Option(False, "--no-stems"),
    store: bool = typer.Option(False, "--store", help="Store the candidate in DB after matching"),
    label: str = typer.Option("untagged", "--label", "-l"),
) -> None:
    """Compute distances from a candidate to all top10 etalons and report."""
    from .download import fetch
    from .fingerprint import extract, FEATURE_VERSION
    from .store import get_all_tracks, upsert_track
    from .match import match_report

    top10_tracks = get_all_tracks(label_filter=["top10"])
    if not top10_tracks:
        console.print("[yellow]No top10 etalons in database. Ingest some with --label top10 first.[/]")
        raise typer.Exit(0)

    resolved = _resolve_source(source)
    console.print(f"[bold]Source:[/] {source}")
    if resolved != source:
        console.print(f"[dim]→ search: {resolved}[/]")

    with console.status("Fetching audio…"):
        audio_path, title = fetch(resolved, progress=False)
    display_title = title or audio_path.stem
    console.print(f"[bold]Title:[/]  {display_title}")

    use_stems = not no_stems
    with console.status("Extracting fingerprint…"):
        features = extract(audio_path, use_stems=use_stems)

    result = match_report(features, display_title, top10_tracks)

    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        raise typer.Exit(1)

    # Build output
    lines = [f"[bold]Candidat :[/] {display_title}\n{'─'*44}"]
    lines.append("[bold]Distance aux étalons top10 :[/]")
    for e in result["per_etalon"]:
        name = f"{e['artist']} — {e['title']}" if e["artist"] else e["title"]
        lines.append(f"  • {name:<38} euclid [cyan]{e['euclidean']:.3f}[/]  cosine-dist [cyan]{e['cosine_dist']:.3f}[/]")

    near = result["nearest"]
    near_name = f"{near['artist']} — {near['title']}" if near["artist"] else near["title"]
    lines.append(f"\n[bold green]Étalon le plus proche :[/] {near_name}  (euclid {near['euclidean']:.3f})")

    lines.append("\n[bold]Features les plus divergentes (vs moyenne top10) :[/]")
    for f in result["top3_diverging_features"]:
        lines.append(
            f"  • [yellow]{f['feature']:<40}[/]  "
            f"candidat=[cyan]{f['candidate_raw']:.4f}[/]  "
            f"top10_moy=[dim]{f['top10_mean_raw']:.4f}[/]  "
            f"écart=[bold]{f['deviation']:.3f}σ[/]"
        )

    console.print(Panel("\n".join(lines), title="music-fp match report"))

    if store:
        try:
            import librosa
            duration = librosa.get_duration(path=str(audio_path))
        except Exception:
            duration = None
        track_id = upsert_track(
            source=source, title=display_title, duration=duration,
            label=label, features=features, feature_version=FEATURE_VERSION,
        )
        console.print(f"[dim]Stored as id={track_id}[/]")


@app.command()
def analyze(
    source: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top", "-k"),
    no_stems: bool = typer.Option(False, "--no-stems"),
) -> None:
    """Analyze a track and predict if you'll like it (vs liked/disliked/top10/valide/hors_sujet)."""
    from .download import fetch
    from .fingerprint import extract, predict_label
    from .store import get_all_tracks

    labeled_tracks = get_all_tracks(labeled_only=True)
    if not labeled_tracks:
        console.print("[yellow]No labeled tracks yet.[/]")
        raise typer.Exit(0)

    resolved = _resolve_source(source)
    with console.status("Fetching audio…"):
        audio_path, title = fetch(resolved, progress=False)
    display_title = title or audio_path.stem

    with console.status("Extracting fingerprint…"):
        features = extract(audio_path, use_stems=not no_stems)

    result = predict_label(features, labeled_tracks, top_k=top_k)

    pred = result["prediction"]
    conf = result["confidence"]
    pred_style = {"liked": "[bold green]", "top10": "[bold green]", "valide": "[green]",
                  "disliked": "[bold red]", "hors_sujet": "[bold red]"}.get(pred, "[bold yellow]")

    console.print(f"\n[bold]{display_title}[/]")
    console.print(f"Prediction: {pred_style}{pred.upper()}[/]  (confidence {conf*100:.1f}%)")
    console.print(f"  avg sim to liked/top10/valide : [green]{result['avg_sim_liked']:.4f}[/]")
    console.print(f"  avg sim to disliked/hors_sujet: [red]{result['avg_sim_disliked']:.4f}[/]")

    if result["top_neighbors"]:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Sim", width=6)
        table.add_column("Label", width=12)
        table.add_column("Title")
        for sim, track in result["top_neighbors"]:
            name = f"{track.get('artist', '')} — {track['title']}" if track.get("artist") else track["title"] or track["source"]
            table.add_row(f"{sim:.3f}", _label_style(track["label"]), name)
        console.print(table)


@app.command()
def stats() -> None:
    """Show collection statistics."""
    from .store import count_by_label
    from .config import DB_PATH

    counts = count_by_label()
    total = sum(counts.values())

    console.print(f"\n[bold]music-fp collection[/]  v{__version__}")
    console.print(f"  Database: [dim]{DB_PATH}[/]")
    console.print(f"  Total : {total}")
    for lbl in ["top10", "valide", "hors_sujet", "liked", "disliked", "untagged"]:
        n = counts.get(lbl, 0)
        if n:
            console.print(f"    {_label_style(lbl)}: {n}")

    labeled = sum(v for k, v in counts.items() if k != "untagged")
    needed = max(0, 50 - labeled)
    if needed:
        console.print(f"\n  [yellow]→ Ingest {needed} more labeled tracks to unlock supervised learning (50+).[/]")
    else:
        console.print("\n  [green]→ Ready for supervised learning (≥50 labeled tracks).[/]")


@app.command()
def seed(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show artists without running searches"),
) -> None:
    """Seed the queue from the built-in taste-profile artist list (no audio download)."""
    from .discover import SEED_ARTISTS, run_discovery_cycle, _liked_artists

    if dry_run:
        console.print("[bold]Seed artists:[/]")
        for a in SEED_ARTISTS:
            console.print(f"  • {a}")
        return

    liked = _liked_artists()
    if liked:
        console.print(f"[dim]{len(liked)} liked artist(s) already in DB — running standard discovery.[/]")
    else:
        console.print(f"[dim]No liked artists yet — seeding from {len(SEED_ARTISTS)} taste-profile artists.[/]")

    with console.status("Searching YouTube metadata (audio loads on demand)…"):
        added = run_discovery_cycle(max_new=30, metadata_only=True)

    console.print(f"[green]✓ {added} candidate(s) added to queue.[/]")


@app.command()
def serve(
    port: int = typer.Option(5000, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
    no_discover: bool = typer.Option(False, "--no-discover", help="Disable background discovery worker"),
) -> None:
    """Start the listening interface (web player)."""
    from .server import create_app

    console.print(f"[bold]music-fp player[/]  →  http://{host}:{port}")
    flask_app = create_app(start_worker=not no_discover)
    flask_app.run(host=host, port=port, debug=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", is_eager=True),
) -> None:
    if version:
        console.print(f"music-fp {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
