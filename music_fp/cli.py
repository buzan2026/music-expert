"""CLI entry point for music-fp."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from . import __version__
from .config import VALID_LABELS, LABEL_LIKED, LABEL_DISLIKED, LABEL_UNTAGGED

app = typer.Typer(
    name="music-fp",
    help="Audio fingerprint tool — ingest tracks and predict if you'll like them.",
    add_completion=False,
)
console = Console()


def _label_style(label: str) -> str:
    return {"liked": "[green]liked[/]", "disliked": "[red]disliked[/]"}.get(label, "[dim]untagged[/]")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="YouTube URL or path to local audio file"),
    label: str = typer.Option(
        LABEL_UNTAGGED,
        "--label", "-l",
        help=f"Tag this track: {', '.join(sorted(VALID_LABELS))}",
    ),
) -> None:
    """Ingest a track: download (if URL), extract fingerprint, store in DB."""
    from .download import fetch, is_youtube_url
    from .fingerprint import extract
    from .store import upsert_track, get_track_by_source

    label = label.lower()
    if label not in VALID_LABELS:
        console.print(f"[red]Invalid label '{label}'. Choose from: {', '.join(sorted(VALID_LABELS))}[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Source:[/] {source}")

    with console.status("Fetching audio…"):
        audio_path, title = fetch(source, progress=False)

    display_title = title or audio_path.stem
    console.print(f"[bold]Title:[/]  {display_title}")
    console.print(f"[bold]File:[/]   {audio_path}")

    with console.status("Extracting fingerprint (this takes ~10–30 s)…"):
        features = extract(audio_path)

    # Estimate duration from librosa metadata
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
    )

    console.print(
        f"[green]✓ Ingested[/] [bold]{display_title}[/] "
        f"(id={track_id}, label={_label_style(label)}, "
        f"duration={int(duration or 0)//60}m{int(duration or 0)%60:02d}s)"
    )


@app.command()
def tag(
    track_id: int = typer.Argument(..., help="Track id (see `list`)"),
    label: str = typer.Argument(..., help=f"New label: {', '.join(sorted(VALID_LABELS))}"),
) -> None:
    """Update the label of an already-ingested track."""
    from .store import update_label

    label = label.lower()
    if label not in VALID_LABELS:
        console.print(f"[red]Invalid label '{label}'.[/]")
        raise typer.Exit(1)

    update_label(track_id, label)
    console.print(f"Track {track_id} → {_label_style(label)}")


@app.command("list")
def list_tracks(
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Filter by label"),
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
    table.add_column("Title", min_width=30)
    table.add_column("Label", width=10)
    table.add_column("Duration", width=8)
    table.add_column("Added", width=20)

    for t in tracks:
        dur = t["duration"]
        dur_str = f"{int(dur)//60}m{int(dur)%60:02d}s" if dur else "—"
        label_str = {"liked": "[green]liked[/]", "disliked": "[red]disliked[/]"}.get(
            t["label"], "[dim]untagged[/]"
        )
        table.add_row(str(t["id"]), t["title"] or t["source"], label_str, dur_str, t["added_at"][:16])

    console.print(table)
    console.print(f"[dim]{len(tracks)} track(s)[/]")


@app.command()
def analyze(
    source: str = typer.Argument(..., help="YouTube URL or local file to analyze"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of nearest neighbors to show"),
) -> None:
    """Analyze a track and predict if you'll like it based on your collection."""
    from .download import fetch
    from .fingerprint import extract, predict_label
    from .store import get_all_tracks, get_track_by_source

    labeled_tracks = get_all_tracks(labeled_only=True)
    if not labeled_tracks:
        console.print("[yellow]No labeled tracks yet. Ingest some with --label liked/disliked first.[/]")
        raise typer.Exit(0)

    liked_count = sum(1 for t in labeled_tracks if t["label"] == LABEL_LIKED)
    disliked_count = sum(1 for t in labeled_tracks if t["label"] == LABEL_DISLIKED)

    if liked_count == 0 or disliked_count == 0:
        console.print(
            "[yellow]Need at least one 'liked' AND one 'disliked' track for a meaningful prediction.[/]\n"
            f"  Current: {liked_count} liked, {disliked_count} disliked"
        )
        # Continue anyway to show similarities

    console.print(f"[bold]Source:[/] {source}")

    with console.status("Fetching audio…"):
        audio_path, title = fetch(source, progress=False)

    display_title = title or audio_path.stem
    console.print(f"[bold]Title:[/]  {display_title}")

    with console.status("Extracting fingerprint…"):
        features = extract(audio_path)

    result = predict_label(features, labeled_tracks, top_k=top_k)

    pred = result["prediction"]
    conf = result["confidence"]
    conf_pct = f"{conf*100:.1f}%"

    pred_style = {"liked": "[bold green]", "disliked": "[bold red]"}.get(pred, "[bold yellow]")
    console.print(f"\n[bold]Prediction:[/] {pred_style}{pred.upper()}[/]  (confidence {conf_pct})")
    console.print(
        f"  avg similarity to liked:    [green]{result['avg_sim_liked']:.4f}[/]\n"
        f"  avg similarity to disliked: [red]{result['avg_sim_disliked']:.4f}[/]"
    )

    if result["top_neighbors"]:
        console.print(f"\n[bold]Top {top_k} nearest tracks:[/]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Sim", width=6)
        table.add_column("Label", width=10)
        table.add_column("Title")
        for sim, track in result["top_neighbors"]:
            label_str = {"liked": "[green]liked[/]", "disliked": "[red]disliked[/]"}.get(
                track["label"], "[dim]untagged[/]"
            )
            table.add_row(f"{sim:.3f}", label_str, track["title"] or track["source"])
        console.print(table)

    console.print(
        f"\n[dim]Comparison base: {liked_count} liked + {disliked_count} disliked tracks[/]"
    )


@app.command()
def stats() -> None:
    """Show collection statistics."""
    from .store import count_by_label, get_all_tracks
    from .config import DB_PATH

    counts = count_by_label()
    total = sum(counts.values())

    console.print(f"\n[bold]music-fp collection[/]  v{__version__}")
    console.print(f"  Database: [dim]{DB_PATH}[/]")
    console.print(f"  Total tracks: {total}")
    console.print(f"    [green]liked[/]:    {counts.get('liked', 0)}")
    console.print(f"    [red]disliked[/]: {counts.get('disliked', 0)}")
    console.print(f"    [dim]untagged[/]: {counts.get('untagged', 0)}")

    labeled = counts.get("liked", 0) + counts.get("disliked", 0)
    if labeled < 30:
        needed = 30 - labeled
        console.print(
            f"\n  [yellow]→ Ingest {needed} more labeled tracks to unlock supervised learning.[/]"
        )
    else:
        console.print("\n  [green]→ Ready for supervised learning (≥30 labeled tracks).[/]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", is_eager=True, help="Show version"),
) -> None:
    if version:
        console.print(f"music-fp {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
