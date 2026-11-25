#!/usr/bin/env python3
"""Find models on Artificial Analysis not yet in llm.json.

"New" is decided by AA release_date: by default, models released within the
last 30 days (--days). Slugs already in llm.json, slugs an llm.json model
already reads on AA, slugs listed in the ignored-mapping file, and slugs
previously dismissed here are filtered out.

The candidates are printed, then (when run interactively) each is offered for
addition:
  [y] add        -> hands off to add.py, which adds the model to llm.json and
                    walks the per-source mapping prompts (scores are filled by
                    a subsequent update.py run).
  [n] never      -> record the slug in the dismiss file so it is not offered
                    again.
  [q] quit       -> stop asking.

Closed-weights releases are found too, but llm.json carries a closed model only
as a reference model (see _reference.py), so they are offered for that instead:
  [r] reference  -> put the slug on reference-models.json and add its entry, the
                    same thing the admin page's Reference tab does.
  [n] never      -> dismissed, exactly as for an open model.
  [q] quit       -> stop asking.
They are queued under their own kind, new-closed-model, so nothing that adds a
queued open model (propose.py, the admin page's "Add all") can ever add one of
these as an open-weight row. --open-only leaves them out altogether.

AA calls a release closed until its weights are on Hugging Face, even when the
lab has announced them. So each release also gets a weight status (see
_openness.py): a person's override in weights-overrides.json first, else the
most open of AA's verdict and the pooled index of the other boards. A closed
release that reads "upcoming" (or "open") that way is offered the other way
round -- [o] add as open is the suggestion -- and the queue carries the status,
so the admin page suggests the open add too.

With no tty (e.g. inside a batch script), it offers nothing and only prints the
candidates. Under --collect-prompts each candidate is queued for review instead
of being offered, and nothing is dismissed.

Both answers also exist unattended: propose.py turns a queued candidate into an
llm.json entry plus a line in check_new-decisions.json, and the reviewer flips
that line to __ignored__ to take the other branch. Every run except --no-add
starts by carrying out whatever those lines already say -- before any score is
fetched, so an ignored model is gone before anything can attach a mapping or a
score to it. See _new_models.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import _openness
import _prompts
from _artificialanalysis_mapping import mapped_aa_slugs
from _new_models import apply_decisions, dismiss, load_decisions, load_dismissed

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
AA_SCRIPT = HERE / "artificialanalysis.py"
ADD_SCRIPT = HERE / "add.py"
IGNORED_MAPPING = HERE / "model-name-mapping-llm-to-artificialanalysis-ignored.json"

# The queued kind of a closed release: its answers are "carry it as a reference
# model" or "ignore it", never the open model's "add it".
NEW_MODEL_KIND = "new-model"
NEW_CLOSED_MODEL_KIND = "new-closed-model"


def existing_slugs(llm_path: Path) -> set[str]:
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    return {m.get("name") for m in doc.get("models", []) if m.get("name")}


def ignored_slugs() -> set[str]:
    if not IGNORED_MAPPING.exists():
        return set()
    mapping = json.loads(IGNORED_MAPPING.read_text(encoding="utf-8"))
    slugs: set[str] = set()
    for variants in mapping.values():
        slugs.update(variants)
    return slugs


def fetch_aa_models(cutoff: date, weights: str) -> list[dict]:
    """AA's models released on/after cutoff; weights is "open", "closed", or
    "tag" for both in one run, each carrying "_open": true or false."""
    cmd = [
        sys.executable,
        str(AA_SCRIPT),
        "--release-date",
        cutoff.isoformat(),
        "--output",
        "json",
        "--no-mmmu-pro",
        "--tag-weights" if weights == "tag" else f"--{weights}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"artificialanalysis.py failed (exit {proc.returncode})")
    return json.loads(proc.stdout).get("data", [])


def split_tagged(tagged: list[dict]) -> list[dict]:
    """A --tag-weights run as the two runs it replaces would have returned it:
    the open models, then the closed ones marked "_closed", each in AA's order."""
    verdicts = [m.pop("_open", None) for m in tagged]
    models = [m for m, is_open in zip(tagged, verdicts) if is_open is True]
    models += [
        {**m, "_closed": True} for m, is_open in zip(tagged, verdicts) if is_open is False
    ]
    return models


def weights_of(m: dict) -> str:
    """A release's weight status: an override, else AA and the other boards.

    AA's own verdict is the floor -- the most open verdict wins, so another
    board that already lists the weights (or an announcement read as
    "upcoming") lifts a release AA still calls closed.
    """
    aa = _openness.CLOSED if m.get("_closed") else _openness.OPEN
    names = [n for n in (m.get("slug"), m.get("name")) if isinstance(n, str) and n]
    for name in names:
        override = _openness.override_for(name)
        if override:
            return override
    if aa == _openness.OPEN:
        return aa
    try:
        index = _openness.status_index()
    except Exception:  # noqa: BLE001 - the index is a hint; AA's verdict stands
        index = {}
    pooled = [index.get(_openness.normalize(name)) for name in names]
    return _openness.merge(aa, *pooled) or aa


def creator_name(m: dict) -> str:
    c = m.get("model_creator")
    if isinstance(c, dict):
        return c.get("name") or c.get("slug") or "?"
    return "?"


def add_reference(slug: str, json_file: str) -> bool:
    """Carry a closed release as a reference model: the list and the entry.

    The same writer the admin page's reference-add record goes through, so the
    entry gets its AA model page as `url` and its `reference` flag at once.
    Imported here because it pulls in every mapping module.
    """
    import _answers

    answer = _answers.Answer(index=0, kind=_answers.REFERENCE_ADD, subject=slug)
    try:
        for line in _answers._add_reference(answer, Path(json_file)):
            print(f"  {line}")
    except Exception as exc:  # noqa: BLE001 - one model's failure must not end the loop
        print(f"  {exc}", file=sys.stderr)
        return False
    return True


def add_model(slug: str, json_file: str) -> bool:
    """Run add.py for a slug, inheriting the terminal for its prompts."""
    proc = subprocess.run([sys.executable, str(ADD_SCRIPT), "--name", slug, json_file])
    return proc.returncode == 0


def prompt_choice(model: dict) -> str:
    """Return 'add', 'open', 'skip', 'quit' or 'defer' for one candidate.

    For a closed release 'add' means carrying it as a reference model, and
    'open' adds it as an open-weight model anyway -- AA labels a release closed
    until its weights are published, even when they are announced.
    """
    slug = model["slug"]
    closed = model.get("_closed", False)
    weights = model.get("_weights") or (_openness.CLOSED if closed else _openness.OPEN)
    if _prompts.collecting():
        note = " - ".join(
            part
            for part in (
                f"released {model.get('release_date') or '?'}",
                creator_name(model),
                (f"weights {weights} per other sources, AA says closed"
                 if closed and weights != _openness.CLOSED else ""),
                model.get("url") or "",
            )
            if part
        )
        _prompts.record(
            kind=NEW_CLOSED_MODEL_KIND if closed else NEW_MODEL_KIND,
            subject=slug,
            question=(
                f"Carry newly released closed model '{slug}' as a reference model?"
                if closed
                else f"Add newly released model '{slug}' to llm.json?"
            ),
            note=note,
            command="./check_new.py",
            source_url=model.get("url") or None,
            weights=weights,
        )
        return "defer"

    if closed and weights != _openness.CLOSED:
        ask = (
            f"AA calls '{slug}' closed, other sources say {weights}: "
            "[o] add as open (weights coming)  [r] reference  [n] never ask again  [q] quit: "
        )
    elif closed:
        ask = (
            f"Closed '{slug}': carry as reference?  [r] reference  "
            "[o] add as open (weights coming)  [n] never ask again  [q] quit: "
        )
        yes = ("r", "ref", "reference")
    else:
        ask = f"Add '{slug}'?  [y] add  [n] never ask again  [q] quit: "
        yes = ("y", "yes")
    while True:
        try:
            ans = input(ask).strip().lower()
        except EOFError:
            return "quit"
        if ans in yes:
            return "add"
        if closed and ans in ("o", "open"):
            return "open"
        if ans in ("n", "no"):
            return "skip"
        if ans in ("q", "quit"):
            return "quit"
        print(f"  answer {yes[0]}, n, or q")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to llm.json to compare against (default: "./llm.json").',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look back this many days from today for release_date (default: 30).",
    )
    parser.add_argument(
        "--date",
        help="Explicit cutoff YYYY-MM-DD (release_date on/after); overrides --days.",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Leave closed-weights releases out (by default they are offered as "
        "reference models).",
    )
    parser.add_argument(
        "--no-add",
        action="store_true",
        help="Only print candidates: never prompt to add, and never apply the "
        "decisions recorded in check_new-decisions.json (report only).",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)

    if args.date:
        try:
            cutoff = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("invalid --date, expected YYYY-MM-DD")
    else:
        cutoff = date.today() - timedelta(days=args.days)

    # A decision recorded in a merged proposal PR is carried out here, before
    # any score is fetched: dismiss() is frozen under collect mode, this is not.
    # --no-add promises to change nothing, so it skips this too.
    if not args.no_add:
        for line in apply_decisions(Path(args.json_file)):
            print(line)

    known = (
        existing_slugs(Path(args.json_file))
        | mapped_aa_slugs()
        | ignored_slugs()
        # A slug still awaiting its decision in the open proposal PR is not a
        # candidate: it is already in front of a person.
        | set(load_decisions())
        | load_dismissed()
    )
    if args.open_only:
        models = fetch_aa_models(cutoff, "open")
    else:
        # One run for both, tagged, rather than an --open and a --closed run:
        # each of those read every recent model's page to decide which it was.
        # Split back into the same two lists, open first, in AA's order.
        models = split_tagged(fetch_aa_models(cutoff, "tag"))

    new = [m for m in models if m.get("slug") and m["slug"] not in known]
    for m in new:
        m["_weights"] = weights_of(m)
    # Open releases first: they are what llm.json is for.
    new.sort(key=lambda m: m.get("release_date") or "", reverse=True)
    new.sort(key=lambda m: m.get("_closed", False))

    kind = "open-weights models" if args.open_only else "models"
    if not new:
        print(f"No new {kind} on AA since {cutoff.isoformat()}.")
        return 0

    print(f"{len(new)} new {kind} on AA since {cutoff.isoformat()} (not in llm.json):\n")
    print(f"  {'RELEASED':10s}  {'WEIGHTS':8s}  {'SLUG':32s}  {'CREATOR':22s}  URL")
    for m in new:
        print(
            f"  {m.get('release_date') or '?':10s}  "
            f"{m['_weights']:8s}  {m['slug']:32s}  "
            f"{creator_name(m):22s}  {m.get('url') or ''}"
        )

    if args.no_add:
        return 0
    if not _prompts.collecting() and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return 0

    print()
    added = referenced = dismissed = deferred = 0
    for m in new:
        slug = m["slug"]
        choice = prompt_choice(m)
        if choice == "quit":
            break
        if choice == "defer":
            deferred += 1
            continue
        if choice == "skip":
            dismiss(slug)
            dismissed += 1
            continue
        if m.get("_closed") and choice != "open":
            if add_reference(slug, args.json_file):
                referenced += 1
            else:
                print(f"  carrying '{slug}' failed; leaving it for next time.", file=sys.stderr)
            continue
        if add_model(slug, args.json_file):
            added += 1
        else:
            print(f"  add.py failed for '{slug}'; leaving it for next time.", file=sys.stderr)

    if deferred:
        print(f"\nQueued {deferred} for manual review; nothing dismissed.")
        return 0
    print(f"\nAdded {added}, carried {referenced} as reference, dismissed {dismissed}.")
    if added or referenced:
        print("Run ./update.py -w (or ./update-all) to fill in scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
