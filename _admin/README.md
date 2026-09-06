# The admin page

Answer the pipeline's queued questions from a browser instead of a pull request.

Every unrecognised source name and every newly released model used to cost a
round trip: the daily run queues the question, `propose.py` opens a proposal PR,
and you open GitHub, read a diff, click a suggestion and merge — for what is
usually one word. This page is the short way round. It renders the same queue,
you tap an answer, and a workflow run applies it and pushes to `main`.

## Why it is not on the public site

GitHub Pages runs Jekyll here (there is no `.nojekyll`), and Jekyll does not copy
`_`-prefixed paths into the built site. That is already why `_matching.py` and
its siblings are not served. So `_admin/` and `_pending/` are *structurally*
unpublishable rather than merely unlinked:

```
$ curl -o /dev/null -w '%{http_code}\n' https://openbench.david-grieser.de/_matching.py
404
```

If a `.nojekyll` file is ever added to this repository, both directories become
publicly readable. Nothing secret lives in them — the token is not here — but the
page would be reachable by anyone. `test_answers.py` asserts the file does not
exist.

## How it fits together

```
browser  ──reads──►  llm.json (the published site)
                     _pending/pending.json (raw.githubusercontent)
             both send Access-Control-Allow-Origin: *, so reads need no token

browser  ──writes─►  api.php  ──►  workflow_dispatch: update-benchmarks.yml
                                        │
                                   ./answer.py -w   validates every record
                                   git push         to main, with the deploy key
                                   ./update-all     scores the new mappings
```

The page holds no credential. `api.php` holds one, and it can do exactly two
things: queue this one workflow, and list its runs.

## Deploying

The page is one file with no build step, so deploying is a copy:

```sh
rsync -av --delete \
  --exclude config.example.php --exclude README.md \
  _admin/ you@your-host:/path/to/docroot/admin/
```

Then, **outside the docroot**, create the config:

```sh
cp _admin/config.example.php /path/outside/docroot/ai-bench-admin-config.php
$EDITOR /path/outside/docroot/ai-bench-admin-config.php   # paste the token
```

`api.php` looks for it at `../../ai-bench-admin-config.php` relative to itself,
or at `$AI_BENCH_ADMIN_CONFIG`. It refuses to start without one rather than
falling back to anything.

Finally, point `.htaccess` at an `.htpasswd`:

```sh
htpasswd -B -c /path/outside/docroot/.htpasswd you
$EDITOR /path/to/docroot/admin/.htaccess   # set AuthUserFile
```

Authentication is the web server's job. Nothing in the PHP tries to do it again,
so if the `.htaccess` is wrong the endpoint is open — check it before you rely on
it:

```sh
curl -si https://your-host/admin/api.php | head -1    # expect 401
```

## The token

A **fine-grained** personal access token:

| | |
| --- | --- |
| Repository access | only `dgrieser/ai-bench` |
| Permissions | **Actions → Read and write**, nothing else |

That scope is the point, not an inconvenience. A `Contents: write` token would
bypass the branch ruleset outright — the repository owner is on its bypass list —
which would turn an internet-facing endpoint into arbitrary-write-to-`main`. With
Actions alone, the worst anyone who gets past the host's auth can do is queue a
workflow whose every record `answer.py` then validates on the runner.

Fine-grained tokens expire within a year, and the failure is opaque. `api.php`
passes GitHub's own `message` through, so when dispatches start failing the page
tells you why.

## Using it

- **Queue** — one card per unanswered source name. Candidates come from
  `_matching.grade()`, best first, each labelled with why it matched; an exact
  match is outlined in green. `__unmappable__` declines the name. The search box
  falls back to the full list.
- **New models** — *Add it* runs `add.py` and records the answer; *Ignore it*
  writes `__ignored__`, which removes the entry and stops the slug being offered.
  (These are genuinely different records, not two values of one field — see the
  comment in `_answers.py`, and `_new_models.apply_decisions`.)
- **Models** — edit `params`, `context` and any non-derived score on an existing
  entry. The fields offered are exactly the ones `edit.py` has a flag for.
- **Runs** — the last few runs of the workflow.

Answers batch up and go in one dispatch, so a sitting is one commit.

Two things worth knowing:

**Reload after a run to confirm.** The run's summary is the record of what
landed, not its colour: `update-all` is `continue-on-error`, so one dead scraper
reds a run whose answers applied perfectly well. Reload and check the question
has left the queue.

**A queued run blocks a second batch.** The workflow's concurrency group holds
one run plus one queued, and a third arrival cancels the queued one silently.
`api.php` returns 409 rather than letting a second batch evict the first — wait
for the run to start, then send.

**"Record only, refresh later"** skips the score refresh, so the answer is
recorded in about a minute instead of up to an hour. It is not an equivalent
path: a mapping does nothing until `update.py` reads it, so the scores it
unlocks arrive with the next scheduled run.

## When the page is down

Nothing here is load-bearing. `propose.py` still opens its PR, and `answer.py`
works from a terminal:

```sh
./answer.py tbench "Fable 5.1" __unmappable__ -w
./answer.py --stdin -w < answers.json
```
