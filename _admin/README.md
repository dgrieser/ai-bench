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
browser  ──asks────►  api.php     "which repo, and where do I read?"
         ◄──────────  { repo, ref, raw, runs }

browser  ──reads───►  <raw>/llm.json
                      <raw>/_pending/pending.json
              raw.githubusercontent sends Access-Control-Allow-Origin: *,
              so reads need no token

browser  ──writes──►  api.php  ──►  workflow_dispatch: update-benchmarks.yml
                                         │
                                    ./answer.py -w   validates every record
                                    git push         to main, with the deploy key
                                    ./update-all     scores the new mappings
```

The page holds no credential. `api.php` holds one, and it can do exactly two
things: queue this one workflow, and list its runs.

## There is no URL to configure

The page has no domain in it. It asks `api.php` where to read, and `api.php`
derives that from the single `repo` in its config — so the repository name is the
only thing set anywhere, and the two cannot drift apart. The repo it resolved is
printed in the page header, so it is obvious which one a batch is about to go to.

Both files are read from `raw.githubusercontent.com`, not from the published
site, and that is deliberate. The site does serve `llm.json`, but only under its
custom domain: `https://dgrieser.github.io/ai-bench/llm.json` answers **301** to
`https://openbench.david-grieser.de/llm.json`, and a cross-origin redirect
carries no CORS headers, so a derived `github.io` URL fails in the browser.
`llm.html` has the same note at its `DATA_URLS`. Reading from raw also means the
page sees `main` as the last run left it rather than whatever the Pages build has
caught up with — which is what you want from an admin tool.

Changing the Pages custom domain therefore needs no change here. Pointing the
page at a fork or a different repository is one line in the config outside the
docroot.

## Deploying

### What actually goes on the host

Three files, and only three:

| | |
| --- | --- |
| `index.html` | the page |
| `api.php` | the dispatch endpoint |
| `.htaccess` | the gate in front of both |

Everything else here is for you, not for a web server: `config.example.php` is
a template whose real copy belongs *outside* the docroot, and `README.md` and
`deploy` are documentation and tooling.

`./deploy` copies exactly that allowlist — dry run by default, `--go` to
commit — and uses `rsync` or falls back to `scp`:

```sh
./_admin/deploy you@your-host:/path/to/docroot/admin/         # see what it would do
./_admin/deploy you@your-host:/path/to/docroot/admin/ --go    # copy
```

The page needs nothing else: no build step, no bundler, no fonts or scripts
from anywhere. It reads `llm.json` and the queue straight from GitHub.

### Then, outside the docroot

Create the config:

```sh
cp _admin/config.example.php /path/outside/docroot/ai-bench-admin-config.php
$EDITOR /path/outside/docroot/ai-bench-admin-config.php   # paste the token
```

`api.php` tries one level above `DOCUMENT_ROOT` first, then two and three levels
above itself, since hosts disagree about the shape above the docroot. Set
`AI_BENCH_ADMIN_CONFIG` if yours sits somewhere else. When it finds nothing it
logs every path it tried and returns a 500 — it never falls back to a default.

Keep it above the docroot rather than beside `api.php`. If PHP ever stops
running — a bad `.htaccess`, a module falling over — a file inside the docroot is
served as plain text, and that file holds the token.

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
