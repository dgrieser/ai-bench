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

**Send `index.html` and `api.php` together.** They share a contract — the page
asks the endpoint which repository to read — and `api.php` reports a version so
a half-updated pair says so. If the page reports that `api.php` is older than
it is, re-run `./deploy`; uploading only the page is what causes it.

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

The config takes an optional third value, `aa_api_key`: an Artificial Analysis
key, which is what lets the Models tab list AA slugs for a rename. Leave it
empty and that one control says so; everything else works untouched.

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

(The optional `aa_api_key` beside it is an Artificial Analysis key, not a GitHub
one. It is read-only against a public leaderboard's API, and the worst it can do
is list model slugs.)

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
  falls back to the full list — of models, of benchmark keys, or, for the
  "which AA slug is this model" question, of Artificial Analysis' own slugs,
  which needs the `aa_api_key` below.
- **New models** — *Add it* runs `add.py` and records the answer; *Ignore it*
  writes `__ignored__`, which removes the entry and stops the slug being offered.
  (These are genuinely different records, not two values of one field — see the
  comment in `_answers.py`, and `_new_models.apply_decisions`.)

  Above those cards is **Add a model**, for the other direction: a release
  Artificial Analysis does not track, so nothing offered it and there is no
  question to answer. Name it and fill in as much as you know; `add.py` fills
  the rest from AA when it happens to know the name, and leaves it blank when it
  does not. The name is checked as you type — it has to be a slug, and it says
  whether AA already publishes one exactly like it, which decides whether scores
  arrive by themselves (see *A hand-added model and AA* below).
- **Models** — edit every field `add.py` asks for — `params`, `context`, `url`,
  `creator`, `creator url`, `date added` — and any non-derived score. The two
  sets are the same on purpose: nothing is enterable once and then frozen.
  (`vram` is not among them; Spheron writes it, and a hand-typed figure would be
  recomputed away.)

  `params`, `context` and `creator` offer what `llm.json` already holds — the
  sizes and creator names in use, sorted by what they measure rather than
  alphabetically, so `1m` follows `512k`. They stay free text: a creator or a
  size the site has never seen still has to be typeable. Naming a creator it
  does know fills in **creator url** as well, with the page most of that
  creator's models point at — nine creators in `llm.json` currently have two
  spellings of theirs, which is what this stops growing. A URL typed by hand is
  never overwritten. The same three lists back **Add a model** above.

  Adding a score also asks when it was read and what page it was read from: the
  date defaults to today (yours, not the runner's — the run can start on the
  other side of midnight) and the page to nothing, which is what a hand edit has
  always meant. Naming one is not cosmetic. `_precedence.source_rank()` reads an
  unattributed score as hand-entered, the weakest rung, so any scraper replaces
  it; crediting the leaderboard it actually came from moves it onto that page's
  rung, where only that page or a better one may. One date and one page cover
  every score in the card, which is how a sitting goes — a second leaderboard is
  a second batch.

  At the foot of the card is **artificial analysis**, which says whether AA
  publishes this exact slug and, when it does not, offers the slugs that look
  like it. Picking one is a *rename*, not a mapping — see below. A rename is
  sent on its own: one record per model per run, so the fields above grey out
  while one is drafted.
- **Runs** — the last few runs of the workflow.

The icon in the header cycles the theme: follow the system, force light, force
dark. It is stored under the same `theme` key `llm.html` uses, so both pages
agree on one device.

The tab icon is the site's, inlined as base64 between this page's
`brand-icons` markers rather than linked: `.htaccess` serves nothing in this
directory but `index.html` and `api.php`, and the host has no copy of the icon
set to link to. `make_favicons.py` writes it — do not edit that block by hand.

Answers batch up and go in one dispatch, so a sitting is one commit — up to 25
of them. The action bar starts naming that cap at 15 ("15 of 25 ready to send —
room for 10 more", in amber) so it is known while there is still a choice about
what belongs in this sitting; past 25 it turns red, says how many to un-answer,
and greys out **Send**, because `api.php` and `answer.py` would both refuse the
batch anyway — and 25 is their number (`_answers.MAX_RECORDS`), not the page's.
The reason for it is the concurrency group described below: a sitting that
needed two runs would evict its own second run.

A few things worth knowing:

**Reload after a run to confirm.** The run's summary is the record of what
landed, not its colour: `update-all` is `continue-on-error`, so one dead scraper
reds a run whose answers applied perfectly well. Reload and check the question
has left the queue.

**A queued run blocks a second batch.** The workflow's concurrency group holds
one run plus one queued, and a third arrival cancels the queued one silently.
`api.php` returns 409 rather than letting a second batch evict the first — wait
for the run to start, then send.

**An answered question can come back before the queue catches up.** Two ways
that happens. The run pushes the re-rendered queue seconds before it reports
completed, and `raw.githubusercontent` does not always serve that push straight
away — so the read that follows a run can return the queue as it was *before*
it, every question the run just answered listed again. And **record only**
skips the refresh, so it does not republish the queue at all: those answers stay
listed until the next scheduled refresh, hours later.

The page therefore keeps the keys of an applied batch after its in-flight lock
lifts (`ai-bench-admin-applied` in `localStorage`, survives a reload): those
cards stay disabled and marked *answered*, with an amber note. An entry clears
itself on the first queue read that no longer asks its question — that read is
the only proof there is, so a clock does not clear it. The two exceptions are a
new-model line, which the site re-offers on every run until the model is
dismissed and which therefore expires after 30 minutes, and a 7-day backstop
against a lock nothing can clear.

Without this, answering one of them again is refused by `answer.py` as a
question nothing asked, and a batch is all or nothing, so one duplicate takes
the whole sitting down with it. Only the queue's own questions are held this
way — they are recorded when the batch is *sent*, not when it settles, because
settling happens inside the run-list read, before the queue has been read at
all. A model edit or a rename is not answered against the queue, so re-sending
one is simply a second edit and is never held.

**What a failed run means depends on which step failed.** `answer.py` writes all
of a batch or none of it, so a batch it refused leaves nothing behind and its
questions must come back. But the answers are applied, committed and pushed
*before* `update-all` runs, and `Fail if a step of update-all failed` is the last
step in the workflow — so a dead scraper reds a run whose answers are already in
`main`, and handing those cards back would offer the duplicate this whole
mechanism exists to refuse.

The run's conclusion cannot tell those apart, so the page asks
`api.php?answered=<run id>` for the outcome of the workflow's own *Apply the
answers* step (`API_VERSION` 4; `ANSWER_STEP` in `api.php` names the step, and
`test_answers.py` checks that name against the workflow). Step failed: the page
says the batch was refused, in red, and unlocks its questions. Step succeeded
but the run failed later: amber, the answers landed, cards stay locked. No
answer — an `api.php` older than this page, or a read that failed: the cards
stay locked and the note says the page could not tell, which is the safe half of
the choice. **Upload `api.php` together with `index.html`**; `./_admin/deploy`
sends both.

**Answered cards stay disabled until their run finishes.** The queue is only
republished when the run pushes, so a question you have just answered is still
listed, and answering it again is worse than a wasted tap: the second record
carries the same `if_previous`, the run applies the first, and `answer.py` then
rejects the second as stale — and a batch is all or nothing, so every other
answer in that sitting goes down with it. The page therefore remembers what it
dispatched, marks those cards *sent*, and disables them until the run is no
longer in flight; everything else stays answerable. The lock survives a reload
(that is the advice above, after all) and lifts only on a run list the page
actually managed to read, never on a failed one.

**"Record only, refresh later"** skips the score refresh, so the answer is
recorded in about a minute instead of up to an hour. It is not an equivalent
path: a mapping does nothing until `update.py` reads it, so the scores it
unlocks arrive with the next scheduled run.

## A hand-added model and AA

A model added by hand exists before any source knows about it. What happens when
Artificial Analysis catches up depends on one comparison, and it is worth
knowing which case you are in because only one of them resolves itself:

- **AA publishes the same slug.** Nothing to do, ever.
  `update.resolve_aa_slugs()` reads AA for a model whose *name* is a slug there,
  so the scores simply start arriving on the next refresh. The Models tab says
  so when this is already true.
- **AA publishes a different slug.** Nothing arrives, and nothing will. Two ways
  out, and the page offers the second:
  - *Map it* — keep the name and record which AA slug it means. The Queue tab
    asks exactly this, as soon as a refresh has noticed the model has no AA
    index. Right when the name is one this site wants to keep.
  - *Rename it* — take the AA slug as the entry's name. Right when the name was
    only ever a placeholder, and it leaves nothing to maintain: the identity
    match above takes over.

The comparison is byte for byte, so `glm-5.3` and `glm-5-3` are the second case,
however alike they look. That pair is what the suggestions call *the same name
bar punctuation*, and it is the usual reason to rename.

A rename is not an edit of `llm.json`. The name is written down in every
source's mapping file too, and one left behind does not fail — it silently stops
matching, and the scores that source used to write stop arriving. `_rename.py`
moves all of them at once, reading the file list off `propose.ROUTES` so a
source added there is never quietly forgotten, and `./rename.py old new` does
the same job from a terminal.

**The slug list needs a key.** The suggestions come from AA's own model list,
which `api.php` fetches with the `aa_api_key` in its config — the same key
`artificialanalysis.py` reads from `ARTIFICIAL_ANALYSIS_API_KEY`. It is proxied
rather than fetched by the page because the AA API authenticates with a header,
so a browser request preflights and AA answers no CORS headers; keeping it
server-side also keeps the key out of every browser that opens the page. Without
a key the page says the list is unavailable and nothing else changes — the
endpoint reports whether it has one, so an older `api.php` costs the
suggestions rather than the whole page.

## When the page is down

The queue is not lost with it. `_pending/pending.json` is published on every
successful refresh whatever else happens, so the questions are all still there,
and `answer.py` works from a terminal:

```sh
./answer.py tbench "Fable 5.1" __unmappable__ -w
./answer.py --stdin -w < answers.json
./rename.py glm-5.3 glm-5-3 -w        # what the Models tab's rename does
```

The proposal PR is the other way back, and it is now **opt-in** — nothing
automatic opens one, since a PR nobody intends to merge is noise that also
blocks the next proposal (the step skips while one is open). Run
`update-benchmarks.yml` by hand with `propose` ticked and it behaves as it
always did.
