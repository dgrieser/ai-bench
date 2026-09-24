"""Which pages a hand-entered score may cite.

A hand entry is the weakest rung of _precedence.source_rank(): it seeds a
column until a scraper measures it. But in a column no scraper reaches, or for
a model no scraper has picked up yet, nothing ever arrives to replace it, and
a hand entry is whatever the column says -- often at the top of it
(issue #229: a news alert led livecodebench, a blog post led
osworld_verified, a gist and a personal tracker carried aime_2026, a tweet
carried deepswe_1_0).

The page cited has to be where the number was *published*: the model card,
the paper, the vendor's own announcement, or the benchmark's leaderboard. A
news story, blog, social post or paste that repeats it is not a source anybody
can check the number against -- it is one retelling removed, and it is the
retelling where transcription errors and rounded or cherry-picked figures come
in. SECONDARY_HOSTS lists the kinds of hosts that have actually been cited
that way; edit.py and the admin answers refuse them as a score's page, and
test_hand_sources.py holds llm.json to the same rule.

A denylist rather than an allowlist because the primary publications are
open-ended -- every lab has its own domain and blog path -- while the
retellings that have shown up come from a short list of hosts. A new one is
added here when it is caught.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Hosts that retell a number rather than publish it. A subdomain matches its
# parent (``foo.substack.com`` is ``substack.com``); ``www.`` is ignored.
SECONDARY_HOSTS = frozenset({
    # Social posts and pastes: a screenshot or a claim, not a publication.
    "x.com",
    "twitter.com",
    "threads.net",
    "bsky.app",
    "linkedin.com",
    "reddit.com",
    "news.ycombinator.com",
    "youtube.com",
    "gist.github.com",
    "pastebin.com",
    # Blog platforms: anyone's write-up of somebody else's table.
    "medium.com",
    "substack.com",
    "dev.to",
    "hashnode.dev",
    # News, newsletters and explainer blogs.
    "aiweekly.co",
    "thenextweb.com",
    "datacamp.com",
    "edenai.co",
    "felloai.com",
    "venturebeat.com",
    "techcrunch.com",
    "theverge.com",
    "marktechpost.com",
    "analyticsvidhya.com",
    # Personal trackers that compile numbers from elsewhere.
    "atlas.kevinhu.io",
})


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_secondary_source(url: str | None) -> bool:
    """Whether url is a retelling of a score rather than where it was published."""
    if not url:
        return False
    host = _host(url)
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in SECONDARY_HOSTS)


def secondary_source_error(url: str) -> str:
    """The message a refused score page is reported with."""
    return (
        f"{url} is a news, blog or social page ({_host(url)}), not where the score was "
        "published. Cite the primary publication instead: the model card, the paper, "
        "the vendor's announcement, or the benchmark's leaderboard."
    )
