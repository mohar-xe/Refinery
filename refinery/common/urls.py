"""URL construction, normalization and corruption — the single owner of URL semantics.

One module owns all three, and the corruptor and the verifier both import it. That
is deliberate and it is the reason the verifier can be exact: `normalize()` is not
a second guess at what "the same URL" means, it is literally the same function the
corruptor used to decide what it broke. A verifier with its own comparison would
quietly disagree with the task generator about equivalence, and every mismatch
would show up as an unexplained `TEST_FAIL`.

The corruption catalog is the interesting part. Each entry is a real way URLs get
mangled in transit — copy-paste out of a chat, HTML unescaping, sentence
punctuation, a log line losing a scheme — rather than random string damage. A
model that learns this catalog learns a repair procedure, not a lookup table.
"""

from __future__ import annotations

import random
import re
import string
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

__all__ = [
    "SCHEMES",
    "TLDS",
    "PATH_WORDS",
    "QUERY_KEYS",
    "QUERY_VALUES",
    "make_url",
    "normalize",
    "corrupt",
    "CORRUPTIONS",
    "compose_url",
]

SCHEMES = ("https", "http")
TLDS = ("com", "org", "net", "io", "dev", "co", "ai", "app", "edu", "gov")
SUBDOMAINS = ("www", "docs", "blog", "api", "cdn", "static", "mail", "shop", "status")
HOST_STEMS = (
    "atlas", "meridian", "northwind", "lumen", "quarry", "beacon", "cobalt", "delta",
    "ember", "fathom", "granite", "harbor", "indigo", "juniper", "kestrel", "lantern",
    "monsoon", "nimbus", "orchard", "pinnacle", "quartz", "ridge", "summit", "tundra",
    "umber", "vertex", "willow", "xenon", "yarrow", "zephyr", "basalt", "cinder",
)
PATH_WORDS = (
    "docs", "guide", "pricing", "blog", "2024", "2025", "release", "notes", "api",
    "v1", "v2", "reference", "tutorial", "changelog", "about", "careers", "legal",
    "privacy", "terms", "status", "download", "install", "setup", "config", "faq",
)
QUERY_KEYS = ("utm_source", "utm_medium", "utm_campaign", "id", "ref", "page", "q", "lang")
QUERY_VALUES = ("newsletter", "email", "twitter", "docs", "2", "7", "en", "en-GB", "summary")
FRAGMENTS = ("intro", "installation", "faq", "usage", "examples", "top")


def _slug(rng: random.Random) -> str:
    return rng.choice(HOST_STEMS)


def make_url(rng: random.Random) -> dict:
    """Build a canonical URL and keep its parts addressable.

    Parts are retained because the task needs them: `compose` tasks splice a path
    from one URL onto a query from another, and the corruption catalog operates on
    fields rather than on raw text.
    """
    scheme = rng.choice(SCHEMES)
    host = f"{rng.choice(SUBDOMAINS)}.{_slug(rng)}.{rng.choice(TLDS)}"
    if rng.random() < 0.15:
        host = f"{_slug(rng)}.{rng.choice(TLDS)}"
    port = "" if rng.random() < 0.85 else f":{rng.choice((8000, 8080, 3000, 5000))}"

    depth = rng.randint(1, 3)
    path = "/" + "/".join(rng.choice(PATH_WORDS) for _ in range(depth))
    if rng.random() < 0.25:
        path += "/" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=4))

    query = ""
    if rng.random() < 0.55:
        n = rng.randint(1, 3)
        pairs = [(rng.choice(QUERY_KEYS), rng.choice(QUERY_VALUES)) for _ in range(n)]
        # Canonical order is sorted, so query reordering is a corruption rather
        # than an accident of generation.
        pairs.sort()
        query = "?" + urlencode(pairs, quote_via=quote)

    fragment = ""
    if rng.random() < 0.3:
        fragment = "#" + rng.choice(FRAGMENTS)

    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "path": path,
        "query": query,
        "fragment": fragment,
        "url": f"{scheme}://{host}{port}{path}{query}{fragment}",
    }


# --------------------------------------------------------------------------- #
# Normalization — the equivalence relation the verifier uses
# --------------------------------------------------------------------------- #
_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize(url: str) -> str:
    """Canonical form used for every URL comparison in the project.

    Deliberately conservative: it folds only differences that are *definitionally*
    not part of the URL — scheme case, host case, default port, percent-encoding
    style, query parameter order. It does **not** fold `http` into `https`, add or
    drop `www.`, or strip a trailing slash, because those are distinctions a real
    reconstruction has to get right.

    A tolerant normalizer is the classic way to make a reconstruction benchmark
    measure the normalizer instead of the model.
    """
    text = (url or "").strip()
    if not text:
        return ""
    # Sentence punctuation and wrapping characters are damage, not content.
    text = text.strip("`\"'<>[](){}.,;:!?*_ ")
    if "://" not in text:
        text = "https://" + text.lstrip("/")
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        port = parts.port  # raises on a malformed port, e.g. a double-encoded "&"
    except ValueError:
        # A damaged URL must never crash the verifier: `normalize` is called on
        # agent output, which is arbitrary text. Fall back to a conservative
        # comparison that can only be *stricter* than the full parse.
        return unquote(text).strip().lower()

    scheme = (parts.scheme or "https").lower()
    if port is not None and str(port) == _DEFAULT_PORTS.get(scheme):
        port = None
    netloc = host if port is None else f"{host}:{port}"

    path = unquote(parts.path) or "/"
    path = re.sub(r"/{2,}", "/", path)

    query = ""
    if parts.query:
        pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
        query = urlencode(pairs, quote_via=quote)

    fragment = unquote(parts.fragment)

    out = f"{scheme}://{netloc}{path}"
    if query:
        out += f"?{query}"
    if fragment:
        out += f"#{fragment}"
    return out


def compose_url(base: dict, query_source: dict) -> str:
    """`compose` task family: base URL's location, another URL's query string.

    The result is a URL that appears in no candidate list, so the agent has to
    combine two retrieved facts. This is the family that makes citation checking
    non-trivial.
    """
    query = query_source["query"] if query_source.get("query") else "?ref=combined"
    return f"{base['scheme']}://{base['host']}{base['port']}{base['path']}{query}"


# --------------------------------------------------------------------------- #
# Corruption catalog
# --------------------------------------------------------------------------- #
def _drop_scheme(url: str, rng: random.Random) -> str:
    return re.sub(r"^[a-z]+://", "", url, flags=re.IGNORECASE)


def _drop_www(url: str, rng: random.Random) -> str:
    return re.sub(r"(://)www\.", r"\1", url, count=1, flags=re.IGNORECASE)


def _strip_percent_encoding(url: str, rng: random.Random) -> str:
    return unquote(url)


def _double_encode(url: str, rng: random.Random) -> str:
    """HTML-escaped query separator — what copy-pasting out of a rendered page
    does. Only `&` is touched; escaping `?` as well destroys the URL structure
    rather than damaging it, which is a different (and unrecoverable) task."""
    return url.replace("&", "&amp;")


def _html_entities(url: str, rng: random.Random) -> str:
    return url.replace("/", "&#x2F;").replace("?", "&quest;")


def _sentence_punctuation(url: str, rng: random.Random) -> str:
    return url + rng.choice((".", ",", "!", "?", ";", ":"))


def _markdown_wrapper(url: str, rng: random.Random) -> str:
    return rng.choice((
        f"`{url}`",
        f"<{url}>",
        f"[{url}]",
        f"{url}.",
        f'"{url}"',
    ))


def _whitespace(url: str, rng: random.Random) -> str:
    return url.replace("/", " / ", 1) if "/" in url[8:] else url


def _newlines(url: str, rng: random.Random) -> str:
    head, sep, tail = url.partition("?")
    return head + sep + tail.replace("&", "\n&", 1) if sep else url


def _reorder_query(url: str, rng: random.Random) -> str:
    head, sep, tail = url.partition("?")
    tail, frag_sep, frag = tail.partition("#")
    pairs = tail.split("&")
    if len(pairs) < 2:
        return url
    rng.shuffle(pairs)
    return head + sep + "&".join(pairs) + frag_sep + frag


def _lose_fragment(url: str, rng: random.Random) -> str:
    return url.split("#", 1)[0]


def _move_fragment(url: str, rng: random.Random) -> str:
    """Fragment hoisted in front of the query string. Requires both parts to be
    present — with only a fragment there is nowhere to move it, and inventing a
    query would make the task unsolvable rather than merely damaged."""
    head, sep, frag = url.partition("#")
    if not sep or "?" not in head:
        return url
    base, _, query = head.partition("?")
    return f"{base}#{frag}?{query}"


def _drop_port(url: str, rng: random.Random) -> str:
    return re.sub(r"(://[^/:]+):\d+", r"\1", url)


def _drop_path_segment(url: str, rng: random.Random) -> str:
    head, sep, rest = url.partition("://")
    path, qsep, tail = rest.partition("?")
    parts = path.split("/")
    if len(parts) <= 3:  # host + one segment: too destructive
        return url
    del parts[rng.randint(2, len(parts) - 1)]
    return head + sep + "/".join(parts) + qsep + tail


def _double_slash(url: str, rng: random.Random) -> str:
    head, sep, rest = url.partition("://")
    path, qsep, tail = rest.partition("?")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:
        return url
    parts.insert(rng.randint(2, len(parts) - 1), parts[1])
    return head + sep + "/" + "/".join(parts) + qsep + tail


def _uppercase_host(url: str, rng: random.Random) -> str:
    return re.sub(
        r"(://)([^/?#]+)",
        lambda m: m.group(1) + m.group(2).upper(),
        url,
        count=1,
    )


def _unicode_host(url: str, rng: random.Random) -> str:
    return re.sub(r"\.[a-z]{2,4}(?=[/:?#]|$)", lambda m: ".ünicode" + m.group(0)[1:], url, count=1)


def _truncate(url: str, rng: random.Random) -> str:
    cut = rng.randint(max(len(url) // 2, 12), max(len(url) - 2, 13))
    return url[:cut]


def _truncate_tail(url: str, rng: random.Random) -> str:
    """Drop the query or fragment but keep the origin — the most common real case."""
    head, sep, _rest = url.partition("?")
    return head if sep else url.split("#", 1)[0]


def _http_to_https_typo(url: str, rng: random.Random) -> str:
    return url.replace("https://", "htps://", 1) if url.startswith("https://") else url.replace(
        "http://", "https:/", 1
    )


#: name -> (function, difficulty 1-3)
#: Difficulty only affects sampling weights; it is recorded on the task so the
#: report can break pass@1 down by how hard the corruption was.
CORRUPTIONS: dict[str, tuple] = {
    "drop_scheme": (_drop_scheme, 1),
    "drop_www": (_drop_www, 1),
    "sentence_punctuation": (_sentence_punctuation, 1),
    "markdown_wrapper": (_markdown_wrapper, 1),
    "whitespace": (_whitespace, 1),
    "newlines": (_whitespace, 1),
    "http_typo": (_http_to_https_typo, 2),
    "uppercase_host": (_uppercase_host, 2),
    "double_slash": (_double_slash, 2),
    "reorder_query": (_reorder_query, 2),
    "drop_fragment": (_lose_fragment, 2),
    "fragment_moved": (_move_fragment, 2),
    "drop_port": (_drop_port, 2),
    "strip_percent_encoding": (_strip_percent_encoding, 2),
    "double_encode": (_double_encode, 3),
    "html_entities": (_html_entities, 3),
    "unicode_host": (_unicode_host, 3),
    "drop_path_segment": (_drop_path_segment, 3),
    "truncate": (_truncate, 3),
    "truncate_tail": (_truncate_tail, 2),
    "lose_fragment": (_lose_fragment, 2),
}


def corrupt(url: str, kinds: list[str], rng: random.Random) -> str:
    """Apply the named corruptions in order. Returns the damaged URL.

    A corruption is skipped if it would be a no-op on this URL (e.g. dropping a
    scheme from a URL that has none), so `kinds` is an upper bound rather than a
    guarantee — the task records which ones actually landed.
    """
    out = url
    for kind in kinds:
        fn, _difficulty = CORRUPTIONS[kind]
        damaged = fn(out, rng)
        if damaged != out:
            out = damaged
    return out
