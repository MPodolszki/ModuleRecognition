"""'Wo ist der i.MX6?' -- which products a spoken or typed request means.

    phytec-find "wo ist der i m x sechs"   # the demo shows only those
    phytec-find "alle zeigen"              # everything again
    phytec-find reset                      # forget the table (products taken
                                           # away for good, guests' modules)

The request (from speech recognition, or typed) is written to
/run/phytec-detect/search; the demo reads it and frames only the products
that match, for SEARCH_S seconds or until "alle zeigen".

Matching works on squeezed names: lower case, spoken German numbers as
digits, everything but letters and digits dropped -- "i m x sechs",
"imx 6" and "i.MX6" all become "imx6". A product matches when its squeezed
name (without the phyCORE/phyFLEX family prefix) or its article number
contains the squeezed request, so "imx6" finds i.MX6 Quad, phyFLEX-i.MX6
Quad and i.MX6UL: more than one product is fine when the request is not
unique.
"""

import difflib
import json
import os
import re
import sys
import time

SEARCH_FILE = "/run/phytec-detect/search"
SEARCH_S = 30.0

_UNITS = {"null": 0, "ein": 1, "eins": 1, "eine": 1, "zwei": 2, "zwo": 2, "drei": 3,
          "vier": 4, "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9}
_TEENS = {"zehn": 10, "elf": 11, "zwoelf": 12, "dreizehn": 13, "vierzehn": 14,
          "fuenfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_TENS = {"zwanzig": 20, "dreissig": 30, "vierzig": 40, "fuenfzig": 50, "sechzig": 60,
         "siebzig": 70, "achtzig": 80, "neunzig": 90}
_FILLER = r"^(hey |hallo )?(wo (ist|sind|liegt|liegen) (der|die|das|den)?|zeig( mir)?( den| die| das)?" \
          r"|such(e)?( den| die| das)?|finde?( den| die| das)?)\s*"
_FAMILY = re.compile(r"^(phycore|phyflex|phyboard)")


def _number_word(word):
    """German number word (up to 999) -> digits, else None."""
    m = re.fullmatch(r"(ein|zwei|drei|vier|fuenf|sechs|sieben|acht|neun)?hundert(\w*)", word)
    if m:
        rest = _number_word(m.group(2)) if m.group(2) else "0"
        if rest is None:
            return None
        return str(100 * _UNITS.get(m.group(1) or "ein", 1) + int(rest))
    if word in _UNITS:
        return str(_UNITS[word])
    if word in _TEENS:
        return str(_TEENS[word])
    if word in _TENS:
        return str(_TENS[word])
    m = re.fullmatch(r"(ein|zwei|drei|vier|fuenf|sechs|sieben|acht|neun)und(\w+zig|dreissig)", word)
    if m and m.group(2) in _TENS:
        return str(_TENS[m.group(2)] + _UNITS[m.group(1)])
    return None


# letters as German speech recognition writes them -> the letter
_LETTERS = {"em": "m", "emm": "m", "ix": "x", "iks": "x", "ex": "x", "ie": "i", "de": "d",
            "es": "s", "ce": "c", "ze": "c", "tse": "c", "el": "l", "ka": "k", "pe": "p",
            "te": "t", "ha": "h", "be": "b", "ef": "f", "en": "n", "we": "w", "zet": "z"}


def squeeze(text):
    """'phyCORE-i.MX 8M Plus' / 'i m x acht m plus' -> 'imx8mplus'."""
    t = text.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    # "vier und sechzig" -> "vierundsechzig" (recognisers split compounds)
    t = re.sub(r"\b(ein|zwei|drei|vier|fuenf|sechs|sieben|acht|neun)\s+und\s+"
               r"(zwanzig|dreissig|vierzig|fuenfzig|sechzig|siebzig|achtzig|neunzig)\b",
               r"\1und\2", t)
    t = re.sub(r"\b(ein|zwei|drei|vier|fuenf|sechs|sieben|acht|neun)?\s*hundert\s+", r"\1hundert", t)
    words = re.split(r"[^a-z0-9]+", t)
    return "".join(_number_word(w) or _LETTERS.get(w, w) for w in words if w)


RESET = re.compile(r"\b(reset|zuruecksetzen|zurücksetzen|neu starten|neustart)\b")


def is_reset(text):
    """'Reset' / 'zurücksetzen': forget what lay on the table, start afresh."""
    return bool(RESET.search(text.lower()))


def request_of(text):
    """The product part of a request ('' = show everything again)."""
    t = text.lower().strip().rstrip("?!.")
    if re.search(r"\balle(s)?\b", t) and not re.search(r"\bwo\b", t):
        return ""
    return re.sub(_FILLER, "", t).strip()


def matches(request, products, fuzzy=0.65):
    """Indices of the products a request means. products: [(article, name)].
    Exact part of a name first ("imx6" -> all i.MX6); failing that the
    names that sound closest (speech recognition is not exact). Leading
    words the filler pattern missed ("Woste I-Mix 6") are tried away."""
    words = request.split()
    best_hits, best_score = [], 0.0
    for k in range(len(words)):
        key = squeeze(" ".join(words[k:]))
        if len(key) < 3 or not re.search(r"[a-z]", key) or not re.search(r"\d", key) \
                and k > 0:  # a lone "6" or a stray word is no product name
            continue
        hits, score = _matches(" ".join(words[k:]), products, fuzzy)
        if hits and score >= 1.0:
            return hits  # exact
        if hits and score > best_score:
            best_hits, best_score = hits, score
    return best_hits


def _matches(request, products, fuzzy):
    """(indices, score): score 1.0 for exact hits, else the fuzzy ratio."""
    key = squeeze(request)
    if not key:
        return [], 0.0
    hits, scores = [], []
    for i, (article, name) in enumerate(products):
        sq = squeeze(name)
        short = _FAMILY.sub("", sq)
        if key in short or key in sq or key in squeeze(article):
            hits.append(i)
        # best match of the request against the start of the short name
        best = max(difflib.SequenceMatcher(None, key, short[:n]).ratio()
                   for n in range(min(len(short), max(2, len(key) - 2)), len(short) + 1))
        scores.append(best)
    if hits:
        return hits, 1.0
    top = max(scores)
    return [i for i, sc in enumerate(scores) if sc >= fuzzy and sc >= top - 0.05], top


def write_request(text, path=SEARCH_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"text": text, "time": time.time()}, f)
    os.chmod(tmp, 0o666)
    try:
        os.replace(tmp, path)
    except PermissionError:  # left by another user in the sticky /run dir
        with open(path, "w", encoding="utf-8") as f, open(tmp, encoding="utf-8") as t:
            f.write(t.read())
        os.remove(tmp)


class SearchState:
    """The demo's view of the current request, re-read twice a second."""

    def __init__(self, products, path=SEARCH_FILE):
        self.products, self.path = products, path  # [(article, name)] in class order
        self.text, self.classes, self.until = "", None, 0.0
        self.resets = 0  # counts reset requests: the demo forgets the table on a change
        self._mtime, self._checked = None, 0.0

    def poll(self):
        now = time.monotonic()
        if now - self._checked >= 0.5:
            self._checked = now
            try:
                mtime = os.stat(self.path).st_mtime
            except OSError:
                mtime = None
            if mtime and mtime != self._mtime:
                self._mtime = mtime
                try:
                    with open(self.path, encoding="utf-8") as f:
                        req = json.load(f)
                except (OSError, ValueError):
                    req = None
                if req and time.time() - req.get("time", 0) < SEARCH_S \
                        and is_reset(req.get("text", "")):
                    self.resets += 1
                    self.text, self.classes = "", None
                elif req and time.time() - req.get("time", 0) < SEARCH_S:
                    what = request_of(req.get("text", ""))
                    if what:
                        self.text, self.classes = what, set(matches(what, self.products))
                        self.until = now + SEARCH_S
                    else:
                        self.text, self.classes = "", None
        if self.classes is not None and now > self.until:
            self.text, self.classes = "", None
        return self.classes  # None: show all


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit('usage: phytec-find "wo ist der i.MX6"  |  phytec-find "alle zeigen"')
    text = " ".join(argv)
    write_request(text)
    print(f"request: {'RESET' if is_reset(text) else request_of(text) or '(alle)'} -> {SEARCH_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
