"""Whisper restricted to the demo's commands (constrained greedy decoding).

Free Whisper-tiny writes what it believes to hear -- at the demo table
"Ich bin 6" for "i.MX 6", "an 3.5" for "AM335". After the wake word the
decoder may now only choose tokens that continue one of the allowed
command sentences (a token trie built from products.csv, in the written
forms Whisper uses: "i.MX6", "IMX 6", "AM62x", "AM 62x", ...). Each step
takes the most likely allowed token; the sentence's mean log-probability
tells whether it was a command at all or just talk that the grammar forced
into shape.

Tokens come from a small byte-level BPE encoder over Whisper's vocab.json
(tiktoken style: a merge's rank is its token id).
"""

import json
import re

import numpy as np

EOT = 50257

_PRETOKEN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d| ?[^\s\w]+|\s+(?!\S)|\s+""")


def _bytes_to_unicode():
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapped, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(c) for c in mapped)))


class Encoder:
    """Text -> Whisper token ids (byte-level BPE, rank = id)."""

    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = {t: int(i) for t, i in json.load(f).items()}
        self.byte_map = _bytes_to_unicode()

    def _bpe(self, piece):
        parts = [self.byte_map[b] for b in piece.encode("utf-8")]
        while len(parts) > 1:
            ranks = [(self.vocab.get(parts[i] + parts[i + 1], 1 << 30), i)
                     for i in range(len(parts) - 1)]
            rank, i = min(ranks)
            if rank == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return [self.vocab[p] for p in parts]

    def encode(self, text):
        ids = []
        for piece in _PRETOKEN.findall(text):
            ids += self._bpe(piece)
        return ids


def written_forms(name):
    """How Whisper might write a product name: 'phyCORE-i.MX6 Quad' ->
    'i.MX6 Quad', 'i.MX6', 'i.MX 6', 'IMX6', 'IMX 6', ..."""
    base = re.sub(r"^phy(core|flex|board)[-\s]*", "", name, flags=re.I).strip()
    forms = {base}
    words = base.split()
    for n in range(1, len(words)):
        forms.add(" ".join(words[:n]))  # "i.MX6" for "i.MX6 Quad"
    for f in list(forms):
        plain = f.replace(".", "")
        forms |= {plain, plain.upper(), f.upper()}
        for g in (f, plain, plain.upper()):
            spaced = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", g)  # "IMX 6", "AM 62x"
            forms.add(spaced)
    m = re.match(r"(.*?\d+)[xX]\b(.*)", base)  # "AM62x" also as "AM62"
    if m:
        forms |= {m.group(1), m.group(1).upper(), re.sub(r"(?<=[A-Za-z])(?=\d)", " ", m.group(1))}
    return sorted(f for f in forms if f)


PREFIXES = ["wo ist der", "Wo ist der", "wo ist das", "Wo ist das", "wo ist", "Wo ist",
            "zeig mir den", "Zeig mir den", "zeig den", "Zeig den", "such den", "Such den"]
OTHER = ["alle zeigen", "Alle zeigen", "alles zeigen", "Alles zeigen", "zurücksetzen",
         "Zurücksetzen", "Reset", "reset"]


def sentences(products):
    """All command sentences (without the wake word)."""
    out = set(OTHER)
    for _, name in products:
        for form in written_forms(name):
            out |= {f"{p} {form}" for p in PREFIXES}
            out.add(form)
    return sorted(out)


WAKE_TEXTS = ["Phytec", "phytec", "PHYTEC", "Fytek", "Feitek", "Fütek", "Fytec", "Phytek",
              "Füttek", "Vitec"]


def _logp(backend, encoded, token_ids, pos):
    logits = backend.decode_logits(encoded, token_ids, pos).astype(np.float64)
    logits -= logits.max()
    return logits - np.log(np.exp(logits).sum())


class Wake:
    """Does an utterance start with 'Phytec'? Whisper scores the wake word's
    spellings (token trie, best token each step) instead of writing down
    freely what it heard -- free transcripts of 'Phytec' came out as
    'Torig', 'Dünne', 'Futeck', ... A few decoder steps per utterance."""

    def __init__(self, encoder):
        self.trie, self.texts = {}, {}
        for w in WAKE_TEXTS:
            for lead in (" ", ""):
                ids = encoder.encode(lead + w)
                node = self.trie
                for t in ids:
                    node = node.setdefault(t, {})
                self.texts[tuple(ids)] = w

    def score(self, backend, encoded, prompt):
        """(mean log-probability of the best wake spelling, its token ids)."""
        token_ids = np.zeros((1, backend.sequence_length), dtype=np.int64)
        token_ids[0, :len(prompt)] = prompt
        pos, node, chosen, logps = len(prompt) - 1, self.trie, [], []
        while node:
            logp = _logp(backend, encoded, token_ids, pos)
            best = max(node, key=lambda t: logp[t])
            chosen.append(best)
            logps.append(float(logp[best]))
            node = node[best]
            pos += 1
            token_ids[0, pos] = best
        return float(np.mean(logps)), chosen


class Grammar:
    """Token trie of the command sentences, each with ' ', ', ' or '. '
    before it (after the wake word) and optional '?'/'.'/'!' at the end."""

    def __init__(self, encoder, products):
        self.trie = {}
        self.texts = {}
        for s in sentences(products):
            for lead in (" ", ", ", ". ", "! "):
                for tail in ("", "?", ".", "!"):
                    ids = encoder.encode(lead + s + tail) + [EOT]
                    node = self.trie
                    for t in ids:
                        node = node.setdefault(t, {})
                    self.texts[tuple(ids)] = s
        self.size = len(self.texts)

    def decode(self, backend, encoded, prefix_ids):
        """Constrained greedy decoding after prefix_ids (forced prompt + wake
        word). Returns (sentence or None, mean log-probability)."""
        seq_len = backend.sequence_length
        token_ids = np.zeros((1, seq_len), dtype=np.int64)
        token_ids[0, :len(prefix_ids)] = prefix_ids
        pos = len(prefix_ids) - 1
        node, chosen, logps = self.trie, [], []
        while node and pos < seq_len - 1:
            logits = backend.decode_logits(encoded, token_ids, pos).astype(np.float64)
            logits -= logits.max()
            logp = logits - np.log(np.exp(logits).sum())
            allowed = list(node)
            best = max(allowed, key=lambda t: logp[t])
            chosen.append(best)
            logps.append(float(logp[best]))
            if best == EOT:
                break
            node = node[best]
            pos += 1
            token_ids[0, pos] = best
        text = self.texts.get(tuple(chosen))
        return text, (float(np.mean(logps)) if logps else -99.0)
