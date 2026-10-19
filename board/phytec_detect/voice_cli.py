"""Voice commands for the demo, always listening (USB microphone).

    phytec-voice                         # phytec-voice.service
    phytec-voice --device Jabra --show   # print everything it hears

  "Wo ist der i.MX6?"       only the matching products are framed
  "Alle zeigen"             everything again (also after 30 s by itself)
  "Reset" / "Zurücksetzen"  forget the table (e.g. a guest's module that
                            was taken away and never comes back)

Speech recognition is Vosk (Kaldi) with the small German model, on the
Cortex-A53s -- Whisper does not fit: on the i.MX8MP NPU it is slower than
on the CPU, on the CPU ~4x slower than real time, and the Hailo-8 is busy
with the products. Free recognition guessed everyday words for product
names ("heimlich sechs" for i.MX6), so the recogniser only gets the
commands as a closed list of sentences, built from products.csv (spoken
letters and German numbers, "zweiundsechzig" as well as "sechs zwei"),
anything else is [unk]. The list is rebuilt when products.csv changes.
The request goes to the demo through /run/phytec-detect/search.
"""

import argparse
import json
import os
import queue
import re
import sys
import time

MODEL = "/opt/phytec-detect/voice/vosk-model-small-de-0.15"
PYLIB = "/opt/phytec-detect/pylib"
PRODUCTS = "/var/lib/phytec-detect/products.csv"
_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
_TEENS = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
          "achtzehn", "neunzehn"]
_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig",
         "neunzig"]
PREFIXES = ["wo ist der", "wo ist das", "wo ist die", "wo ist", "wo liegt der", "zeig mir den",
            "zeig mir das", "zeig den", "zeige", "such den", "suche", "finde den"]
OTHER = ["alle zeigen", "alles zeigen", "zeig alle", "reset", "zurücksetzen", "neustart"]


def german(n):
    """0..999 as one German word: 62 -> 'zweiundsechzig'."""
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return (("ein" if ones == 1 else _ONES[ones]) + "und" if ones else "") + _TENS[tens]
    hundreds, rest = divmod(n, 100)
    return ("" if hundreds == 1 else _ONES[hundreds]) + "hundert" + (german(rest) if rest else "")


def spoken(name):
    """Ways to say a product name, as word lists: 'AM62x DSC' ->
    'a m zweiundsechzig x d s c', 'a m sechs zwei x d s c', ..."""
    base = re.sub(r"^phy(core|flex|board)[-\s]*", "", name, flags=re.I)
    variants = [[]]
    for part in re.findall(r"[a-z]+|\d+", base.lower()):
        if part.isdigit():
            options = [[german(int(part))]]
            if len(part) > 1:
                options.append([_ONES[int(d)] for d in part])  # "sechs zwei"
            if int(part) >= 100 and int(part) % 100:  # "dreihundert fünfunddreißig"
                options.append([german(int(part) // 100 * 100), german(int(part) % 100)])
        elif part in ("quad", "plus", "dual", "solo", "lite", "mini"):
            options = [[part]]
        else:
            options = [list(part)]  # letters: "i m x"
        variants = [v + o for v in variants for o in options]
    return [" ".join(v) for v in variants if v]


def grammar(products):
    """Closed sentence list for the recogniser."""
    phrases = set(OTHER)
    for _, name in products:
        for say in spoken(name):
            words = say.split()
            # the whole name and its beginnings ("i m x sechs" for all i.MX6)
            for n in range(min(3, len(words)), len(words) + 1):
                part = " ".join(words[:n])
                phrases.update(f"{p} {part}" for p in PREFIXES)
                phrases.add(part)
    return sorted(phrases) + ["[unk]"]


COMMAND = re.compile(r"\b(wo (ist|sind|liegt|liegen)|zeig|such|finde|alle|reset|"
                     r"zurück ?setzen|zuruecksetzen|neu ?start)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phytec-voice", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="Jabra",
                        help="input device: name part or index (default: Jabra)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--products", default=PRODUCTS)
    parser.add_argument("--show", action="store_true", help="print every recognised sentence")
    args = parser.parse_args(argv)

    sys.path.insert(0, PYLIB)
    import sounddevice as sd
    import vosk

    from .photo_cli import read_products
    from .search import is_reset, matches, request_of, write_request

    vosk.SetLogLevel(-1)
    device = int(args.device) if args.device.isdigit() else None
    if device is None:
        for i, d in enumerate(sd.query_devices()):
            if args.device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                device = i
                break
    info = sd.query_devices(device, "input")
    rate = 16000
    try:
        sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
    except Exception:  # noqa: BLE001 -- the device's own rate then; Vosk resamples
        rate = int(info["default_samplerate"])
    print(f"listening on {info['name']} at {rate} Hz", flush=True)

    model = vosk.Model(args.model)
    csv_mtime, rec = None, None
    audio = queue.Queue(maxsize=50)

    def callback(data, frames, t, status):
        try:
            audio.put_nowait(bytes(data))
        except queue.Full:
            pass

    with sd.RawInputStream(device=device, samplerate=rate, channels=1, dtype="int16",
                           blocksize=rate // 4, callback=callback):
        while True:
            try:
                mtime = os.stat(args.products).st_mtime
            except OSError:
                mtime = 0
            if mtime != csv_mtime:  # (re)build the sentence list
                csv_mtime = mtime
                phrases = grammar(read_products(args.products))
                rec = vosk.KaldiRecognizer(model, rate,
                                           json.dumps(phrases, ensure_ascii=False))
                print(f"{len(phrases)} sentences from {args.products}", flush=True)
            if not rec.AcceptWaveform(audio.get()):
                continue
            text = json.loads(rec.Result()).get("text", "").strip()
            if not text:
                continue
            if args.show:
                print(f"heard: {text}", flush=True)
            if not COMMAND.search(text):
                continue
            products = read_products(args.products)
            if is_reset(text):
                write_request("reset")
                print(f"'{text}' -> reset", flush=True)
                continue
            what = request_of(text)
            found = [products[i][1] for i in matches(what, products)] if what else []
            if what and not found:
                print(f"'{text}' -> no product matches '{what}'", flush=True)
            write_request(text)  # unknown products: the demo says so
            print(f"'{text}' -> {', '.join(found) if found else 'alle'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
