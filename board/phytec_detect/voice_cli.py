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
with the products. Only sentences that start like a command count; the
product part is matched loosely against the product names (search.py),
and the request goes to the demo through /run/phytec-detect/search.
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
    rec = vosk.KaldiRecognizer(model, rate)
    audio = queue.Queue(maxsize=50)

    def callback(data, frames, t, status):
        try:
            audio.put_nowait(bytes(data))
        except queue.Full:
            pass

    with sd.RawInputStream(device=device, samplerate=rate, channels=1, dtype="int16",
                           blocksize=rate // 4, callback=callback):
        while True:
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
