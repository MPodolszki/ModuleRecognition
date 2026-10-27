"""Voice commands with Whisper-tiny on the Hailo-8, inside the demo process.

The board has no HailoRT multi-process service, so a second program cannot
use the Hailo-8 while the demo runs; this thread shares the demo's device
(VDevice group "SHARED", round-robin scheduler) instead. It uses the
whisper_bench package and HEFs of demo-whisper-benchmark (encoder for a
10 s window, decoder with 32 tokens), forced to German.

Always listening on the USB microphone: a simple energy detector cuts out
utterances (start: clearly above the background, end: 0.6 s quiet, at most
the encoder's window: 10 s tiny, 5 s base), Whisper writes them down, and a command counts only after the wake
word "Phytec" -- which Whisper writes in many ways (Fytek, Feitek, Vitec,
...), so it is matched loosely. The rest goes to search.py like a typed
phytec-find request.

Without the wake word (spot=True, the default) a product named anywhere in
an utterance -- "... das ist der i.MX 8M Plus ..." -- frames only that
product for SPOT_S seconds (whisper_grammar.Spotter).
"""

import queue
import re
import threading
import time

import numpy as np

from .whisper_grammar import Encoder, Grammar, Spotter, Wake

DATA = "/usr/share/demo-whisper-benchmark"
# Whisper on the Hailo-8: the HEFs (encoder window in s) and the decoder's
# host-side tokenization arrays. base: Hailo's precompiled hailo-apps
# models (resources_config.yaml, whisper_h8), copied to /opt/phytec-detect.
MODELS = {
    "tiny": (f"{DATA}/hailo", "tiny-whisper-encoder-10s_15dB.hef",
             "tiny-whisper-decoder-fixed-sequence-matmul-split.hef", "tiny", 10),
    "base": ("/opt/phytec-detect/whisper-base", "base-whisper-encoder-5s.hef",
             "base-whisper-decoder-fixed-sequence-matmul-split.hef", "base", 5),
}
RATE = 16000
WAKE_S = 6.0
MIN_SCORE = -2.5  # mean log-probability a restricted command needs
WAKE_MIN = -2.0   # mean log-probability of "Phytec" at the start of an utterance
SPOT_MIN = -4.5   # mean log-probability (no bonus) of a product name said in passing
SPOT_S = 5.0      # a product named in passing: only its frame this long
# <|startoftranscript|> <|de|> <|transcribe|> <|notimestamps|>
GERMAN_PROMPT = [50258, 50261, 50359, 50363]


class Listener:
    """Microphone -> utterances -> Whisper on the Hailo-8 -> search requests."""

    def __init__(self, device="Jabra", log=print,
                 products="/var/lib/phytec-detect/products.csv", spot=True, model="tiny"):
        import sounddevice as sd
        from whisper_bench import audio as wa
        from whisper_bench import tokenizer as tok
        from whisper_bench.backend_hailo import HailoBackend

        tok.FORCED_DECODER_IDS = GERMAN_PROMPT  # generate() reads it at call time
        self._sd, self._wa, self._tok, self.log = sd, wa, tok, log
        folder, encoder, decoder, size, window = MODELS[model]
        self.model = model
        self.backend = HailoBackend(f"{folder}/{encoder}", f"{folder}/{decoder}",
                                    f"{folder}/token_embedding_weight_{size}.npy",
                                    f"{folder}/onnx_add_input_{size}.npy", chunk_length=window)
        self.filters = wa.load_mel_filters(f"{DATA}/assets/mel_filters.npz")
        self.tokenizer = tok.WhisperTokenizer(f"{DATA}/assets/vocab.json")
        self.encoder = Encoder(f"{DATA}/assets/vocab.json")
        self.wake = Wake(self.encoder)
        self.products_csv, self._csv_mtime, self.grammar = products, None, None
        self.spot, self.spotter = spot, None
        self.device = None
        for i, d in enumerate(sd.query_devices()):
            if device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                self.device = i
                break
        self.last_ms = 0.0
        self.awake_until = 0.0  # "Phytec" heard: the demo shows a red dot until then
        self._chunks = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self.backend.close()

    def _grammar(self):
        """The command trie, rebuilt when products.csv changes."""
        import os

        from .photo_cli import read_products
        try:
            mtime = os.stat(self.products_csv).st_mtime
        except OSError:
            mtime = 0
        if mtime != self._csv_mtime:
            self._csv_mtime = mtime
            products = read_products(self.products_csv)
            self.grammar = Grammar(self.encoder, products)
            self.spotter = Spotter(self.encoder, products)
            self.log(f"voice: {self.grammar.size} command token paths, "
                     f"{self.spotter.size} spoken product names")
        return self.grammar

    def transcribe(self, audio, awake):
        """Score the wake word; after it -- in this utterance or the one
        before -- the command the restricted decoder picks:
        (wake score, wake word seen, command or None, its mean log-probability,
        product names said in passing [(name, score)], the free transcript)."""
        wa = self._wa
        audio = wa.normalize_peak(audio)
        window = wa.pad_or_trim(audio, self.backend.chunk_length * RATE)
        mel = wa.log_mel_spectrogram(window, self.filters)
        t0 = time.perf_counter()
        encoded = self.backend.encode(mel)
        wake_score, wake_ids = self.wake.score(self.backend, encoded, GERMAN_PROMPT)
        woke = wake_score >= WAKE_MIN
        command, score, spotted, heard = None, -99.0, [], ""
        if woke or awake:
            prefix = list(GERMAN_PROMPT) + (wake_ids if woke else [])
            command, score = self._grammar().decode(self.backend, encoded, prefix)
        elif self.spot:
            self._grammar()
            # ~6 tokens a second of speech at most: more is Whisper making things up
            spotted, free = self.spotter.spot(self.backend, encoded, GERMAN_PROMPT,
                                              max_tokens=8 + int(6 * len(audio) / RATE))
            heard = self.tokenizer.decode(free).strip()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return wake_score, woke, command, score, spotted, heard

    def _utterances(self):
        """Yield float32 utterances from the energy detector."""
        frame = RATE // 50  # 20 ms
        floor, active, quiet, buf = None, False, 0, []
        while not self._stop.is_set():
            try:
                block = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            for i in range(0, len(block) - frame + 1, frame):
                f = block[i:i + frame]
                rms = float(np.sqrt(np.mean(f * f))) + 1e-6
                floor = rms if floor is None else (0.995 * floor + 0.005 * rms
                                                   if not active else floor)
                # clearly above the room, and not just a clatter: every
                # utterance costs the Hailo 0.5-2.5 s that detection misses
                loud = rms > 4.0 * floor and rms > 0.01
                if not active:
                    buf = (buf + [f])[-15:]  # keep 0.3 s before the start
                    if loud:
                        active, quiet = True, 0
                    continue
                buf.append(f)
                quiet = 0 if loud else quiet + 1
                if quiet >= 30 or len(buf) >= self.backend.chunk_length * 50:  # 0.6 s quiet, or the window
                    if len(buf) - quiet >= 35:  # at least ~0.7 s of sound
                        yield np.concatenate(buf)
                    active, buf = False, []

    def _run(self):
        from .search import is_reset, write_request

        # Whisper wants 16 kHz mono; USB microphones such as the SC420 offer
        # only 44.1/48 kHz stereo (PortAudio's check passes 16 kHz anyway,
        # opening fails): then 48 kHz, every 3 samples averaged
        def callback_for(step):
            def callback(data, frames, t, status):
                mono = data[:, 0]
                if step > 1:
                    mono = mono[:len(mono) // step * step].reshape(-1, step).mean(1)
                try:
                    self._chunks.put_nowait(mono.astype(np.float32))
                except queue.Full:
                    pass
            return callback

        stream, rate = None, RATE
        channels = min(2, int(self._sd.query_devices(self.device, "input")["max_input_channels"]))
        for rate, ch in ((RATE, 1), (3 * RATE, 1), (3 * RATE, channels)):
            try:
                stream = self._sd.InputStream(device=self.device, samplerate=rate, channels=ch,
                                              dtype="float32", blocksize=rate // 10,
                                              callback=callback_for(rate // RATE))
                break
            except self._sd.PortAudioError:
                continue
        if stream is None:
            self.log("voice: the microphone opens neither at 16 nor at 48 kHz")
            return

        woken_until = 0.0
        with stream:
            self.log(f"voice: listening at {rate} Hz (Whisper-{self.model} on the Hailo-8, German"
                     + (", product names without wake word)" if self.spot else ")"))
            for audio in self._utterances():
                awake = time.monotonic() < woken_until
                try:
                    wake_score, woke, command, score, spotted, heard = self.transcribe(audio, awake)
                except Exception as error:  # noqa: BLE001 -- keep listening
                    self.log(f"voice: {type(error).__name__}: {error}")
                    continue
                self.log(f"voice: {len(audio) / RATE:.1f} s, Phytec {wake_score:.2f}"
                         + (" [Phytec]" if woke else "")
                         + (f" -> '{command}' ({score:.2f})" if command else "")
                         + (f" '{heard}'" if heard else "")
                         + "".join(f" [{t} {sc:.2f}]" for t, sc in spotted)
                         + f" ({self.last_ms:.0f} ms)")
                named = [t for t, sc in sorted(spotted, key=lambda x: -x[1]) if sc >= SPOT_MIN]
                if named:
                    self.awake_until = time.monotonic() + 1.5  # the red dot: heard
                    try:
                        write_request(named[0], seconds=SPOT_S)
                    except OSError as error:
                        self.log(f"voice: {error}")
                    continue
                if woke:
                    woken_until = time.monotonic() + WAKE_S
                    self.awake_until = woken_until
                if not (woke or awake) or command is None:
                    continue
                if score < MIN_SCORE:  # talk forced into a command, or only "Phytec"
                    continue
                woken_until = 0.0
                self.awake_until = time.monotonic() + 1.5  # keep the dot a moment
                try:
                    write_request("reset" if is_reset(command) else command)
                except OSError as error:
                    self.log(f"voice: {error}")
