"""Where does the demo recognise badly? A recorder that runs along.

The Smoother (postprocess.py) keeps, per spot on the table, a running score
for every class the accelerator reported there. From that, two problems
show without anyone watching:

  unsicher     the best class stays between LOW and SURE for UNSURE_S
  verwechselt  two classes compete at one spot (both above RIVAL) for
               CONFUSED_S

For each, the detection picture is saved with the scores (at most once per
REPEAT_S for the same products and problem, at most MAX_FILES pictures) --
to look at, and as training pictures for the next model. summary.json is
rewritten every minute: per product how often it was seen, its mean
score, and with which products it was confused -- kept across restarts
(stats.json). nachtrainieren.csv lists the products below RETRAIN_BELOW
mean score (seen at least MIN_FRAMES times), worst first; its first two
columns are those of products.csv, so phytec-photo-queue can take it.
"""

import json
import os
import threading
import time

import cv2

LOW, SURE, RIVAL = 0.3, 0.7, 0.3
UNSURE_S, CONFUSED_S, REPEAT_S = 2.0, 1.0, 10.0
MAX_FILES = 300
RETRAIN_BELOW, MIN_FRAMES = 0.90, 50


class QualityRecorder:
    def __init__(self, out_dir, labels, names=None):
        self.out_dir, self.labels, self.names = out_dir, labels, names or {}
        os.makedirs(out_dir, exist_ok=True)
        self.files = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
        self._since = {}      # (spot key, kind, classes) -> first seen
        self._saved = {}      # (kind, classes) -> last saved
        self.stats = {}       # article -> {"frames", "score_sum", "confused": {article: n}}
        try:  # what earlier runs of the demo saw
            with open(os.path.join(out_dir, "stats.json"), encoding="utf-8") as f:
                self.stats = json.load(f)
        except (OSError, ValueError):
            pass
        self._last_summary = time.monotonic()
        self._busy = False

    def _art(self, cls):
        return self.labels[cls] if cls < len(self.labels) else str(cls)

    def observe(self, frame, spots):
        """frame: the detection picture the spots belong to; spots: the
        Smoother's [box, {class: score}, frames since seen]."""
        now = time.monotonic()
        active = set()
        for box, scores, missing, *_ in list(spots):
            if missing or not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            (c1, s1), (c2, s2) = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
            a1 = self._art(c1)
            st = self.stats.setdefault(a1, {"frames": 0, "score_sum": 0.0, "confused": {}})
            st["frames"] += 1
            st["score_sum"] += s1
            # a spot key that survives small box jitter
            spot = (int((box[0] + box[2]) / 80), int((box[1] + box[3]) / 80))
            problems = []
            if LOW <= s1 < SURE:
                problems.append(("unsicher", (c1,), UNSURE_S))
            if c2 is not None and s1 >= RIVAL and s2 >= RIVAL:
                pair = tuple(sorted((c1, c2)))
                problems.append(("verwechselt", pair, CONFUSED_S))
                a2 = self._art(c2)
                st["confused"][a2] = st["confused"].get(a2, 0) + 1
            for kind, classes, hold in problems:
                key = (spot, kind, classes)
                active.add(key)
                first = self._since.setdefault(key, now)
                if now - first >= hold and now - self._saved.get((kind, classes), -1e9) >= REPEAT_S \
                        and self.files < MAX_FILES and not self._busy:
                    self._saved[(kind, classes)] = now
                    self._save(frame, kind, box, ranked, spots)
        for key in [k for k in self._since if k not in active]:
            del self._since[key]
        if now - self._last_summary >= 60:
            self._last_summary = now
            self._write_summary()

    def _save(self, frame, kind, box, ranked, spots):
        self.files += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        arts = "+".join(self._art(c) for c, _ in ranked[:2])
        base = os.path.join(self.out_dir, f"{stamp}_{kind}_{arts}")
        meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
                "box": [float(v) for v in box],
                "scores": {self._art(c): round(float(s), 3) for c, s in ranked[:4]},
                "spots": [{"box": [float(v) for v in b],
                           "scores": {self._art(c): round(float(s), 3)
                                      for c, s in sorted(sc.items(), key=lambda kv: -kv[1])[:3]}}
                          for b, sc, m, *_ in list(spots) if not m and sc],
                "size": [int(frame.shape[1]), int(frame.shape[0])]}
        picture = frame.copy()
        self._busy = True

        def work():  # JPEG of a 1920 picture: off the drawing loop
            try:
                cv2.imwrite(base + ".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(base + ".json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=1, ensure_ascii=False)
            finally:
                self._busy = False
        threading.Thread(target=work, daemon=True).start()

    def _write_summary(self):
        out = {a: {"name": self.names.get(a, a), "frames": s["frames"],
                   "mean_score": round(s["score_sum"] / max(s["frames"], 1), 3),
                   "confused_with": dict(sorted(s["confused"].items(), key=lambda kv: -kv[1]))}
               for a, s in sorted(self.stats.items())}
        weak = sorted(((a, o) for a, o in out.items()
                       if o["frames"] >= MIN_FRAMES and o["mean_score"] < RETRAIN_BELOW),
                      key=lambda ao: ao[1]["mean_score"])
        try:
            with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
                json.dump(self.stats, f)
            with open(os.path.join(self.out_dir, "nachtrainieren.csv"), "w", encoding="utf-8") as f:
                f.write("Artikelnummer;Bezeichnung;mittlere Sicherheit;gesehen (Bilder);"
                        "verwechselt mit\n")
                for a, o in weak:
                    conf = ", ".join(f"{self.names.get(b, b)} ({n})"
                                     for b, n in list(o["confused_with"].items())[:3])
                    f.write(f"{a};{o['name']};{o['mean_score'] * 100:.0f} %;{o['frames']};{conf}\n")
        except OSError:
            pass
        try:
            with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pictures": self.files,
                           "products": out}, f, indent=1, ensure_ascii=False)
        except OSError:
            pass
