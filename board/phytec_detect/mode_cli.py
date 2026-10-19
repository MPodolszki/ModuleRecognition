"""Two push buttons switch the trade-fair setup between its two modes.

    phytec-mode              # watch the buttons (phytec-mode.service, as root)
    phytec-mode demo         # switch by hand
    phytec-mode training

  demo       phytec-detect: products on the table, framed in the live picture
  training   phytec-trial: the guided photo procedure for a new product, as
             it is really done -- but nothing is saved and nothing trained

The two services conflict, so starting one stops the other. The buttons
are GPIO lines, set in /etc/default/phytec-mode:

    PHYTEC_MODE_DEMO_GPIO="gpiochip3 5"       # chip and line offset
    PHYTEC_MODE_TRAINING_GPIO="gpiochip3 6"
    PHYTEC_MODE_BIAS=pull-up                  # pull-up / pull-down / disabled
    PHYTEC_MODE_ACTIVE_LOW=1                  # button pulls the line to GND

Without them the service waits (and says so) -- switching by hand works
anyway.
"""

import os
import subprocess
import sys
import threading
import time

UNITS = {"demo": "phytec-detect.service", "training": "phytec-trial.service"}


def switch(mode):
    unit = UNITS[mode]
    active = subprocess.run(["systemctl", "is-active", "-q", unit]).returncode == 0
    if active:
        print(f"{mode}: already running", flush=True)
        return
    print(f"-> {mode} ({unit})", flush=True)
    subprocess.run(["systemctl", "start", "--no-block", unit], check=False)


def watch(mode, spec, bias, active_low, lock, last):
    chip, line = spec.split()
    # pressed = the line becomes active; with -l (active low) that is the
    # falling electrical edge, reported as rising
    cmd = ["gpiomon", "-c", chip, "-e", "rising", "-b", bias,
           "--debounce-period", "30ms", "-F", "%o"] + (["-l"] if active_low else []) + [line]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        print(f"{mode} button: {chip} line {line}", flush=True)
        for _ in proc.stdout:
            with lock:
                now = time.monotonic()
                if now - last[0] < 2.0:  # a service just started: let it come up
                    continue
                last[0] = now
                switch(mode)
        proc.wait()
        print(f"gpiomon for {mode} ended ({proc.returncode}), again in 5 s", flush=True)
        time.sleep(5)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        if argv[0] not in UNITS:
            sys.exit(f"usage: phytec-mode [{'|'.join(UNITS)}]")
        switch(argv[0])
        return 0
    bias = os.environ.get("PHYTEC_MODE_BIAS", "pull-up")
    active_low = os.environ.get("PHYTEC_MODE_ACTIVE_LOW", "1") not in ("0", "", "no")
    lock, last, threads = threading.Lock(), [0.0], []
    for mode in UNITS:
        spec = os.environ.get(f"PHYTEC_MODE_{mode.upper()}_GPIO", "").strip()
        if not spec:
            print(f"no button for {mode} (PHYTEC_MODE_{mode.upper()}_GPIO in "
                  f"/etc/default/phytec-mode)", flush=True)
            continue
        t = threading.Thread(target=watch, args=(mode, spec, bias, active_low, lock, last),
                             daemon=True)
        t.start()
        threads.append(t)
    while True:  # nothing configured: stay, so the unit does not flap
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
