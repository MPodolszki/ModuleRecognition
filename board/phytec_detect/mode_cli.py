"""Two push buttons switch the trade-fair setup between its two modes.

    phytec-mode              # watch the buttons (phytec-mode.service, as root)
    phytec-mode demo         # switch by hand
    phytec-mode training

  demo       phytec-detect: products on the table, framed in the live picture;
             pressed again while the demo runs: reset (forget the table)
  training   products of the list without pictures yet (phytec-photo-queue:
             products.csv, no capture/<article>/DONE): phytec-photo-queue
             photographs them one after the other, the open ones listed on
             the monitor; none open: phytec-trial, the guided photo procedure
             as a show -- nothing saved, nothing trained

The two services conflict, so starting one stops the other. The buttons
are GPIO lines, set in /etc/default/phytec-mode:

    PHYTEC_MODE_DEMO_GPIO="gpiochip3 5"       # chip and line offset
    PHYTEC_MODE_TRAINING_GPIO="gpiochip3 6"
    PHYTEC_MODE_BIAS=pull-up                  # pull-up / pull-down / disabled
    PHYTEC_MODE_ACTIVE_LOW=1                  # button pulls the line to GND
    PHYTEC_MODE_PAD_PULLUP="0x3033043C 0x3033047C"  # i.MX8MP pad registers
The i.MX8MP GPIO driver ignores the bias asked for by libgpiod; where the
device tree leaves a button's pad on pull-down (0x106 on the phyVIP's
GPIO1/GPIO2: SPDIF_EXT_CLK, I2C4_SDA), PHYTEC_MODE_PAD_PULLUP lists the
pads' IOMUXC SW_PAD_CTL registers, and the pull is switched to up at start
(only the pull bits; the device tree overlay would be the clean way).

Without them the service waits (and says so) -- switching by hand works
anyway.

A line whose interrupt is taken (phyVIP i.MX95: GPIO2 shares its IRQ with
the pwm-fan tachometer, gpiomon gets "Device or resource busy") is read
every POLL_S instead, through the GPIO character device directly.
"""

import os
import subprocess
import sys
import threading
import time

UNITS = {"demo": "phytec-detect.service", "training": "phytec-trial.service"}
QUEUE_UNIT = "phytec-photo-queue.service"
PRODUCTS_CSV = "/var/lib/phytec-detect/products.csv"
CAPTURE = "/var/lib/phytec-detect/capture"
POLL_S = 0.05


def open_products():
    """[(article, name)] of the list without pictures yet (as phytec-photo-queue)."""
    from .photo_cli import read_products
    try:
        products = read_products(PRODUCTS_CSV)
    except OSError:
        return []
    return [(a, n) for a, n in products
            if not any(os.path.exists(os.path.join(CAPTURE, a, m)) for m in ("DONE", "SKIP"))]


def _active(unit):
    return subprocess.run(["systemctl", "is-active", "-q", unit]).returncode == 0


def switch(mode):
    unit = UNITS[mode]
    if mode == "training":
        if _active(QUEUE_UNIT) or _active(unit):
            print("training: already running", flush=True)
            return
        todo = open_products()
        if todo:  # real work to do: the products still without pictures
            print(f"training: {len(todo)} products without pictures -- "
                  + ", ".join(n or a for a, n in todo), flush=True)
            unit = QUEUE_UNIT
    active = _active(unit)
    if active:
        if mode == "demo":  # demo button while the demo runs: start afresh
            from .search import write_request
            write_request("reset")
            print("demo: reset (table forgotten)", flush=True)
        else:
            print(f"{mode}: already running", flush=True)
        return
    print(f"-> {mode} ({unit})", flush=True)
    subprocess.run(["systemctl", "start", "--no-block", unit], check=False)


PAD_PE, PAD_PUE = 1 << 8, 1 << 6  # i.MX8MP SW_PAD_CTL: pull enable, pull up


def pad_pullup(addresses):
    """Set pull-up (enable + up) in i.MX8MP pad control registers, leaving
    the other bits (drive strength, hysteresis, ...) as they are."""
    import mmap
    import struct
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    try:
        for addr in addresses:
            base, off = addr & ~0xFFF, addr & 0xFFF
            m = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
            old = struct.unpack_from("<I", m, off)[0]
            struct.pack_into("<I", m, off, old | PAD_PE | PAD_PUE)
            print(f"pad 0x{addr:08X}: 0x{old:X} -> 0x{old | PAD_PE | PAD_PUE:X} (pull-up)",
                  flush=True)
            m.close()
    finally:
        os.close(fd)


def _pressed(mode, lock, last):
    with lock:
        now = time.monotonic()
        if now - last[0] < 2.0:  # a service just started: let it come up
            return
        last[0] = now
        switch(mode)


def poll(mode, chip, line, active_low, lock, last):
    """The line read every POLL_S (GPIO v2 uAPI, no interrupt): pressed when
    it turns active and stays so for two reads."""
    import ctypes
    import fcntl
    import struct

    def iowr(nr, size):
        return (3 << 30) | (size << 16) | (0xB4 << 8) | nr
    flags = (1 << 2) | (1 << 8) | ((1 << 1) if active_low else 0)  # input, pull-up, active low
    req = bytearray(592)
    struct.pack_into("<I", req, 0, int(line))                       # offsets[0]
    req[256:256 + 11] = b"phytec-mode"                              # consumer
    struct.pack_into("<Q", req, 288, flags)                          # config.flags
    struct.pack_into("<I", req, 560, 1)                              # num_lines
    buf = ctypes.create_string_buffer(bytes(req), len(req))
    with open(f"/dev/{chip}", "rb") as dev:
        fcntl.ioctl(dev.fileno(), iowr(0x07, 592), buf)
    fd = struct.unpack_from("<i", buf.raw, 588)[0]
    print(f"{mode} button: {chip} line {line} (read every {POLL_S * 1000:.0f} ms)", flush=True)
    values, before = bytearray(struct.pack("<QQ", 0, 1)), [False, False]
    try:
        while True:
            fcntl.ioctl(fd, iowr(0x0E, 16), values)
            on = bool(struct.unpack_from("<Q", values, 0)[0] & 1)
            if on and before[-1] and not before[-2]:
                _pressed(mode, lock, last)
            before = [before[-1], on]
            time.sleep(POLL_S)
    finally:
        os.close(fd)


def watch(mode, spec, bias, active_low, lock, last):
    chip, line = spec.split()
    # pressed = the line becomes active; with -l (active low) that is the
    # falling electrical edge, reported as rising
    cmd = ["gpiomon", "-c", chip, "-e", "rising", "-b", bias,
           "--debounce-period", "30ms", "-F", "%o"] + (["-l"] if active_low else []) + [line]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"{mode} button: {chip} line {line}", flush=True)
        for _ in proc.stdout:
            _pressed(mode, lock, last)
        proc.wait()
        if "busy" in proc.stderr.read():  # its interrupt belongs to another driver
            try:
                poll(mode, chip, line, active_low, lock, last)
            except OSError as error:
                print(f"{mode} button: reading {chip} line {line} failed: {error}", flush=True)
        print(f"gpiomon for {mode} ended ({proc.returncode}), again in 5 s", flush=True)
        time.sleep(5)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        if argv[0] not in UNITS:
            sys.exit(f"usage: phytec-mode [{'|'.join(UNITS)}]")
        switch(argv[0])
        return 0
    pads = os.environ.get("PHYTEC_MODE_PAD_PULLUP", "").split()
    if pads:
        try:
            pad_pullup([int(a, 0) for a in pads])
        except OSError as error:
            print(f"pad pull-up failed: {error}", flush=True)
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
