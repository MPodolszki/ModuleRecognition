"""Run a phytec_detect tool as another user, without a login session.

    python3 -m phytec_detect.asuser weston phytec_detect.photo_cli --article ...

The bin wrappers used `su`, which opens a PAM session: the tool then lived
outside the systemd unit that started it, so stopping the unit (e.g. an
aborted phytec-photo) left it running with the camera still open. The board
has neither setpriv nor runuser, so the switch is done here.
"""

import os
import pwd
import runpy
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: python3 -m phytec_detect.asuser USER MODULE [ARGS...]")
    user, module = sys.argv[1], sys.argv[2]
    pw = pwd.getpwnam(user)
    if os.getuid() == 0:
        os.initgroups(user, pw.pw_gid)
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
    os.environ.update(HOME=pw.pw_dir, USER=user, LOGNAME=user,
                      XDG_RUNTIME_DIR=f"/run/user/{pw.pw_uid}")
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
