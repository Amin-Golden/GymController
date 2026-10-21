#!/usr/bin/env python3
"""
Build finger_UDP.ino for each unit and update it over WiFi (ArduinoOTA).

The units must already run a version of this sketch with OTA in it - that
first version goes over USB once. After that:

    py -3 finger_UDP/ota_upload.py                 # entrance and locker
    py -3 finger_UDP/ota_upload.py --units locker  # one unit
    py -3 finger_UDP/ota_upload.py --compile-only  # just check that both build

Each unit is built with its ROLE (entrance 0, locker 1) and its IP. The OTA
password is read from the sketch (otaPassword). An update takes about 30 s
per unit; the unit reboots and the script waits until it answers PING again.
Needs arduino-cli (the Arduino IDE ships one) and the esp32 core installed.
"""

import argparse
import glob
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKETCH = HERE / "finger_UDP.ino"
FQBN = "esp32:esp32:esp32"
UNITS = {"entrance": (0, "192.168.1.120"), "locker": (1, "192.168.1.121")}
OTA_PORT = 3232
CMD_PORT = 4210


def find_cli(explicit):
    if explicit:
        return explicit
    found = shutil.which("arduino-cli")
    if found:
        return found
    for cand in (r"C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
                 os.path.expandvars(r"%LOCALAPPDATA%\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"),
                 "/usr/share/arduino-ide/resources/app/lib/backend/resources/arduino-cli",
                 "/opt/arduino-ide/resources/app/lib/backend/resources/arduino-cli"):
        if os.path.exists(cand):
            return cand
    sys.exit("arduino-cli not found - install it or pass --cli PATH")


def find_espota():
    roots = [os.path.expandvars(r"%LOCALAPPDATA%\Arduino15"), os.path.expanduser("~/.arduino15"),
             os.path.expanduser("~/Library/Arduino15")]
    for root in roots:
        cores = sorted(glob.glob(os.path.join(root, "packages", "esp32", "hardware", "esp32", "*")))
        for core in reversed(cores):
            exe = os.path.join(core, "tools", "espota.exe")
            if os.name == "nt" and os.path.exists(exe):
                return [exe]
            py = os.path.join(core, "tools", "espota.py")
            if os.path.exists(py):
                return [sys.executable, py]
    sys.exit("espota not found - install the esp32 core in the Arduino IDE (Boards Manager)")


def ota_password():
    m = re.search(r'otaPassword\s*=\s*"([^"]*)"', SKETCH.read_text(encoding="utf-8", errors="replace"))
    if not m:
        sys.exit(f"no otaPassword in {SKETCH}")
    return m.group(1)


def ping(ip, timeout=2.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(b"PING", (ip, CMD_PORT))
        return s.recvfrom(512)[0].decode(errors="replace").strip()
    except OSError:
        return None
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default="entrance,locker", help="comma separated: entrance, locker")
    ap.add_argument("--entrance-ip", default=UNITS["entrance"][1])
    ap.add_argument("--locker-ip", default=UNITS["locker"][1])
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--cli", help="path of arduino-cli")
    args = ap.parse_args()

    units = [u.strip() for u in args.units.split(",") if u.strip()]
    for u in units:
        if u not in UNITS:
            sys.exit(f"unknown unit '{u}' (entrance, locker)")
    cli = find_cli(args.cli)
    espota = None if args.compile_only else find_espota()
    password = ota_password()
    ips = {"entrance": args.entrance_ip, "locker": args.locker_ip}
    build_root = Path(tempfile.gettempdir()) / "finger_udp_ota"

    failed = []
    for unit in units:
        role, ip = UNITS[unit][0], ips[unit]
        build = build_root / unit
        print(f"\n=== {unit}: building (ROLE {role})")
        r = subprocess.run([cli, "compile", "--fqbn", FQBN, "--build-path", str(build),
                            "--build-property", f"compiler.cpp.extra_flags=-DROLE={role}", str(HERE)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:] + r.stderr[-3000:])
            failed.append(unit)
            continue
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("Sketch uses", "Global variables"))))
        if args.compile_only:
            continue

        before = ping(ip)
        if before is None:
            print(f"{unit} ({ip}) does not answer PING - is it on, and running a sketch with OTA?")
        print(f"=== {unit}: uploading to {ip} over WiFi")
        r = subprocess.run(espota + ["-i", ip, "-p", str(OTA_PORT), f"--auth={password}", "-r",
                                     "-f", str(build / "finger_UDP.ino.bin")])
        if r.returncode != 0:
            print(f"upload to {unit} failed - first OTA version must go over USB; "
                  f"also check the password and that the PC and unit share the network")
            failed.append(unit)
            continue
        deadline = time.time() + 60
        reply = None
        while time.time() < deadline and reply is None:
            time.sleep(2)
            reply = ping(ip)
        print(f"=== {unit}: {'back online - ' + reply if reply else 'did not answer within 60 s'}")
        if reply is None:
            failed.append(unit)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nall done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
