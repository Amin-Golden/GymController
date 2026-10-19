#!/usr/bin/env python3
"""
Hardware test for the RFID reader on an ESP32 (finger_UDP/finger_UDP.ino) -
the entrance unit (cards on UDP 8891) or, with --locker, the locker unit (8892).

Run it on the PC whose IP is pcIP in the sketch. It watches the ESP32 two ways
at once - its USB serial log and the UDP packets it sends - so a failure says
WHERE the chain breaks: reader -> ESP32 (wiring), or ESP32 -> PC (network).

    1. BOOT     - firmware started the RFID UART and sends cards to this PC
    2. PING     - ESP32 reachable on its command port
    3. CARD     - a tapped card arrives here as "RFID:<uid>"
    4. VERDICT  - the ESP32 accepts the PC's answer (OK = blue ring, UNKNOWN = red)
    5. HOLD     - a card held on the reader is sent once, not repeatedly
    6. RETAP    - the same card is sent again once it is taken away and shown again
    7. NOISE    - no checksum errors on the reader line during the test

STOP RetinaFaceL.py if it runs on this PC (it owns UDP 8891), and close the
Arduino Serial Monitor (it owns the COM port).

Usage:
    py -3 test/test_rfid_esp32.py                      # COM4, resets the ESP32 first
    py -3 test/test_rfid_esp32.py --port COM5
    py -3 test/test_rfid_esp32.py --locker             # locker ESP32 (ROLE 1, port 8892)
    py -3 test/test_rfid_esp32.py --no-reset           # keep it running, skip BOOT
    py -3 test/test_rfid_esp32.py --monitor            # just print every card read
                                                       # (handy to collect card IDs)
"""

import argparse
import re
import socket
import sys
import threading
import time

try:
    import serial
except ImportError:
    print("pyserial is missing:  py -3 -m pip install pyserial")
    sys.exit(1)

RESULTS = []   # (name, "PASS"/"FAIL"/"SKIP"/"WARN", detail)


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


class EspSerial:
    """Reads the ESP32's USB log on a background thread and lets tests wait for lines."""

    def __init__(self, port, baud, echo=True):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.2
        # Opening with DTR/RTS asserted can hold the ESP32 in reset or in the bootloader
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.echo = echo
        self.lines = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                print(f"    esp| serial port lost: {e}")
                break
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            with self.lock:
                self.lines.append(line)
            if self.echo:
                print(f"    esp| {line}")

    def reset(self):
        """Pulse EN through RTS, like the Arduino IDE does after an upload."""
        self.ser.rts = True
        time.sleep(0.15)
        self.ser.rts = False

    def mark(self):
        with self.lock:
            return len(self.lines)

    def wait_for(self, pattern, timeout, since=None):
        """First log line matching pattern from line index `since` on; None on timeout."""
        regex = re.compile(pattern)
        start = self.mark() if since is None else since
        deadline = time.time() + timeout
        while True:
            with self.lock:
                new = self.lines[start:]
            for line in new:
                m = regex.search(line)
                if m:
                    return m
            if time.time() >= deadline:
                return None
            time.sleep(0.05)

    def count(self, pattern, since=0):
        regex = re.compile(pattern)
        with self.lock:
            return sum(1 for line in self.lines[since:] if regex.search(line))

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


class CardReceiver:
    """The PC end of the card channel - what rfid_gym.py binds in production."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))

    def wait_card(self, timeout):
        """Next "RFID:<uid>" packet as (uid, sender address), or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.sock.settimeout(max(0.05, min(0.5, deadline - time.time())))
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            if data.startswith(b"RFID:"):
                return data[5:].decode("ascii", "replace").strip(), addr
        return None

    def drain(self, seconds):
        """Swallow packets for a while; returns how many arrived."""
        n = 0
        while self.wait_card(seconds) is not None:
            n += 1
        return n

    def reply(self, addr, text):
        self.sock.sendto(text.encode(), addr)

    def close(self):
        self.sock.close()


def local_ips():
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("192.168.1.1", 9))          # no packet is sent
        ips.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return ips


def countdown(text, seconds, receiver=None):
    print(f"\n>>> {text}")
    for left in range(seconds, 0, -1):
        print(f"    {left}...", end="\r")
        if receiver:
            receiver.drain(1.0)
        else:
            time.sleep(1.0)
    print("          ")


# ---------------------------------------------------------------- tests -----

def test_boot(esp, reset, expect_port):
    if not reset:
        record("BOOT", "SKIP", "--no-reset")
        return None

    start = esp.mark()
    esp.reset()
    print("Resetting the ESP32 and reading its boot log...")
    m = esp.wait_for(r"RFID reader on GPIO (\d+) @ (\d+) baud -> cards to ([\d.]+):(\d+)", 25, since=start)
    if m is None:
        if esp.wait_for(r"Connecting to WiFi", 0, since=start) and not esp.wait_for(r"WiFi OK", 0, since=start):
            record("BOOT", "FAIL", "stuck connecting to WiFi - check ssid/password in the sketch")
        elif esp.mark() == start:
            record("BOOT", "FAIL", "no serial output - wrong COM port, or the board did not reset (press EN)")
        else:
            record("BOOT", "FAIL", "no 'RFID reader on GPIO' line - is the new sketch flashed?")
        return None

    pin, baud, dest_ip, dest_port = m.group(1), m.group(2), m.group(3), m.group(4)
    mine = local_ips()
    if dest_ip not in mine:
        record("BOOT", "FAIL", f"cards go to {dest_ip}:{dest_port} but this PC is {', '.join(sorted(mine))} "
                               f"- set pcIP in the sketch or run this test on {dest_ip}")
    elif int(dest_port) != expect_port:
        role = {8891: "ROLE 0 (entrance)", 8892: "ROLE 1 (locker)"}.get(int(dest_port), "another role")
        record("BOOT", "FAIL", f"the board is {role} and sends to port {dest_port}, but this test listens "
                               f"on {expect_port} - add or drop --locker")
    else:
        record("BOOT", "PASS", f"reader on GPIO {pin} @ {baud} baud, cards to {dest_ip}:{dest_port}")

    r503 = esp.wait_for(r"R503 BOOT (OK|FAIL)|R503 not responding", 10, since=start)
    if r503 and r503.group(1) == "OK":
        record("R503", "PASS", "fingerprint sensor up - the ring will show card verdicts")
    else:
        record("R503", "WARN", "fingerprint sensor not up - cards still work, but the ring shows no verdicts")

    wifi = esp.wait_for(r"WiFi OK\s+ip=([\d.]+)", 0, since=start)
    return wifi.group(1) if wifi else None


def test_ping(esp_ip, cmd_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    try:
        for _ in range(3):
            sock.sendto(b"PING", (esp_ip, cmd_port))
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            reply = data.decode(errors="replace").strip()
            if reply.startswith("PONG"):
                record("PING", "PASS", reply)
                return True
        record("PING", "FAIL", f"no reply from {esp_ip}:{cmd_port} - is this PC on the same network?")
        return False
    finally:
        sock.close()


def test_card(esp, rx):
    start = esp.mark()
    print("\n>>> TAP A CARD on the reader (30 s)...")
    got = rx.wait_card(30)
    if got:
        uid, addr = got
        record("CARD", "PASS", f"card {uid} received from {addr[0]}")
        return got

    seen = esp.wait_for(r"Card (\w+) ->", 0, since=start)
    if seen:
        record("CARD", "FAIL", f"the ESP32 read card {seen.group(1)} but no packet reached this PC - "
                               "allow Python through Windows Firewall (UDP 8891, Private network)")
    elif esp.count(r"RFID checksum error", since=start):
        record("CARD", "FAIL", "the reader sends data but every frame fails its checksum - "
                               "check GND is shared and the TX divider, or RFID_BAUD")
    else:
        record("CARD", "FAIL", "the ESP32 got nothing from the reader - check reader power, "
                               "reader TX -> GPIO 25, and that the card type matches the reader (125 kHz / 13.56 MHz)")
    return None


def test_verdict(esp, rx, addr, ring_ok, ask):
    ok = True
    for verdict in ("OK", "UNKNOWN"):
        start = esp.mark()
        rx.reply(addr, f"RFID:{verdict}")
        if esp.wait_for(rf"RFID verdict: {verdict}", 2, since=start) is None:
            ok = False
            record("VERDICT", "FAIL", f"ESP32 did not log 'RFID verdict: {verdict}' - old firmware?")
            break
        time.sleep(2.2)                               # let each colour show
    if not ok:
        return
    if not ring_ok:
        record("VERDICT", "PASS", "ESP32 accepted OK and UNKNOWN (ring not checked - R503 down)")
    elif ask:
        answer = input("\n>>> Did the ring go BLUE, then flash RED? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            record("VERDICT", "PASS", "blue for OK, red for UNKNOWN")
        else:
            record("VERDICT", "WARN", "ESP32 accepted the verdicts but the ring did not show them")
    else:
        record("VERDICT", "PASS", "ESP32 accepted OK and UNKNOWN")


def test_hold(rx, hold_seconds):
    countdown("REMOVE the card from the reader", 4, rx)
    print(f">>> Now PLACE the card on the reader and KEEP IT THERE until told to remove it (30 s)...")
    first = rx.wait_card(30)
    if first is None:
        record("HOLD", "FAIL", "no card read")
        return None
    print(f"    read {first[0]} - keep holding...")
    extra = rx.drain(hold_seconds)
    print(">>> OK, REMOVE the card.")
    if extra == 0:
        record("HOLD", "PASS", f"held {hold_seconds} s, sent once")
    else:
        record("HOLD", "FAIL", f"sent {extra + 1} times while held - the reader drops the card "
                               "(hold it flat and closer), or raise RFID_REPEAT_MS")
    return first[0]


def test_retap(rx, uid):
    countdown("Keep the card AWAY from the reader", 4, rx)
    print(">>> TAP the same card again (20 s)...")
    got = rx.wait_card(20)
    if got is None:
        record("RETAP", "FAIL", "second tap was not sent - the repeat filter is blocking it")
    elif uid and got[0] != uid:
        record("RETAP", "WARN", f"got a different card ({got[0]}, expected {uid})")
    else:
        record("RETAP", "PASS", "card sent again after being taken away")


def monitor(rx, esp):
    print("\nMonitoring - every card read is printed and answered RFID:OK. Ctrl+C to stop.\n")
    seen = {}
    try:
        while True:
            got = rx.wait_card(1.0)
            if got is None:
                continue
            uid, addr = got
            seen[uid] = seen.get(uid, 0) + 1
            rx.reply(addr, "RFID:OK")
            print(f"{time.strftime('%H:%M:%S')}  card {uid}  from {addr[0]}  (read {seen[uid]}x)")
    except KeyboardInterrupt:
        pass
    if seen:
        print("\nCards seen (paste into clients.rfid_uid):")
        for uid in seen:
            print(f"  {uid}")
    if esp and esp.count(r"RFID checksum error"):
        print(f"\n{esp.count(r'RFID checksum error')} checksum errors on the reader line")
    return 0


def summary():
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    worst = 0
    for name, status, detail in RESULTS:
        print(f"  {name:<8} {status:<5} {detail}")
        worst = max(worst, {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}[status])
    verdict = ["ALL TESTS PASSED", "PASSED WITH WARNINGS", "FAILURES - see above"][worst]
    print(f"\n  -> {verdict}")
    return 0 if worst < 2 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM4", help="ESP32 USB serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--esp", default="192.168.1.120", help="ESP32 IP if not read from the boot log")
    parser.add_argument("--cmd-port", type=int, default=4210)
    parser.add_argument("--rfid-port", type=int, default=None,
                        help="card port (default 8891 entrance, 8892 with --locker)")
    parser.add_argument("--locker", action="store_true", help="test the locker ESP32 (ROLE 1)")
    parser.add_argument("--hold", type=int, default=6, help="seconds to hold the card in the HOLD test")
    parser.add_argument("--no-reset", action="store_true", help="do not reset the ESP32 (skips BOOT)")
    parser.add_argument("--no-serial", action="store_true", help="UDP only (COM port in use elsewhere)")
    parser.add_argument("--quiet", action="store_true", help="do not echo the ESP32 log")
    parser.add_argument("--yes", action="store_true", help="do not ask about the ring colours")
    parser.add_argument("--monitor", action="store_true", help="print every card read until Ctrl+C")
    args = parser.parse_args()
    if args.rfid_port is None:
        args.rfid_port = 8892 if args.locker else 8891
    if args.locker and args.esp == "192.168.1.120":
        args.esp = "192.168.1.121"

    print("=" * 62)
    print(f"RFID ESP32 test ({'locker' if args.locker else 'entrance'}) - serial {args.port}, "
          f"cards on UDP :{args.rfid_port}")
    print("=" * 62)

    esp = None
    if not args.no_serial:
        try:
            esp = EspSerial(args.port, args.baud, echo=not args.quiet)
        except serial.SerialException as e:
            record("SERIAL", "FAIL", f"cannot open {args.port} ({e}) - close the Arduino Serial Monitor, "
                                     "or use --no-serial")
            return summary()

    try:
        rx = CardReceiver(args.rfid_port)
    except OSError as e:
        record("UDP", "FAIL", f"cannot bind UDP {args.rfid_port} ({e}) - stop RetinaFaceL.py on this PC")
        return summary()

    try:
        if args.monitor:
            if esp and not args.no_reset:
                esp.reset()
            return monitor(rx, esp)

        esp_ip = None
        if esp:
            esp_ip = test_boot(esp, reset=not args.no_reset, expect_port=args.rfid_port)
        else:
            record("BOOT", "SKIP", "--no-serial")
        ring_ok = not any(name == "R503" and status != "PASS" for name, status, _ in RESULTS)
        test_ping(esp_ip or args.esp, args.cmd_port)

        noise_from = esp.mark() if esp else 0
        got = test_card(esp, rx) if esp else None
        if esp is None:
            got = rx.wait_card(30)
            record("CARD", "PASS" if got else "FAIL", f"card {got[0]}" if got else "no card packet")
        if got is None:
            return summary()

        if esp:
            test_verdict(esp, rx, got[1], ring_ok, ask=not args.yes)
        else:
            record("VERDICT", "SKIP", "needs the serial log")
        uid = test_hold(rx, args.hold)
        test_retap(rx, uid or got[0])

        if esp:
            errors = esp.count(r"RFID checksum error", since=noise_from)
            if errors:
                record("NOISE", "WARN", f"{errors} checksum errors - check the TX divider and shared GND")
            else:
                record("NOISE", "PASS", "no checksum errors")
        return summary()
    finally:
        rx.close()
        if esp:
            esp.close()


if __name__ == "__main__":
    sys.exit(main())
