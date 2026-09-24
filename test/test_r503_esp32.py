#!/usr/bin/env python3
"""
Functional test for the R503 ESP32 firmware (finger_UDP/finger_UDP.ino).

Exercises every firmware capability end to end, receiving images through the
SAME UDPFingerprintReceiver class the gym system uses, so a pass here means
fingerprint_gym.py will parse this sensor's output:

    1. PING        - ESP32 reachable, role/uptime/sensor state
    2. STATUS      - R503 responding on UART (baud, library size, templates)
    3. LED         - Aura ring control (red / blue / purple / breathing / off)
    4. FEEDBACK    - enrollment status strings trigger LED prompts
    5. CAPTURE     - full image path: sensor -> UART -> UDP -> production parser
                     (geometry, blankness, dynamic range, transfer time)
    6. QUALITY     - minutiae count via NBIS if installed, else SIFT features
    7. RELIABILITY - repeated captures, success rate

STOP the main gym application first - this script needs the image port.

Usage:
    py -3 test/test_r503_esp32.py                          # entrance ESP32
    py -3 test/test_r503_esp32.py --esp 192.168.1.121 --image-port 8890
    py -3 test/test_r503_esp32.py --skip-scan              # network-only tests
    py -3 test/test_r503_esp32.py --scans 5
    py -3 test/test_r503_esp32.py --monitor                # live view: show every
                                                           # fingerprint received
                                                           # (Ctrl+C or q to stop)
"""

import argparse
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []   # (name, "PASS"/"FAIL"/"SKIP"/"WARN", detail)


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ ", "WARN": "⚠️ "}[status]
    print(f"{icon} {name}: {status}" + (f" - {detail}" if detail else ""))


class CommandChannel:
    """Talks to the firmware's command port (4210) and reads its replies."""

    def __init__(self, esp_ip, cmd_port):
        self.target = (esp_ip, cmd_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 0))                       # ephemeral: replies come back here
        self.sock.settimeout(2.0)

    def send(self, text):
        self.sock.sendto(text.encode(), self.target)

    def ask(self, text, timeout=2.0, retries=3):
        """Send and wait for one reply. Returns the reply string or None."""
        for _ in range(retries):
            self.send(text)
            try:
                self.sock.settimeout(timeout)
                data, _addr = self.sock.recvfrom(512)
                return data.decode(errors="replace").strip()
            except socket.timeout:
                continue
        return None

    def wait_reply(self, timeout):
        """Wait for an unsolicited reply (e.g. CAPTURE:SENT)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.sock.settimeout(min(1.0, max(0.05, deadline - time.time())))
                data, _addr = self.sock.recvfrom(512)
                return data.decode(errors="replace").strip()
            except socket.timeout:
                continue
        return None

    def wait_capture_result(self, timeout):
        """Wait for CAPTURE:SENT/TIMEOUT/UPLOAD_FAIL, skipping the CAPTURE:WAIT ack."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            reply = self.wait_reply(deadline - time.time())
            if reply is None:
                return None
            if reply != "CAPTURE:WAIT":
                return reply
        return None


# ---------------------------------------------------------------- tests -----

def test_ping(cmd):
    reply = cmd.ask("PING")
    if reply and reply.startswith("PONG"):
        status = "PASS" if "sensor=ok" in reply else "WARN"
        record("PING", status, reply)
        return True
    record("PING", "FAIL", "no reply - is the ESP32 powered and on this network?")
    return False


def test_status(cmd):
    reply = cmd.ask("STATUS")
    if not reply or not reply.startswith("STATUS"):
        record("STATUS", "FAIL", "no reply")
        return False
    ok = "sensor=ok" in reply
    record("STATUS", "PASS" if ok else "FAIL", reply)
    if "baud=115200" not in reply:
        record("STATUS baud", "WARN", "sensor not at 115200 - image upload will be slow")
    return ok


def test_led(cmd, auto):
    sequence = ["RED", "BLUE", "PURPLE", "OFF", "BREATH"]
    failures = []
    for color in sequence:
        reply = cmd.ask(f"LED:{color}")
        if reply != "LED:OK":
            failures.append(f"{color}->{reply}")
        time.sleep(0.8)
    if failures:
        record("LED", "FAIL", ", ".join(failures))
        return False
    if auto:
        record("LED", "PASS", "all commands acknowledged")
    else:
        seen = input("  Did the ring light cycle red/blue/purple then breathe? [Y/n] ").strip().lower()
        record("LED", "PASS" if seen in ("", "y", "yes") else "FAIL", "visual check")
    return True


def test_feedback(cmd):
    """The strings fingerprint_gym.py sends during enrollment must not be
    mistaken for commands, and should drive the prompt LEDs."""
    cmd.send("Finger 1 - Sample 2")
    time.sleep(1.0)
    cmd.send("Success Finger 1")
    time.sleep(1.5)
    # No reply is expected for either - just make sure the ESP32 still answers.
    alive = cmd.ask("PING")
    record("FEEDBACK", "PASS" if alive else "FAIL",
           "enrollment strings handled" if alive else "ESP32 stopped responding")
    return bool(alive)


def open_receiver(image_port):
    try:
        from fingerprint_gym import UDPFingerprintReceiver   # the production parser
        return UDPFingerprintReceiver(image_port)
    except OSError as exc:
        record("IMAGE PORT", "FAIL",
               f"cannot bind {image_port} ({exc}) - stop the main gym app first")
        return None


def validate_image(image, elapsed):
    problems = []
    if image is None:
        return ["no image received (timeout, lost packets, or size mismatch)"]
    height, width = image.shape
    if (width, height) not in ((192, 192), (160, 160)):
        problems.append(f"geometry {width}x{height}, expected 192x192 or 160x160")
    elif (width, height) == (160, 160):
        print("    note: this R503 batch has a 160x160 sensor (the 150-template variant) - "
              "the gym pipeline handles it, no change needed")
    spread = float(image.std())
    unique = len(np.unique(image))
    if spread < 8:
        problems.append(f"image looks blank (std {spread:.1f})")
    if unique < 8:
        problems.append(f"only {unique} grey levels - check 4-bit unpacking")
    print(f"    {width}x{height}, std {spread:.1f}, {unique} grey levels, "
          f"range {image.min()}-{image.max()}, received in {elapsed:.1f}s")
    return problems


def test_capture(cmd, receiver, save_dir, auto):
    if not auto:
        input("  Place a finger on the sensor and press Enter... ")
    cmd.send("CAPTURE")

    started = time.time()
    image = receiver.receive_image(timeout=30.0)
    elapsed = time.time() - started

    esp_report = cmd.wait_capture_result(3.0)      # CAPTURE:SENT uart=..ms udp=..ms
    if esp_report:
        print(f"    ESP32 says: {esp_report}")

    problems = validate_image(image, elapsed)
    if image is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"r503_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), image)
        print(f"    saved: {path}")

    if image is not None and esp_report and esp_report.startswith("CAPTURE:SENT") and not problems:
        record("CAPTURE", "PASS", f"{elapsed:.1f}s end to end")
    elif image is not None and not problems:
        record("CAPTURE", "WARN", "image OK but no ESP32 timing report")
    else:
        record("CAPTURE", "FAIL", "; ".join(problems) or "unknown")
    return image


def test_quality(image):
    if image is None:
        record("QUALITY", "SKIP", "no image to analyse")
        return
    try:
        import fingerprint_nbis
        engine = fingerprint_nbis.NbisEngine(verbose=False)
        if engine.available:
            template = engine.extract_template(image)
            if template:
                count = template["minutiae_count"]
                status = "PASS" if count >= 15 else "WARN"
                record("QUALITY (NBIS)", status,
                       f"{count} minutiae, mean quality {template['quality']:.0f}"
                       + ("" if count >= 15 else " - low; press firmer / clean the window"))
            else:
                record("QUALITY (NBIS)", "FAIL", "MINDTCT found no minutiae")
            return
    except Exception as exc:  # noqa: BLE001
        print(f"    (NBIS unavailable: {exc})")

    from fingerprint_processor_improved import ImprovedFingerprintProcessor
    processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)
    features = processor.extract_features(image)
    if features:
        record("QUALITY (SIFT)", "PASS", f"{features['num_features']} keypoints")
    else:
        record("QUALITY (SIFT)", "FAIL", "no usable features in the image")


def test_reliability(cmd, receiver, scans, auto):
    if scans <= 1:
        record("RELIABILITY", "SKIP", "needs --scans > 1")
        return
    ok = 0
    times = []
    for i in range(scans):
        if not auto:
            input(f"  Scan {i + 1}/{scans}: place a finger and press Enter... ")
        cmd.send("CAPTURE")
        started = time.time()
        image = receiver.receive_image(timeout=30.0)
        cmd.wait_capture_result(2.0)
        if image is not None and image.std() > 8:
            ok += 1
            times.append(time.time() - started)
            print(f"    scan {i + 1}: OK ({times[-1]:.1f}s)")
        else:
            print(f"    scan {i + 1}: FAILED")
        time.sleep(1.0)
    rate = ok / scans
    detail = f"{ok}/{scans} received" + (f", avg {np.mean(times):.1f}s" if times else "")
    record("RELIABILITY", "PASS" if rate == 1.0 else ("WARN" if rate >= 0.7 else "FAIL"), detail)
    if rate < 1.0:
        print("    Lost captures usually mean dropped UDP chunks - the protocol has no "
              "retransmission, so check WiFi signal (STATUS shows rssi).")


# ---------------------------------------------------------------- monitor ---

def monitor(args):
    """Live view: print, save and display every fingerprint the ESP32 sends.

    Runs until Ctrl+C (or 'q' in the image window). Touch the sensor and each
    accepted capture appears here - handy for tuning IMG_MIN_STD, checking the
    Img2Tz gate, and watching enrollment traffic.
    """
    receiver = open_receiver(args.image_port)
    if receiver is None:
        return 1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Quality engine: NBIS if installed, SIFT otherwise (loaded once).
    quality = None
    try:
        import fingerprint_nbis
        engine = fingerprint_nbis.NbisEngine(verbose=False)
        if engine.available:
            def quality(img):
                template = engine.extract_template(img)
                return f"{template['minutiae_count']} minutiae" if template else "no minutiae!"
    except Exception:  # noqa: BLE001
        pass
    if quality is None:
        from fingerprint_processor_improved import ImprovedFingerprintProcessor
        processor = ImprovedFingerprintProcessor(embedding_dim=128, use_pca=True)

        def quality(img):
            features = processor.extract_features(img)
            return f"{features['num_features']} SIFT keypoints" if features else "no features!"

    show = not args.no_window
    count = 0
    print(f"\n👂 Monitoring port {args.image_port} - touch the sensor... (Ctrl+C to stop)\n")
    try:
        while True:
            image = receiver.receive_image(timeout=5.0)
            if image is None:
                if show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            count += 1
            now = datetime.now()
            height, width = image.shape
            path = save_dir / f"monitor_{now.strftime('%Y%m%d_%H%M%S_%f')[:19]}.png"
            cv2.imwrite(str(path), image)
            print(f"#{count:<3} {now.strftime('%H:%M:%S')}  {width}x{height}  "
                  f"std {image.std():5.1f}  range {image.min()}-{image.max()}  "
                  f"{quality(image)}  -> {path.name}")

            if show:
                try:
                    # Display only: cubic upscale softens the 16-level (4-bit)
                    # sensor output so it reads as greyscale; saved PNGs stay raw.
                    view = cv2.resize(image, (width * 3, height * 3),
                                      interpolation=cv2.INTER_CUBIC)
                    cv2.putText(view, f"#{count} {now.strftime('%H:%M:%S')} std={image.std():.0f}",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
                    cv2.imshow("R503 live fingerprints (q to quit)", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False                    # headless OpenCV: console only
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if show:
            cv2.destroyAllWindows()
    print(f"\nDone - {count} fingerprints received, saved in {save_dir}")
    return 0


# ---------------------------------------------------------------- main ------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esp", default="192.168.1.121", help="ESP32 IP (entrance default)")
    parser.add_argument("--cmd-port", type=int, default=4210)
    parser.add_argument("--image-port", type=int, default=8889,
                        help="8889 entrance, 8890 locker")
    parser.add_argument("--scans", type=int, default=3, help="captures for the reliability test")
    parser.add_argument("--skip-scan", action="store_true", help="network/LED tests only")
    parser.add_argument("--auto", action="store_true", help="no interactive prompts")
    parser.add_argument("--monitor", action="store_true",
                        help="live view: show every fingerprint received until Ctrl+C")
    parser.add_argument("--no-window", action="store_true",
                        help="monitor without the OpenCV image window (console only)")
    parser.add_argument("--save-dir",
                        default=str(Path(__file__).resolve().parent / "r503_captures"))
    args = parser.parse_args()

    if args.monitor:
        return monitor(args)

    print("=" * 62)
    print(f"R503 ESP32 functional test - {args.esp} (images on :{args.image_port})")
    print("=" * 62)

    cmd = CommandChannel(args.esp, args.cmd_port)

    if not test_ping(cmd):
        summary()
        return 1
    test_status(cmd)
    test_led(cmd, args.auto)
    test_feedback(cmd)

    if args.skip_scan:
        record("CAPTURE", "SKIP", "--skip-scan")
    else:
        receiver = open_receiver(args.image_port)
        if receiver:
            image = test_capture(cmd, receiver, Path(args.save_dir), args.auto)
            test_quality(image)
            test_reliability(cmd, receiver, args.scans, args.auto)
            receiver.close()

    return summary()


def summary():
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    worst = 0
    for name, status, detail in RESULTS:
        print(f"  {name:<18} {status:<5} {detail}")
        worst = max(worst, {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}[status])
    verdict = ["ALL TESTS PASSED", "PASSED WITH WARNINGS", "FAILURES - see above"][worst]
    print(f"\n  -> {verdict}")
    return 0 if worst < 2 else 1


if __name__ == "__main__":
    sys.exit(main())
