#!/usr/bin/env python3
"""
RFID cards for the gym system - entrance and locker room.

Both ESP32 units (finger_UDP.ino) forward each card their serial reader sees as
one UDP packet "RFID:<uid>". The port says which reader it was, the same way
the fingerprint images use 8889/8890:

    UDP_PORT_RFID_ENTRANCE (8891) - entrance ESP32: check in / check out
    UDP_PORT_RFID_LOCKER   (8892) - locker ESP32:   open the member's locker

The card is looked up in clients.rfid_uid and the member is handed to that
reader's callback - the same callbacks the fingerprint readers use, so a card
behaves exactly like a fingerprint at each place.

A verdict goes back to the ESP32 that sent the card, which shows it on the
R503 ring: "RFID:OK" (blue, done), "RFID:DENIED" (red, the callback refused -
no sessions, no locker, cooldown...), "RFID:UNKNOWN" (red, card not linked),
"RFID:ENROLLED" (blue flash) or "RFID:ENROLL_FAIL" (red).

Cards are linked to members with an RFID_ENROLL notification:
    NOTIFY client_changes, '{"action": "RFID_ENROLL", "client_id": 42}';
The next card shown at the ENTRANCE reader within RFID_ENROLL_TIMEOUT seconds
is stored for that member (the locker reader never links cards). Unknown cards are also printed and sent to the status
monitor, so an ID can be copied into clients.rfid_uid by hand.
"""

import os
import socket
import textwrap
import threading
import time

UDP_PORT_RFID_ENTRANCE = int(os.environ.get("RFID_ENTRANCE_PORT", "8891"))
UDP_PORT_RFID_LOCKER = int(os.environ.get("RFID_LOCKER_PORT", "8892"))
RFID_ENROLL_TIMEOUT = float(os.environ.get("RFID_ENROLL_TIMEOUT", "60"))
RFID_PREFIX = b"RFID:"
# Characters per line on the entrance LCD (same setting as fingerprint_gym.py)
LCD_LINE_WIDTH = int(os.environ.get("LCD_LINE_WIDTH", "40"))


def normalize_uid(raw):
    """Canonical card ID as stored in clients.rfid_uid: upper-case, no separators."""
    return "".join(ch for ch in str(raw).upper() if ch.isalnum())


class RFIDReader:
    """One ESP32 card reader: its UDP port and what a member's card does there."""

    def __init__(self, label, port, callback):
        self.label = label
        self.port = port
        self.callback = callback
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(('', port))
        self.receiver.settimeout(1.0)
        self.thread = None
        print(f"✓ RFID {label} receiver listening on port {port}")


class RFIDCardSystem:
    """Listens to the entrance and locker card readers and runs each card's action."""

    def __init__(self, db_helper, entrance_callback=None, locker_callback=None, sock=None,
                 status_endpoints=None, entrance_port=UDP_PORT_RFID_ENTRANCE,
                 locker_port=UDP_PORT_RFID_LOCKER):
        self.db = db_helper
        self.sock = sock
        self.status_endpoints = status_endpoints or []

        self.readers = {}
        if entrance_callback:
            self.readers["ENTRANCE"] = RFIDReader("ENTRANCE", entrance_port, entrance_callback)
        if locker_callback:
            self.readers["LOCKER"] = RFIDReader("LOCKER", locker_port, locker_callback)

        self.pending_enroll = None  # (client_id, display name, deadline), entrance reader only
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        if self.running:
            print("⚠️  RFID listeners already running")
            return
        self.running = True
        for reader in self.readers.values():
            reader.thread = threading.Thread(target=self._listener_loop, args=(reader,), daemon=True)
            reader.thread.start()
            print(f"💳 RFID {reader.label.lower()} listener started (Port {reader.port})")

    def stop(self):
        self.running = False
        for reader in self.readers.values():
            if reader.thread:
                reader.thread.join(timeout=2.0)
        print("🛑 RFID listeners stopped")

    def close(self):
        self.stop()
        for reader in self.readers.values():
            reader.receiver.close()

    def request_enrollment(self, client_id, timeout=RFID_ENROLL_TIMEOUT):
        """Store the next card shown at the entrance reader for this member."""
        info = self.db.get_client_info(client_id)
        if not info:
            print(f"❌ RFID enrollment: client {client_id} not found")
            return False

        name = f"{info['fname']} {info['lname'] or ''}".strip()
        with self.lock:
            self.pending_enroll = (client_id, name, time.time() + timeout)
        self._send_status(f"Show the RFID card for {name} at the entrance reader ({int(timeout)}s)...")
        return True

    def handle_notification(self, notification):
        """Handler for RFID_ENROLL database notifications."""
        client_id = notification.get('client_id')
        if client_id is None:
            print("❌ RFID enrollment notification missing client_id")
            return
        timeout = notification.get('timeout', RFID_ENROLL_TIMEOUT)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = RFID_ENROLL_TIMEOUT
        self.request_enrollment(int(client_id), timeout)

    def _listener_loop(self, reader):
        print(f"👂 RFID {reader.label.lower()} reader listening on port {reader.port}...")

        while self.running:
            try:
                data, addr = reader.receiver.recvfrom(256)
            except socket.timeout:
                self._expire_enrollment()
                continue
            except OSError as e:
                if self.running:
                    print(f"❌ RFID {reader.label.lower()} listener error: {e}")
                    time.sleep(1)
                continue

            if not data.startswith(RFID_PREFIX):
                continue
            uid = normalize_uid(data[len(RFID_PREFIX):].decode('ascii', 'ignore'))
            if not uid:
                continue

            try:
                self.handle_card(uid, addr, reader.label)
            except Exception as e:  # noqa: BLE001 - a bad read must not kill the listener
                print(f"❌ RFID card handling error: {e}")
                import traceback
                traceback.print_exc()

    def handle_card(self, uid, reply_to=None, reader_label="ENTRANCE"):
        """Process one card read: enroll it if an enrollment is waiting, else run the reader's action."""
        reader = self.readers[reader_label]
        print(f"\n💳 {reader.label} - RFID card {uid}")

        # Cards are linked only at the entrance; the locker reader keeps opening lockers meanwhile
        enrollment = self._take_enrollment() if reader.label == "ENTRANCE" else None
        if enrollment:
            self._enroll_card(enrollment, uid, reply_to)
            return

        owner = self.db.get_client_by_rfid(uid)
        if owner is None:
            print(f"❌ [RFID {reader.label}] Unknown card {uid} - Access DENIED")
            self._send_status(f"Unknown RFID card {uid}")
            self._reply(reply_to, "RFID:UNKNOWN")
            self.db.log_access(None, False, None)
            return

        client_id, name = owner
        print(f"✅ [RFID {reader.label}] MATCH: {name} (ID: {client_id})")

        try:
            result = reader.callback(client_id, name, 1.0)
        except Exception as e:
            print(f"❌ RFID {reader.label.lower()} callback error: {e}")
            result = False
        # Callbacks that return nothing are treated as done
        self._reply(reply_to, "RFID:DENIED" if result is False else "RFID:OK")

    def _take_enrollment(self):
        with self.lock:
            enrollment = self.pending_enroll
            if enrollment and time.time() > enrollment[2]:
                enrollment = None
            return enrollment

    def _expire_enrollment(self):
        with self.lock:
            enrollment = self.pending_enroll
            if not enrollment or time.time() <= enrollment[2]:
                return
            self.pending_enroll = None
        self._send_status(f"RFID enrollment for {enrollment[1]} timed out - no card shown")

    def _enroll_card(self, enrollment, uid, reply_to):
        client_id, name, _ = enrollment

        owner = self.db.get_client_by_rfid(uid)
        if owner is not None and owner[0] != client_id:
            # Keep waiting: the member may have picked up someone else's card
            msg = f"Card {uid} already belongs to {owner[1]} (ID: {owner[0]}) - show another card"
            print(msg)
            self._send_status(msg)
            self._reply(reply_to, "RFID:ENROLL_FAIL")
            return

        with self.lock:
            self.pending_enroll = None

        if owner is not None or self.db.set_client_rfid(client_id, uid):
            msg = f"RFID card {uid} linked to {name}"
            self._reply(reply_to, "RFID:ENROLLED")
        else:
            msg = f"Could not save RFID card {uid} for {name}"
            self._reply(reply_to, "RFID:ENROLL_FAIL")
        print(msg)
        self._send_status(msg)

    def _reply(self, addr, message):
        """Send the verdict back to the ESP32 that read the card."""
        if not self.sock or not addr:
            return
        try:
            self.sock.sendto(message.encode("utf-8"), addr)
        except Exception as e:
            print(f"⚠️ Unable to send RFID verdict: {e}")

    def _send_status(self, message):
        if not self.sock or not self.status_endpoints:
            return
        # One packet per display line, so long messages are not cut off on the LCD
        for line in textwrap.wrap(message, LCD_LINE_WIDTH) or [""]:
            for endpoint in self.status_endpoints:
                try:
                    self.sock.sendto(line.encode("utf-8"), endpoint)
                except Exception as send_err:
                    print(f"⚠️ Unable to send status message: {send_err}")
