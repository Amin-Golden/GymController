#!/usr/bin/env python3
"""
RFID card entrance for the gym system.

The entrance ESP32 (finger_UDP.ino, ROLE 0) forwards each card its serial
reader sees as one UDP packet "RFID:<uid>" to UDP_PORT_RFID. This module looks
the card up in clients.rfid_uid and hands the member to the same entrance
callback the fingerprint reader uses, so entry/exit, locker assignment,
sessions and the shared cooldown behave exactly as for a fingerprint.

A verdict goes back to the ESP32 that sent the card, which shows it on the
R503 ring: "RFID:OK" (blue), "RFID:UNKNOWN" (red), "RFID:ENROLLED" (blue flash)
or "RFID:ENROLL_FAIL" (red).

Cards are linked to members with an RFID_ENROLL notification:
    NOTIFY client_changes, '{"action": "RFID_ENROLL", "client_id": 42}';
The next card shown at the entrance within RFID_ENROLL_TIMEOUT seconds is
stored for that member. Unknown cards are also printed and sent to the status
monitor, so an ID can be copied into clients.rfid_uid by hand.
"""

import os
import socket
import threading
import time

UDP_PORT_RFID = int(os.environ.get("RFID_UDP_PORT", "8891"))
RFID_ENROLL_TIMEOUT = float(os.environ.get("RFID_ENROLL_TIMEOUT", "60"))
RFID_PREFIX = b"RFID:"


def normalize_uid(raw):
    """Canonical card ID as stored in clients.rfid_uid: upper-case, no separators."""
    return "".join(ch for ch in str(raw).upper() if ch.isalnum())


class RFIDEntranceSystem:
    """Listens for card reads from the entrance ESP32 and grants entry/exit."""

    def __init__(self, db_helper, entrance_callback, sock=None, status_endpoints=None, port=UDP_PORT_RFID):
        self.db = db_helper
        self.entrance_callback = entrance_callback
        self.sock = sock
        self.status_endpoints = status_endpoints or []

        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(('', port))
        self.receiver.settimeout(1.0)
        self.port = port

        self.pending_enroll = None  # (client_id, display name, deadline)
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        print(f"✓ RFID UDP receiver listening on port {port}")

    def start(self):
        if self.running:
            print("⚠️  RFID listener already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.thread.start()
        print(f"💳 RFID entrance listener started (Port {self.port})")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("🛑 RFID listener stopped")

    def close(self):
        self.stop()
        self.receiver.close()

    def request_enrollment(self, client_id, timeout=RFID_ENROLL_TIMEOUT):
        """Store the next card shown at the entrance for this member."""
        info = self.db.get_client_info(client_id)
        if not info:
            print(f"❌ RFID enrollment: client {client_id} not found")
            return False

        name = f"{info['fname']} {info['lname'] or ''}".strip()
        with self.lock:
            self.pending_enroll = (client_id, name, time.time() + timeout)
        self._send_status(f"💳 Show the RFID card for {name} at the entrance reader ({int(timeout)}s)...")
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

    def _listener_loop(self):
        print(f"👂 RFID reader listening on port {self.port}...")

        while self.running:
            try:
                data, addr = self.receiver.recvfrom(256)
            except socket.timeout:
                self._expire_enrollment()
                continue
            except OSError as e:
                if self.running:
                    print(f"❌ RFID listener error: {e}")
                    time.sleep(1)
                continue

            if not data.startswith(RFID_PREFIX):
                continue
            uid = normalize_uid(data[len(RFID_PREFIX):].decode('ascii', 'ignore'))
            if not uid:
                continue

            try:
                self.handle_card(uid, addr)
            except Exception as e:  # noqa: BLE001 - a bad read must not kill the listener
                print(f"❌ RFID card handling error: {e}")
                import traceback
                traceback.print_exc()

    def handle_card(self, uid, reply_to=None):
        """Process one card read: enroll it if an enrollment is waiting, else grant entry/exit."""
        print(f"\n💳 ENTRANCE - RFID card {uid}")

        enrollment = self._take_enrollment()
        if enrollment:
            self._enroll_card(enrollment, uid, reply_to)
            return

        owner = self.db.get_client_by_rfid(uid)
        if owner is None:
            print(f"❌ [RFID] Unknown card {uid} - Access DENIED")
            self._send_status(f"❌ Unknown RFID card {uid}")
            self._reply(reply_to, "RFID:UNKNOWN")
            self.db.log_access(None, False, None)
            return

        client_id, name = owner
        print(f"✅ [RFID] MATCH: {name} (ID: {client_id})")
        self._reply(reply_to, "RFID:OK")

        if self.entrance_callback:
            try:
                self.entrance_callback(client_id, name, 1.0)
            except Exception as e:
                print(f"❌ RFID entrance callback error: {e}")

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
        self._send_status(f"⚠️ RFID enrollment for {enrollment[1]} timed out - no card shown")

    def _enroll_card(self, enrollment, uid, reply_to):
        client_id, name, _ = enrollment

        owner = self.db.get_client_by_rfid(uid)
        if owner is not None and owner[0] != client_id:
            # Keep waiting: the member may have picked up someone else's card
            msg = f"⚠️ Card {uid} already belongs to {owner[1]} (ID: {owner[0]}) - show another card"
            print(msg)
            self._send_status(msg)
            self._reply(reply_to, "RFID:ENROLL_FAIL")
            return

        with self.lock:
            self.pending_enroll = None

        if owner is not None or self.db.set_client_rfid(client_id, uid):
            msg = f"✅ RFID card {uid} linked to {name}"
            self._reply(reply_to, "RFID:ENROLLED")
        else:
            msg = f"❌ Could not save RFID card {uid} for {name}"
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
        for endpoint in self.status_endpoints:
            try:
                self.sock.sendto(message.encode("utf-8"), endpoint)
            except Exception as send_err:
                print(f"⚠️ Unable to send status message: {send_err}")
