#!/usr/bin/env python3
"""
Simulate an ESP32's RFID reader (entrance or locker) without hardware.

Sends "RFID:<uid>" to the controller exactly like finger_UDP.ino does and
prints the verdict it answers with (RFID:OK / DENIED / UNKNOWN / ENROLLED / ENROLL_FAIL).

    python test/test_rfid_udp.py 0F00A1B2C3                     # controller on localhost
    python test/test_rfid_udp.py 0F00A1B2C3 --host 192.168.1.100
    python test/test_rfid_udp.py 0F00A1B2C3 --host 192.168.1.100 --port 8892   # locker reader

A card read here goes through the real flow - entry/exit on 8891, opening the
member's locker on 8892 - so use a test member's card ID.
"""

import argparse
import socket

parser = argparse.ArgumentParser(description="Send a fake RFID card read to rfid_gym.py")
parser.add_argument("uid", help="card ID, e.g. 0F00A1B2C3")
parser.add_argument("--host", default="127.0.0.1", help="machine running RetinaFaceL.py")
parser.add_argument("--port", type=int, default=8891, help="8891 entrance, 8892 locker")
parser.add_argument("--timeout", type=float, default=5.0)
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(args.timeout)
sock.sendto(f"RFID:{args.uid}".encode("ascii"), (args.host, args.port))
print(f"Sent RFID:{args.uid} to {args.host}:{args.port}")

try:
    data, addr = sock.recvfrom(256)
    print(f"Verdict from {addr[0]}: {data.decode('utf-8', 'replace')}")
except socket.timeout:
    print("No verdict - is RetinaFaceL.py running and port 8891 open?")
