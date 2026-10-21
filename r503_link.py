#!/usr/bin/env python3
"""
Talking to the R503 through the ESP32's command port (UDP 4210), for the
commands that move images and templates: finger_UDP/finger_UDP.ino.

    replies     text "<CMD>:..." and binary blobs: "FPBD" + kind + total(u32 LE)
                + offset(u32 LE) + payload, reassembled by offset
    uploads     "FPUP" + slot + total + offset + payload; the ESP32 acks every
                chunk with "UP:<slot> <got>/<total>" ('I' image, 'B' template)

SensorLink is what the gym system uses for the sensor double-check
(fingerprint_gym.py):
    image_template(image)   the R503's own template of an image (IMG2TMPL FAST):
                            stored at enrollment, ~2 KB
    prepare_confirm(stamp)  the sensor rebuilds the template of the press it just
                            sent (CONFIRM): it becomes the side Match compares with
    match(template)         the R503's own Match of a stored template against that
                            press (MATCHTMPL) -> (matched, score)

Each SensorLink has its own UDP socket, so the entrance and locker listeners
can double-check at the same time. The test tools in test/ use the same helpers.
"""

import socket
import struct
import time
from typing import Optional, Tuple

import numpy as np

COMMAND_PORT = 4210

# UpImage sizes this firmware knows: bytes -> (width, height, 4-bit packed)
FORMATS = {
    36864: (192, 192, False), 18432: (192, 192, True),
    25600: (160, 160, False), 12800: (160, 160, True),
}
BLOB_HEADER = struct.Struct("<4scII")          # "FPBD", kind, total, offset
UP_HEADER = struct.Struct("<4scII")            # "FPUP", slot, total, offset
CHUNK = 1380


def decode_image(data: bytes):
    """Sensor UpImage bytes -> 8-bit image, or None for an unknown size."""
    fmt = FORMATS.get(len(data))
    if fmt is None:
        return None
    w, h, packed = fmt
    a = np.frombuffer(data, np.uint8)
    if packed:                                   # high nibble = first pixel
        a = np.stack([a >> 4, a & 0x0F], axis=1).reshape(-1) * 17
    return a.reshape(h, w)


def pack_image(img, depth: int = 4) -> bytes:
    """8-bit image -> the bytes DownImage takes (4-bit packed, as UpImage sends)."""
    if depth == 8:
        return np.ascontiguousarray(img, np.uint8).tobytes()
    nib = np.clip(np.rint(img.astype(np.float32) / 17.0), 0, 15).astype(np.uint8).reshape(-1)
    return ((nib[0::2] << 4) | nib[1::2]).astype(np.uint8).tobytes()


class _Blob:
    def __init__(self, total):
        self.data = bytearray(total)
        self.have = 0

    def put(self, offset, payload):
        self.data[offset:offset + len(payload)] = payload
        self.have += len(payload)

    def complete(self):
        return self.have >= len(self.data)


def run_command(sock, target, cmd, timeout, on_status=None, prefix=None):
    """Send one command and collect its blobs until "<prefix>:DONE|FAIL|TIMEOUT".

    Returns (final reply or None on timeout, {kind: bytes}). prefix: what the
    replies start with when it differs from the command ("TMPL RAW" answers "TMPL:...").
    """
    prefix = prefix or cmd.split()[0]
    terminal = tuple(f"{prefix}:{end}" for end in ("DONE", "FAIL", "TIMEOUT"))
    blobs = {}
    data = None
    for _ in range(3):                            # the request itself is plain UDP
        sock.sendto(cmd.encode(), target)
        try:
            sock.settimeout(2.0)
            data, _ = sock.recvfrom(2048)
            break
        except socket.timeout:
            continue
    if data is None:
        return None, blobs

    deadline = time.time() + timeout
    while True:
        if not data.startswith(b"FPBD"):
            text = data.decode(errors="replace").strip()
            if text.startswith(terminal):
                return text, {k: bytes(b.data) for k, b in blobs.items() if b.complete()}
            if on_status:
                on_status(text)
        elif len(data) >= BLOB_HEADER.size:
            _, kind, total, offset = BLOB_HEADER.unpack_from(data)
            blobs.setdefault(kind.decode(), _Blob(total)).put(offset, data[BLOB_HEADER.size:])

        remaining = deadline - time.time()
        if remaining <= 0:
            return None, blobs
        try:
            sock.settimeout(remaining)
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            return None, blobs


def wait_for(sock, text, timeout) -> bool:
    """Wait for one exact text reply; False on timeout or an upload error."""
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        try:
            sock.settimeout(remaining)
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            return False
        reply = data.decode(errors="replace").strip()
        if reply == text:
            return True
        if reply.startswith("UP:") and reply.endswith("ERR"):
            return False


def upload(sock, target, slot, data, timeout=1.0, retries=5) -> bool:
    """Send data to the ESP32's upload slot ('I' image, 'B' template), chunk by acked chunk."""
    for off in range(0, len(data), CHUNK):
        part = data[off:off + CHUNK]
        pkt = UP_HEADER.pack(b"FPUP", slot.encode(), len(data), off) + part
        want = f"UP:{slot} {off + len(part)}/{len(data)}"
        for _ in range(retries):
            sock.sendto(pkt, target)
            if wait_for(sock, want, timeout):
                break
        else:
            return False
    return True


def fields(reply: str) -> dict:
    """'CMD:DONE a=1 b=2' -> {'a': '1', 'b': '2'}"""
    return dict(kv.split("=", 1) for kv in (reply or "").split()[1:] if "=" in kv)


class SensorLink:
    """One conversation with one ESP32, on its own socket."""

    def __init__(self, ip: str, port: int = COMMAND_PORT, timeout: float = 15.0):
        self.target = (ip, port)
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 0))                   # replies come back to this socket

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def image_template(self, image) -> Optional[bytes]:
        """The R503's template of an 8-bit grayscale image, or None."""
        if not upload(self.sock, self.target, "I", pack_image(image)):
            return None
        reply, blobs = run_command(self.sock, self.target, "IMG2TMPL FAST", self.timeout)
        if not reply or not reply.startswith("IMG2TMPL:DONE"):
            return None
        return blobs.get("T")

    def prepare_confirm(self, stamp: int) -> Tuple[bool, str]:
        """Have the sensor rebuild the template of the press it sent with this stamp.

        (ok, reply). Not ok when the ESP32 has captured another press since,
        has no copy of the image, or does not answer."""
        reply, _ = run_command(self.sock, self.target, f"CONFIRM {int(stamp)}", self.timeout)
        return bool(reply and reply.startswith("CONFIRM:DONE")), reply or "no answer"

    def match(self, template: bytes) -> Optional[Tuple[bool, int]]:
        """The R503's Match of a stored template against the prepared press: (matched, score)."""
        if not upload(self.sock, self.target, "B", template):
            return None
        reply, _ = run_command(self.sock, self.target, "MATCHTMPL", self.timeout)
        if not reply or not reply.startswith("MATCHTMPL:DONE"):
            return None
        f = fields(reply)
        return int(f["match"]) == 0, int(f["score"])
