# import socket

# esp32_ip = "192.168.1.110"   # Must match Serial Monitor IP
# esp32_port = 4210

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# for n in [11, 22, 33]:
#     msg = str(n)
#     sock.sendto(msg.encode("utf-8"), (esp32_ip, esp32_port))
#     print(f"Sent: {msg}")



# import socket

# UDP_IP = "0.0.0.0"       # listen on all interfaces
# UDP_PORT = 9999          # must match ESP32 udpServerPort
# ACK_MSG = b"OK"

# output_file = "fingerprint.bin"

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# print(f"Listening on {UDP_IP}:{UDP_PORT}")
# with open(output_file, "wb") as f:
#     while True:
#         data, addr = sock.recvfrom(2048)  # read up to 2 KB
#         if not data:
#             continue

#         # write raw fingerprint image data
#         f.write(data)
#         f.flush()

#         # send ACK back
#         sock.sendto(ACK_MSG, addr)
#         print(f"Received {len(data)} bytes, sent ACK")



import socket
import time

# Configuration
RASPBERRY_PI_IP = "10.183.120.18"  # Your Raspberry Pi IP
UDP_PORT = 4210

# Create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send test locker numbers
for locker in [15, 23, 42, 105]:
    message = str(locker).encode('utf-8')
    sock.sendto(message, (RASPBERRY_PI_IP, UDP_PORT))
    print(f"Sent locker #{locker}")
    time.sleep(3)

print("Test complete")