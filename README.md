PC = 192.168.1.3
pc_port = 4211
orangepi = 192.168.1.100
espcam_locker_ip = 192.168.1.110
espcam_locke_port = 4210
esp32_1_ip = "192.168.1.201"
esp32_1_port = 4210
esp32_2_ip = "192.168.1.202"
esp32_2_port = 4210
espcam_ip = "192.168.1.111"
espcam_port = 4210
finger_enter_esp = 192, 168, 1, 120
UDP_PORT_ENTRANCE = 8889  # Port for entrance ESP32 
finger_locker_esp192, 168, 1, 121
UDP_PORT_LOCKER = 8890    # Port for locker ESP32 


gst-launch-1.0 rtspsrc location=rtsp://192.168.1.110:8554/live latency=100 ! rtph264depay ! h264parse ! mppvideodec ! autovideosink

On PC :
# Run PowerShell as Administrator
New-NetFirewallRule -DisplayName "PostgreSQL" -Direction Inbound -LocalPort 5432 -Protocol TCP -Action Allow

# Run as Administrator
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Open firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22

# GymController
Orange pi face recognition and locker room controller
[Unit]
Description=RetinaFace Detection Service
After=network.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/Projects/GymController
ExecStart=/usr/bin/python3 /home/orangepi/Projects/GymController/RetinaFaceL.py --model_path model/retinafacefp.rknn --db_host 192.168.1.3
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target


 sudo sshfs 'Amin@192.168.1.104:E:\ASAgym\ClientImages' /mnt/winshare -o allow_other
'Amin@192.168.1.104:E:\ASAgym\ClientImages' /mnt/winshare fuse.sshfs delay_connect,_netdev,user,idmap=user,transform_symlinks,identityfile=/home/orangepi/.ssh/id_rsa,allow_other,default_permissions,uid=1000,gid=1000 0 0


 ///////////////////////////////////

Alternative: Use SSHFS with Password File
Since SSH key setup on Windows can be tricky, here's a more reliable method using a password file:
Step 1: Create password file
bash nano ~/.sshfs_pass
```

Add your password:
```
your_password_here
Secure it:
bash chmod 600 ~/.sshfs_pass
Step 2: Install sshpass
bashsudo apt-get install sshpass
Step 3: Create mount script
bash  nano ~/mount_sshfs.sh
Add:
bash#!/bin/bash

# Wait for network
sleep 15

# Create mount point
sudo mkdir -p /mnt/winshare

# Get password
PASS=$(cat /home/orangepi/.sshfs_pass)

# Mount using sshpass
echo "$PASS" | sudo sshfs -o password_stdin,allow_other,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,StrictHostKeyChecking=no Amin@192.168.1.104:'E:\ASAgym\ClientImages' /mnt/winshare

# Log result
if mountpoint -q /mnt/winshare; then
    echo "$(date): SSHFS mounted successfully" >> /home/orangepi/mount.log
else
    echo "$(date): SSHFS mount failed" >> /home/orangepi/mount.log
fi
Make executable:
bash  chmod +x ~/mount_sshfs.sh
Step 4: Test the script
bash ~/mount_sshfs.sh
ls /mnt/winshare
Step 5: Create systemd service
bash  sudo nano /etc/systemd/system/sshfs-mount.service
Add:
ini[Unit]
Description=Mount Windows Share via SSHFS
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/home/orangepi/mount_sshfs.sh
RemainAfterExit=yes
User=root

[Install]
WantedBy=multi-user.target

Step 6: Enable the service
bash  sudo systemctl daemon-reload
sudo systemctl enable sshfs-mount.service
sudo systemctl start sshfs-mount.service


Step 7: Update RetinaFace service

bash sudo nano /etc/systemd/system/retinaface.service
Update to:
ini[Unit]
Description=RetinaFace Detection Service
After=network-online.target sshfs-mount.service
Requires=sshfs-mount.service
Wants=network-online.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/your_project_directory
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/orangepi/.Xauthority"
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/python3 /home/orangepi/your_project_directory/RetinaFaceL.py --model_path /path/to/your/model.rknn --target rk3566
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target


Step 8: Enable RetinaFace service

bash sudo systemctl daemon-reload
sudo systemctl enable retinaface.service

Test Everything
bash# Check mount service
sudo systemctl status sshfs-mount.service

# Check if mounted
df -h | grep winshare
ls /mnt/winshare

# Check RetinaFace service
sudo systemctl status retinaface.service

# View logs
sudo journalctl -u sshfs-mount.service -f
sudo journalctl -u retinaface.service -f

# Reboot to test
sudo reboot