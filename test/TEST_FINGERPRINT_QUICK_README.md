# Fingerprint Quick Test - UDP Receiver

Standalone fingerprint test script with enrollment and identification capabilities.

## Features

- ✅ **UDP Receiver**: Receives fingerprint images from ESP32 via UDP (port 8889)
- ✅ **Local Storage**: Stores fingerprints locally (no database required)
- ✅ **Enrollment**: Enroll new fingerprints with multiple samples
- ✅ **Identification**: Identify fingerprints against enrolled database
- ✅ **Management**: List and delete enrolled fingerprints

## Usage

### Start the Script
```bash
python test_fingerprint_quick.py
```

### Commands

| Command | Description |
|---------|-------------|
| `e` | Enroll new fingerprint |
| `i` | Identify fingerprint |
| `l` | List all enrolled fingerprints |
| `d` | Delete a fingerprint |
| `q` | Quit |

## Example Workflow

### 1. Enroll a Fingerprint
```
Enter command (e/i/l/d/q): e
Enter person name/ID: John Doe
Number of samples (default 5): 5

🖐️  ENROLLMENT: John Doe
Please scan 5 fingerprint samples...

📌 Sample 1/5
   Place finger on sensor...
⏳ Waiting for fingerprint image...
📥 Header received from 192.168.1.120:8889
   Dimensions: 64x80
   Expected size: 5120 bytes
✓ End marker received
✅ Image received successfully!
🔍 Extracting features...
✅ Sample 1 captured (245 features)
   Remove and place finger again...

... (repeat for 5 samples)

✅ Enrollment complete!
   Name: John Doe
   Samples: 5
```

### 2. Identify a Fingerprint
```
Enter command (e/i/l/d/q): i

🔍 IDENTIFICATION
Please scan fingerprint...

⏳ Waiting for fingerprint image...
📥 Header received from 192.168.1.120:8889
✓ End marker received
✅ Image received successfully!
🔍 Extracting features...
✅ Extracted 238 features

🔎 Checking against 3 enrolled persons...

⏱️  Matching took 45.23ms
✅ MATCH FOUND!
   Name: John Doe
   Confidence: 87.3%
   Threshold: 70.0%
```

### 3. List Enrolled Fingerprints
```
Enter command (e/i/l/d/q): l

📋 ENROLLED FINGERPRINTS
============================================================
1. John Doe
   Samples: 5
   Enrolled: 2024-01-15T10:30:45

2. Jane Smith
   Samples: 5
   Enrolled: 2024-01-15T11:20:12

Total: 2 persons
```

## Storage

All data is stored locally in the `fingerprint_storage/` directory:

```
fingerprint_storage/
├── fingerprints.db          # Database file (pickle format)
└── images/                  # Saved fingerprint images
    ├── John Doe_sample_1.png
    ├── John Doe_sample_2.png
    ├── query_20240115_103045.png
    └── ...
```

## Configuration

Edit the script to change:

- **UDP Port**: `UDP_PORT = 8889`
- **Match Threshold**: 
  - Improved processor: `MATCH_THRESHOLD = 0.7`
  - Old processor: `MATCH_THRESHOLD_OLD = 0.12`
- **Storage Directory**: `STORAGE_DIR = "fingerprint_storage"`

## Requirements

- Python 3.6+
- numpy
- opencv-python
- (Optional) faiss-cpu (for improved processor)

## Network Setup

Make sure:
1. ESP32 is sending to the correct IP and port
2. Firewall allows UDP port 8889
3. PC and ESP32 are on the same network

## Troubleshooting

### "No data received" / Timeout
- Check ESP32 is sending data
- Verify network connection
- Check UDP port (default 8889)
- Ensure firewall allows UDP traffic

### "Failed to extract features"
- Image quality may be too poor
- Try cleaning the sensor
- Ensure proper finger placement

### "NO MATCH" even with enrolled fingerprint
- Try adjusting threshold (lower = more matches)
- Re-enroll with better quality samples
- Check that multiple samples were enrolled

### Database not saving
- Check write permissions in `fingerprint_storage/` directory
- Ensure disk space is available

## Notes

- **First run**: Creates `fingerprint_storage/` directory
- **Images**: All received images are saved automatically
- **Database**: Stored as pickle file (not human-readable)
- **Samples**: Recommended 5 samples per person for best accuracy
- **Processor**: Automatically uses improved processor if available, falls back to old

## Performance

- **Enrollment**: ~2-3 seconds per sample
- **Identification**: 
  - Old processor: ~50-200ms per person
  - Improved processor: ~10-50ms per person
- **Storage**: ~50-100KB per enrolled person

