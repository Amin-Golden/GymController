# Fingerprint Detection Test Scripts

Two test scripts are provided to test fingerprint detection separately from the main system.

## Quick Test (`test_fingerprint_quick.py`)

Simple test that doesn't require a database. Good for:
- Verifying processors are working
- Testing feature extraction
- Basic matching tests

### Usage:
```bash
python test_fingerprint_quick.py
```

### What it tests:
- ✅ Feature extraction (old and improved processors)
- ✅ Fingerprint matching
- ✅ Embedding generation
- ✅ Performance timing

---

## Full Test (`test_fingerprint_detection.py`)

Comprehensive test script with database support. Good for:
- Testing with real database
- Performance comparisons
- Identification testing
- Scalability testing

### Basic Usage:

#### 1. Test with Mock Data (No Database)
```bash
python test_fingerprint_detection.py --mock
```

#### 2. Test with Database
```bash
python test_fingerprint_detection.py \
    --db_host 192.168.1.3 \
    --db_name gym-db \
    --db_user postgres \
    --db_password 123456
```

#### 3. Test with Image File
```bash
python test_fingerprint_detection.py --image fingerprint.png
```

#### 4. Performance Comparison
```bash
python test_fingerprint_detection.py --mock --performance
```

#### 5. Test Identification Only
```bash
python test_fingerprint_detection.py \
    --db_host 192.168.1.3 \
    --db_name gym-db \
    --db_user postgres \
    --db_password 123456 \
    --test_identification
```

### Command Line Options:

| Option | Description | Example |
|--------|-------------|---------|
| `--db_host` | Database host IP | `--db_host 192.168.1.3` |
| `--db_name` | Database name | `--db_name gym-db` |
| `--db_user` | Database user | `--db_user postgres` |
| `--db_password` | Database password | `--db_password 123456` |
| `--db_port` | Database port | `--db_port 5432` |
| `--image` | Path to fingerprint image | `--image fp.png` |
| `--mock` | Use mock fingerprint data | `--mock` |
| `--test_extraction` | Test feature extraction only | `--test_extraction` |
| `--test_matching` | Test fingerprint matching | `--test_matching` |
| `--test_identification` | Test identification with DB | `--test_identification` |
| `--performance` | Run performance comparison | `--performance` |
| `--threshold` | Similarity threshold | `--threshold 0.7` |
| `--use_improved` | Use improved processor | `--use_improved` |

### Examples:

#### Example 1: Quick Feature Extraction Test
```bash
python test_fingerprint_detection.py --mock --test_extraction
```

#### Example 2: Full Test with Database
```bash
python test_fingerprint_detection.py \
    --db_host 192.168.1.3 \
    --db_name gym-db \
    --db_user postgres \
    --db_password 123456 \
    --test_extraction \
    --test_identification \
    --performance
```

#### Example 3: Test with Custom Threshold
```bash
python test_fingerprint_detection.py \
    --db_host 192.168.1.3 \
    --db_name gym-db \
    --db_user postgres \
    --db_password 123456 \
    --test_identification \
    --threshold 0.8
```

## Expected Output

### Quick Test Output:
```
============================================================
🧪 QUICK FINGERPRINT TEST
============================================================
✅ Old processor loaded
✅ Improved processor loaded

🎲 Generating test fingerprint...
✅ Test image created: (80, 64)

🔍 Testing OLD processor...
✅ Extracted 245 features in 15.23ms

🔍 Testing IMPROVED processor...
✅ Extracted 245 features in 14.87ms

🔗 Testing matching (OLD)...
✅ Self-match score: 100.0% (took 2.45ms)

🔗 Testing matching (IMPROVED)...
✅ Self-match score: 100.0% (took 0.89ms)

📦 Testing embedding generation...
✅ Generated embedding: shape=(128,), dtype=float32
   Embedding norm: 1.0000

============================================================
✅ Quick test completed!
============================================================
```

### Full Test Output (with database):
```
============================================================
🧪 FINGERPRINT DETECTION TEST
============================================================
✅ Database connected
✅ Old processor initialized
✅ Improved processor initialized

📷 Loading image: fingerprint.png
✅ Image loaded: (80, 64)

🔍 Testing old feature extraction...
✅ Extracted 245 features in 15.23ms

🔍 Testing improved feature extraction...
✅ Extracted 245 features in 14.87ms

🔍 Testing identification with database...
📊 Loaded 150 fingerprint profiles from database
🔨 Building FAISS index...
✓ FAISS index built in 2.34s
✅ MATCH FOUND in 23.45ms
   Client ID: 42
   Name: John Doe
   Confidence: 87.3%

============================================================
📊 PERFORMANCE COMPARISON
============================================================

🔹 Old Processor (SIFT/ORB):
   Feature extraction: 15.23ms ± 1.2ms

🔹 Improved Processor (FAISS):
   Feature extraction: 14.87ms ± 0.9ms

============================================================
✅ Test completed
```

## Troubleshooting

### "Database helper not available"
- Make sure `db_helper.py` is in the same directory
- Check that all dependencies are installed

### "Processor not available"
- Check that `fingerprint_gym.py` or `fingerprint_processor_improved.py` exists
- Verify imports are working

### "No fingerprints in database"
- Make sure you have enrolled some fingerprints
- Check database connection parameters

### "FAISS not available"
- Install FAISS: `pip install faiss-cpu`
- The test will still work but won't use FAISS indexing

## Requirements

- Python 3.6+
- numpy
- opencv-python
- (Optional) psycopg2 (for database tests)
- (Optional) faiss-cpu (for improved processor)

## Notes

- Mock fingerprints are generated with random patterns - they won't match real fingerprints
- For real testing, use actual fingerprint images or database
- Performance numbers may vary based on hardware
- The improved processor requires FAISS for full functionality

