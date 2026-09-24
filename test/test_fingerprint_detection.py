#!/usr/bin/env python3
"""
Standalone test script for fingerprint detection/identification
Tests both old and improved fingerprint processors
"""

import argparse
import time
import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import database helper (optional)
try:
    from db_helper import DatabaseHelper
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️  Database helper not available - will use mock data")

# Import fingerprint processors
try:
    from fingerprint_gym import FingerprintProcessor
    OLD_PROCESSOR_AVAILABLE = True
except ImportError:
    OLD_PROCESSOR_AVAILABLE = False
    print("⚠️  Old fingerprint processor not available")

try:
    from fingerprint_processor_improved import ImprovedFingerprintProcessor
    IMPROVED_PROCESSOR_AVAILABLE = True
except ImportError:
    IMPROVED_PROCESSOR_AVAILABLE = False
    print("⚠️  Improved fingerprint processor not available")

try:
    from fingerprint_utils import aggregate_descriptors_simple
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️  Fingerprint utils not available")


class FingerprintTester:
    """Test class for fingerprint detection"""
    
    def __init__(self, db_helper=None, use_improved=True):
        """
        Initialize tester
        
        Args:
            db_helper: DatabaseHelper instance (optional)
            use_improved: Use improved processor if available
        """
        self.db = db_helper
        self.use_improved = use_improved
        
        # Initialize processors
        self.old_processor = None
        self.improved_processor = None
        
        if OLD_PROCESSOR_AVAILABLE:
            try:
                self.old_processor = FingerprintProcessor()
                print("✅ Old processor initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize old processor: {e}")
        
        if IMPROVED_PROCESSOR_AVAILABLE and use_improved:
            try:
                self.improved_processor = ImprovedFingerprintProcessor(
                    embedding_dim=128,
                    use_pca=True,
                    pca_components=64
                )
                print("✅ Improved processor initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize improved processor: {e}")
    
    def generate_mock_fingerprint(self, seed=None):
        """
        Generate a mock fingerprint image for testing
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            numpy array (64x80) representing fingerprint image
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create base pattern with ridges
        img = np.zeros((80, 64), dtype=np.uint8)
        
        # Add some ridge patterns
        for i in range(0, 80, 3):
            img[i:i+2, :] = 200 + np.random.randint(-30, 30, (2, 64))
        
        # Add some noise
        noise = np.random.randint(0, 50, (80, 64))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some circular patterns (simulating fingerprint)
        center = (32, 40)
        y, x = np.ogrid[:80, :64]
        mask = (x - center[0])**2 + (y - center[1])**2 < 400  # radius ~20
        img[mask] = np.clip(img[mask].astype(int) + 30, 0, 255).astype(np.uint8)
        
        return img
    
    def load_fingerprint_from_file(self, filepath):
        """
        Load fingerprint image from file
        
        Args:
            filepath: Path to image file
        
        Returns:
            numpy array or None
        """
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Could not load image: {filepath}")
                return None
            
            # Resize if needed
            if img.shape != (80, 64):
                img = cv2.resize(img, (64, 80))
                print(f"⚠️  Resized image to 64x80")
            
            return img
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def test_feature_extraction(self, image, processor_name="old"):
        """
        Test feature extraction
        
        Args:
            image: Fingerprint image
            processor_name: "old" or "improved"
        
        Returns:
            Features dict or None
        """
        processor = self.old_processor if processor_name == "old" else self.improved_processor
        
        if processor is None:
            print(f"❌ {processor_name} processor not available")
            return None
        
        print(f"\n🔍 Testing {processor_name} feature extraction...")
        start_time = time.time()
        
        features = processor.extract_features(image)
        
        elapsed = time.time() - start_time
        
        if features:
            num_features = features.get('num_features', 0)
            print(f"✅ Extracted {num_features} features in {elapsed*1000:.2f}ms")
            return features
        else:
            print(f"❌ Feature extraction failed")
            return None
    
    def test_matching(self, features1, features2, processor_name="old"):
        """
        Test fingerprint matching
        
        Args:
            features1: First fingerprint features
            features2: Second fingerprint features
            processor_name: "old" or "improved"
        
        Returns:
            Similarity score (0-1)
        """
        processor = self.old_processor if processor_name == "old" else self.improved_processor
        
        if processor is None:
            return 0.0
        
        if processor_name == "old":
            score = processor.match_fingerprints(features1, features2)
        else:
            score = processor.match_fingerprints_improved(features1, features2)
        
        return score
    
    def test_identification_with_db(self, image, threshold=0.7):
        """
        Test identification using database
        
        Args:
            image: Query fingerprint image
            threshold: Similarity threshold
        
        Returns:
            (client_id, name, confidence) or None
        """
        if self.db is None:
            print("❌ Database not available")
            return None
        
        if self.improved_processor is None:
            print("❌ Improved processor not available")
            return None
        
        print("\n🔍 Testing identification with database...")
        
        # Get all fingerprint profiles
        try:
            profiles = self.db.get_all_fingerprint_features()
            print(f"📊 Loaded {len(profiles)} fingerprint profiles from database")
        except Exception as e:
            print(f"❌ Error loading fingerprints: {e}")
            return None
        
        if len(profiles) == 0:
            print("⚠️  No fingerprints in database")
            return None
        
        # Build FAISS index if needed
        if self.improved_processor.faiss_index is None:
            print("🔨 Building FAISS index...")
            self.improved_processor.build_faiss_index(profiles)
        
        # Extract features
        features = self.improved_processor.extract_features(image)
        if features is None:
            print("❌ Could not extract features")
            return None
        
        # Fast identification
        start_time = time.time()
        result = self.improved_processor.identify_fingerprint_fast(
            image,
            profiles,
            threshold=threshold,
            top_k_candidates=5
        )
        elapsed = time.time() - start_time
        
        if result:
            client_id, name, confidence = result
            print(f"✅ MATCH FOUND in {elapsed*1000:.2f}ms")
            print(f"   Client ID: {client_id}")
            print(f"   Name: {name}")
            print(f"   Confidence: {confidence*100:.1f}%")
            return result
        else:
            print(f"❌ NO MATCH found in {elapsed*1000:.2f}ms")
            return None
    
    def test_identification_brute_force(self, image, threshold=0.12):
        """
        Test identification using brute-force matching (old method)
        
        Args:
            image: Query fingerprint image
            threshold: Similarity threshold
        
        Returns:
            (client_id, name, confidence) or None
        """
        if self.db is None:
            print("❌ Database not available")
            return None
        
        if self.old_processor is None:
            print("❌ Old processor not available")
            return None
        
        print("\n🔍 Testing brute-force identification...")
        
        # Get all fingerprint profiles
        try:
            profiles = self.db.get_all_fingerprint_features()
            print(f"📊 Loaded {len(profiles)} fingerprint profiles")
        except Exception as e:
            print(f"❌ Error loading fingerprints: {e}")
            return None
        
        if len(profiles) == 0:
            print("⚠️  No fingerprints in database")
            return None
        
        # Extract features
        features = self.old_processor.extract_features(image)
        if features is None:
            print("❌ Could not extract features")
            return None
        
        # Brute-force matching
        start_time = time.time()
        best_match = None
        best_score = 0.0
        
        for profile in profiles:
            client_id = profile['client_id']
            name = profile.get('name', f'Client {client_id}')
            
            # Get all samples
            if 'fingers' in profile:
                all_samples = []
                for finger in profile['fingers']:
                    all_samples.extend(finger.get('samples', []))
            else:
                all_samples = profile.get('fingerprints', [])
            
            # Compare with all samples
            scores = []
            for enrolled_fp in all_samples:
                if not isinstance(enrolled_fp, dict) or 'descriptors' not in enrolled_fp:
                    continue
                score = self.old_processor.match_fingerprints(features, enrolled_fp)
                scores.append(score)
            
            if scores:
                avg_score = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = (client_id, name)
        
        elapsed = time.time() - start_time
        
        if best_match and best_score >= threshold:
            client_id, name = best_match
            print(f"✅ MATCH FOUND in {elapsed*1000:.2f}ms")
            print(f"   Client ID: {client_id}")
            print(f"   Name: {name}")
            print(f"   Confidence: {best_score*100:.1f}%")
            return (client_id, name, best_score)
        else:
            print(f"❌ NO MATCH found in {elapsed*1000:.2f}ms")
            if best_match:
                print(f"   Best score: {best_score*100:.1f}% (threshold: {threshold*100:.1f}%)")
            return None
    
    def performance_comparison(self, image, num_iterations=10):
        """
        Compare performance between old and improved processors
        
        Args:
            image: Test fingerprint image
            num_iterations: Number of iterations to average
        """
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON")
        print("="*60)
        
        # Test old processor
        if self.old_processor:
            print("\n🔹 Old Processor (SIFT/ORB):")
            times = []
            for i in range(num_iterations):
                start = time.time()
                features = self.old_processor.extract_features(image)
                elapsed = (time.time() - start) * 1000
                if features:
                    times.append(elapsed)
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"   Feature extraction: {avg_time:.2f}ms ± {std_time:.2f}ms")
        
        # Test improved processor
        if self.improved_processor:
            print("\n🔹 Improved Processor (FAISS):")
            times = []
            for i in range(num_iterations):
                start = time.time()
                features = self.improved_processor.extract_features(image)
                elapsed = (time.time() - start) * 1000
                if features:
                    times.append(elapsed)
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"   Feature extraction: {avg_time:.2f}ms ± {std_time:.2f}ms")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Test fingerprint detection/identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with mock data (no database needed)
  python test_fingerprint_detection.py --mock

  # Test with database
  python test_fingerprint_detection.py --db_host 192.168.1.3 --db_name gym-db --db_user postgres --db_password 123456

  # Test with image file
  python test_fingerprint_detection.py --image fingerprint.png

  # Performance comparison
  python test_fingerprint_detection.py --mock --performance
        """
    )
    
    parser.add_argument('--db_host', type=str, help='Database host IP')
    parser.add_argument('--db_name', type=str, default='gym-db', help='Database name')
    parser.add_argument('--db_user', type=str, default='postgres', help='Database user')
    parser.add_argument('--db_password', type=str, help='Database password')
    parser.add_argument('--db_port', type=int, default=5432, help='Database port')
    
    parser.add_argument('--image', type=str, help='Path to fingerprint image file (64x80)')
    parser.add_argument('--mock', action='store_true', help='Use mock fingerprint data')
    
    parser.add_argument('--test_extraction', action='store_true', help='Test feature extraction only')
    parser.add_argument('--test_matching', action='store_true', help='Test fingerprint matching')
    parser.add_argument('--test_identification', action='store_true', help='Test identification with database')
    parser.add_argument('--performance', action='store_true', help='Run performance comparison')
    
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold (default: 0.7)')
    parser.add_argument('--use_improved', action='store_true', default=True, help='Use improved processor (default: True)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 FINGERPRINT DETECTION TEST")
    print("="*60)
    
    # Initialize database if provided
    db = None
    if args.db_host:
        if not DB_AVAILABLE:
            print("❌ Database helper not available")
            return
        
        try:
            db = DatabaseHelper(
                host=args.db_host,
                database=args.db_name,
                user=args.db_user,
                password=args.db_password,
                port=args.db_port
            )
            print("✅ Database connected")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return
    
    # Initialize tester
    tester = FingerprintTester(db_helper=db, use_improved=args.use_improved)
    
    # Get test image
    if args.image:
        print(f"\n📷 Loading image: {args.image}")
        image = tester.load_fingerprint_from_file(args.image)
        if image is None:
            return
    elif args.mock:
        print("\n🎲 Generating mock fingerprint...")
        image = tester.generate_mock_fingerprint(seed=42)
    else:
        print("❌ Please provide --image or --mock flag")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Run tests
    if args.test_extraction or not any([args.test_matching, args.test_identification, args.performance]):
        # Test feature extraction
        if tester.old_processor:
            features_old = tester.test_feature_extraction(image, "old")
        
        if tester.improved_processor:
            features_improved = tester.test_feature_extraction(image, "improved")
    
    if args.test_matching:
        # Test matching (need two images)
        print("\n🔗 Testing fingerprint matching...")
        image2 = tester.generate_mock_fingerprint(seed=43) if args.mock else image
        
        if tester.old_processor:
            features1 = tester.old_processor.extract_features(image)
            features2 = tester.old_processor.extract_features(image2)
            if features1 and features2:
                score = tester.test_matching(features1, features2, "old")
                print(f"   Match score (old): {score*100:.1f}%")
        
        if tester.improved_processor:
            features1 = tester.improved_processor.extract_features(image)
            features2 = tester.improved_processor.extract_features(image2)
            if features1 and features2:
                score = tester.test_matching(features1, features2, "improved")
                print(f"   Match score (improved): {score*100:.1f}%")
    
    if args.test_identification:
        if db is None:
            print("❌ Database required for identification test")
        else:
            # Test with improved processor
            if tester.improved_processor:
                tester.test_identification_with_db(image, threshold=args.threshold)
            
            # Test with old processor
            if tester.old_processor:
                tester.test_identification_brute_force(image, threshold=0.12)
    
    if args.performance:
        tester.performance_comparison(image)
    
    # Cleanup
    if db:
        db.close_all_connections()
    
    print("\n✅ Test completed")


if __name__ == '__main__':
    main()

