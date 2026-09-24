#!/usr/bin/env python3
"""
Improved Fingerprint Processor with FAISS Indexing for Scalability
Addresses performance issues with large client databases
"""

import numpy as np
import cv2
import pickle
from typing import Optional, List, Tuple, Dict
import time

# Try to import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
    print("[OK] FAISS available - using fast similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS not available - install with: pip install faiss-cpu")
    print("   Falling back to optimized brute-force matching")

# Try to import scikit-learn for PCA (optional optimization)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available - PCA optimization disabled")


class ImprovedFingerprintProcessor:
    """
    Improved fingerprint processor with:
    1. FAISS indexing for fast similarity search
    2. Descriptor aggregation for reduced memory
    3. Two-stage matching (coarse + fine)
    """
    
    def __init__(self, embedding_dim=256, use_pca=True, pca_components=128, target_size=(192, 192)):
        """
        Initialize improved fingerprint processor
        
        Args:
            embedding_dim: Target embedding dimension (128 or 256 recommended)
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components (if use_pca=True)
        """
        # Initialize feature extractors (tuned for larger 192x192 images)
        # Increase ORB keypoints to capture more detail on bigger images
        self.orb = cv2.ORB_create(nfeatures=1500)
        
        try:
            self.sift = cv2.SIFT_create()
            self.use_sift = True
            print("[OK] Using SIFT for fingerprint feature extraction")
        except:
            self.use_sift = False
            print("[OK] Using ORB for fingerprint feature extraction")
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.client_id_map = []  # Maps FAISS index to client_id
        self.finger_index_map = []  # Maps FAISS index to finger_index
        self.sample_index_map = []  # Maps FAISS index to sample_number
        self.embedding_dim = embedding_dim
        self.use_pca = use_pca and SKLEARN_AVAILABLE
        self.pca_components = pca_components
        # Target size for input images (width, height)
        self.target_size = tuple(target_size)
        
        # PCA transformer (if enabled)
        self.pca = None
        self.pca_fitted = False
        
        # Cache for aggregated descriptors per client
        self.client_embeddings_cache = {}
        self.cache_valid = False
        
        print(f"[OK] Improved fingerprint processor initialized")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - FAISS available: {FAISS_AVAILABLE}")
        print(f"  - PCA enabled: {self.use_pca}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for small fingerprint images
        
        Args:
            image: Input fingerprint image (64×80)
        
        Returns:
            Preprocessed image
        """
        # Resize to the canonical target size for this processor
        try:
            target_w, target_h = self.target_size
            if (image.shape[1], image.shape[0]) != (target_w, target_h):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

        # Enhance contrast with CLAHE (tuned for larger images)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Denoise (adjusted for larger images)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=12, templateWindowSize=7, searchWindowSize=21)
        
        # Sharpen (helps with feature extraction)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Normalize
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def extract_features(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract features from fingerprint image
        
        Args:
            image: Input fingerprint image
        
        Returns:
            Feature dictionary with descriptors and metadata
        """
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Extract keypoints and descriptors
            if self.use_sift:
                keypoints, descriptors = self.sift.detectAndCompute(processed, None)
            else:
                keypoints, descriptors = self.orb.detectAndCompute(processed, None)
            
            if descriptors is None or len(descriptors) < 10:
                return None
            
            return {
                'descriptors': descriptors,
                'keypoints': [(kp.pt, kp.size, kp.angle) for kp in keypoints],
                'num_features': len(descriptors),
                'confidence': 1.0
            }
        except Exception as e:
            print(f"[ERROR] Feature extraction error: {e}")
            return None
    
    def aggregate_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Aggregate multiple descriptors into a single embedding
        
        Methods:
        1. Average pooling (simple, effective)
        2. PCA reduction (if enabled)
        3. Max pooling (alternative)
        
        Args:
            descriptors: Array of descriptors (N × descriptor_dim)
        
        Returns:
            Single embedding vector (embedding_dim,)
        """
        if descriptors is None or len(descriptors) == 0:
            return None

        # Improved pooling: combine mean and std statistics to increase discriminative power
        avg_embedding = np.mean(descriptors, axis=0)
        std_embedding = np.std(descriptors, axis=0)
        combined = np.concatenate([avg_embedding, std_embedding], axis=0)

        # Normalize combined vector
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        avg_embedding = combined
        
        # Apply PCA if enabled and fitted
        if self.use_pca and self.pca_fitted and self.pca is not None:
            try:
                avg_embedding = self.pca.transform(avg_embedding.reshape(1, -1))[0]
                # Normalize again after PCA
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
            except Exception:
                pass
        
        # Resize to target dimension if needed
        if len(avg_embedding) != self.embedding_dim:
            if len(avg_embedding) > self.embedding_dim:
                # Truncate
                avg_embedding = avg_embedding[:self.embedding_dim]
            else:
                # Pad with zeros
                padding = np.zeros(self.embedding_dim - len(avg_embedding))
                avg_embedding = np.concatenate([avg_embedding, padding])
        
        return avg_embedding.astype(np.float32)
    
    def build_faiss_index(self, fingerprint_profiles: List[Dict], fit_pca=True):
        """
        Build FAISS index from fingerprint profiles
        
        Args:
            fingerprint_profiles: List of fingerprint profiles from database
            fit_pca: Whether to fit PCA on the data
        """
        if not FAISS_AVAILABLE:
            print("[WARNING] FAISS not available, cannot build index")
            return False
        
        print(f"🔨 Building FAISS index from {len(fingerprint_profiles)} profiles...")
        start_time = time.time()
        
        # Collect all embeddings
        all_embeddings = []
        self.client_id_map = []
        self.finger_index_map = []
        self.sample_index_map = []
        
        # First pass: collect all descriptors for PCA fitting
        all_descriptors_for_pca = []
        
        for profile in fingerprint_profiles:
            client_id = profile['client_id']
            
            # Get all fingerprints (support both old and new structure)
            if 'fingers' in profile:
                all_samples = []
                for finger in profile['fingers']:
                    all_samples.extend(finger.get('samples', []))
            else:
                all_samples = profile.get('fingerprints', [])
            
            for sample in all_samples:
                if not isinstance(sample, dict) or 'descriptors' not in sample:
                    continue
                descriptors = sample.get('descriptors')
                if descriptors is None or len(descriptors) == 0:
                    continue
                
                all_descriptors_for_pca.append(descriptors)
        
        # Fit PCA if enabled
        if self.use_pca and fit_pca and len(all_descriptors_for_pca) > 0:
            print("  Fitting PCA...")
            # Flatten all descriptors
            flat_descriptors = np.vstack(all_descriptors_for_pca)
            self.pca = PCA(n_components=min(self.pca_components, flat_descriptors.shape[1]))
            self.pca.fit(flat_descriptors)
            self.pca_fitted = True
            print(f"  [OK] PCA fitted: {flat_descriptors.shape[1]} -> {self.pca_components} dimensions")
        
        # Second pass: create embeddings
        for profile in fingerprint_profiles:
            client_id = profile['client_id']
            
            # Get all fingerprints
            if 'fingers' in profile:
                for finger_idx, finger in enumerate(profile['fingers'], start=1):
                    samples = finger.get('samples', [])
                    for sample_idx, sample in enumerate(samples, start=1):
                        if not isinstance(sample, dict) or 'descriptors' not in sample:
                            continue
                        descriptors = sample.get('descriptors')
                        if descriptors is None or len(descriptors) == 0:
                            continue
                        
                        # Aggregate to single embedding
                        embedding = self.aggregate_descriptors(descriptors)
                        if embedding is not None:
                            all_embeddings.append(embedding)
                            self.client_id_map.append(client_id)
                            self.finger_index_map.append(finger_idx)
                            self.sample_index_map.append(sample_idx)
            else:
                # Old structure
                samples = profile.get('fingerprints', [])
                for sample_idx, sample in enumerate(samples, start=1):
                    if not isinstance(sample, dict) or 'descriptors' not in sample:
                        continue
                    descriptors = sample.get('descriptors')
                    if descriptors is None or len(descriptors) == 0:
                        continue
                    
                    embedding = self.aggregate_descriptors(descriptors)
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        self.client_id_map.append(client_id)
                        self.finger_index_map.append(1)  # Default to finger 1
                        self.sample_index_map.append(sample_idx)
        
        if len(all_embeddings) == 0:
            print("❌ No valid embeddings found")
            return False
        
        # Convert to numpy array
        embeddings_array = np.vstack(all_embeddings).astype('float32')
        print(f"  Created {len(all_embeddings)} embeddings, shape: {embeddings_array.shape}")
        
        # Create FAISS index
        # Use inner product (cosine similarity) since embeddings are normalized
        dimension = embeddings_array.shape[1]
        
        if len(all_embeddings) < 1000:
            # Small dataset: use flat index
            self.faiss_index = faiss.IndexFlatIP(dimension)
        else:
            # Large dataset: use inverted file index for faster search
            nlist = min(100, len(all_embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.nprobe = min(10, nlist)  # Number of clusters to search
        
        # An IVF index must be trained before anything can be added to it,
        # otherwise FAISS raises once the dataset crosses the 1000 embedding
        # mark (about 100 clients at 10 samples each).
        if isinstance(self.faiss_index, faiss.IndexIVFFlat):
            print("  Training FAISS index...")
            self.faiss_index.train(embeddings_array)
        
        # Add embeddings to index (exactly once)
        self.faiss_index.add(embeddings_array)
        
        elapsed = time.time() - start_time
        print(f"[OK] FAISS index built in {elapsed:.2f}s")
        print(f"  - Total embeddings: {len(all_embeddings)}")
        print(f"  - Index type: {type(self.faiss_index).__name__}")
        
        return True
    
    def search_similar(self, query_features: Dict, top_k=10) -> List[Tuple[int, float]]:
        """
        Search for similar fingerprints using FAISS
        
        Args:
            query_features: Feature dictionary from extract_features()
            top_k: Number of top matches to return
        
        Returns:
            List of (client_id, similarity_score) tuples
        """
        if self.faiss_index is None:
            print("[WARNING] FAISS index not built")
            return []
        
        if query_features is None or 'descriptors' not in query_features:
            return []
        
        descriptors = query_features['descriptors']
        if descriptors is None or len(descriptors) == 0:
            return []
        
        # Aggregate query descriptors to embedding
        query_embedding = self.aggregate_descriptors(descriptors)
        if query_embedding is None:
            return []
        
        # Reshape for FAISS (needs 2D array)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        start_time = time.time()
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        search_time = time.time() - start_time
        
        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.client_id_map):
                continue
            
            # FAISS returns inner product (higher is better for cosine similarity)
            # Convert to similarity score (0-1)
            similarity = float(dist)  # Already normalized, so this is cosine similarity
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            client_id = self.client_id_map[idx]
            results.append((client_id, similarity))
        
        return results
    
    def match_fingerprints_improved(self, features1: Dict, features2: Dict) -> float:
        """
        Improved matching with better normalization
        
        Args:
            features1: First fingerprint features
            features2: Second fingerprint features
        
        Returns:
            Similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        desc1 = features1.get('descriptors')
        desc2 = features2.get('descriptors')
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0.0
        
        # Aggregate both to embeddings
        emb1 = self.aggregate_descriptors(desc1)
        emb2 = self.aggregate_descriptors(desc2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity (both are normalized)
        similarity = np.dot(emb1, emb2)
        return float(max(0.0, min(1.0, similarity)))
    
    def match_fingerprints_detailed(self, features1: Dict, features2: Dict) -> float:
        """
        Detailed matching using original descriptors (for verification)
        
        Args:
            features1: First fingerprint features
            features2: Second fingerprint features
        
        Returns:
            Similarity score (0-1)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        desc1 = features1.get('descriptors')
        desc2 = features2.get('descriptors')
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0.0
        
        # Use BFMatcher
        if self.use_sift:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            # KNN matching
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Improved score calculation
            if len(desc1) > 0 and len(desc2) > 0:
                avg_descriptors = (len(desc1) + len(desc2)) / 2.0
                score = len(good_matches) / max(avg_descriptors, 1.0)
                
                # Boost for strong matches
                if len(good_matches) > 20:
                    score = score * 1.1
                
                return min(score, 1.0)
            return 0.0
                
        except Exception as e:
            print(f"[ERROR] Matching error: {e}")
            return 0.0
    
    def identify_fingerprint_fast(self, image: np.ndarray, fingerprint_profiles: List[Dict], 
                                   threshold=0.7, top_k_candidates=5) -> Optional[Tuple[int, str, float]]:
        """
        Fast fingerprint identification using FAISS
        
        Args:
            image: Input fingerprint image
            fingerprint_profiles: List of all fingerprint profiles
            threshold: Similarity threshold (0-1)
            top_k_candidates: Number of candidates to verify in detail
        
        Returns:
            Tuple of (client_id, name, confidence) or None
        """
        # Extract features
        features = self.extract_features(image)
        if features is None:
            return None
        
        # Rebuild index if needed (e.g., after new enrollments)
        if self.faiss_index is None or not self.cache_valid:
            self.build_faiss_index(fingerprint_profiles)
            self.cache_valid = True
        
        # Fast search using FAISS
        candidates = self.search_similar(features, top_k=top_k_candidates)
        
        if not candidates:
            return None
        
        # Detailed verification of top candidates
        best_match = None
        best_score = 0.0
        
        # Create lookup for profiles
        profile_dict = {p['client_id']: p for p in fingerprint_profiles}
        
        for client_id, coarse_score in candidates:
            if coarse_score < threshold * 0.8:  # Early termination
                continue
            
            profile = profile_dict.get(client_id)
            if not profile:
                continue
            
            # Get all samples for this client
            if 'fingers' in profile:
                all_samples = []
                for finger in profile['fingers']:
                    all_samples.extend(finger.get('samples', []))
            else:
                all_samples = profile.get('fingerprints', [])
            
            # Detailed matching with all samples
            scores = []
            for enrolled_fp in all_samples:
                if not isinstance(enrolled_fp, dict) or 'descriptors' not in enrolled_fp:
                    continue
                score = self.match_fingerprints_detailed(features, enrolled_fp)
                scores.append(score)
            
            if scores:
                scores.sort(reverse=True)
                avg_score = np.mean(scores[:5])  # Average of the 5 best samples
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = (client_id, profile.get('name', f'Client {client_id}'))
        
        # Check threshold
        if best_match and best_score >= threshold:
            client_id, name = best_match
            return (client_id, name, best_score)
        
        return None


# Example usage and migration guide
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Improved Fingerprint Processor                            ║
    ║  For use with fingerprint_gym.py                          ║
    ╚════════════════════════════════════════════════════════════╝
    
    This module provides:
    1. FAISS-based fast similarity search
    2. Descriptor aggregation for reduced memory
    3. Two-stage matching (coarse + fine)
    
    Installation:
    pip install faiss-cpu scikit-learn
    
    Integration:
    Replace FingerprintProcessor with ImprovedFingerprintProcessor
    in fingerprint_gym.py
    """)

