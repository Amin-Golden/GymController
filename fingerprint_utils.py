#!/usr/bin/env python3
"""
Utility functions for fingerprint processing
Can be used independently without processor instances
"""

import numpy as np


def aggregate_descriptors_simple(descriptors: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """
    Aggregate multiple descriptors into a single embedding (standalone function)
    
    This is a simplified version that doesn't require a processor instance.
    Used for database storage of embeddings.
    
    Args:
        descriptors: Array of descriptors (N × descriptor_dim)
        target_dim: Target embedding dimension (default: 128)
    
    Returns:
        Single embedding vector (target_dim,) as float32, or None if invalid
    """
    if descriptors is None or len(descriptors) == 0:
        return None
    
    try:
        # Average pooling (simple and effective)
        avg_embedding = np.mean(descriptors, axis=0)
        
        # Normalize to unit vector
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        else:
            return None
        
        # Resize to target dimension if needed
        if len(avg_embedding) != target_dim:
            if len(avg_embedding) > target_dim:
                # Truncate
                avg_embedding = avg_embedding[:target_dim]
            else:
                # Pad with zeros
                padding = np.zeros(target_dim - len(avg_embedding), dtype=avg_embedding.dtype)
                avg_embedding = np.concatenate([avg_embedding, padding])
        
        return avg_embedding.astype(np.float32)
    
    except Exception as e:
        print(f"⚠️  Error aggregating descriptors: {e}")
        return None

