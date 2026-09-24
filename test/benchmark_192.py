#!/usr/bin/env python3
"""
Benchmark identification on 192x192 samples using ImprovedFingerprintProcessor.

Usage: python test/benchmark_192.py
"""
import os
import time
from collections import defaultdict
import cv2

from fingerprint_processor_improved import ImprovedFingerprintProcessor, FAISS_AVAILABLE


IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fingerprint_storage', 'images')


def find_samples():
    samples = defaultdict(list)
    for fn in sorted(os.listdir(IMAGES_DIR)):
        if '_sample_' in fn and fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            client = fn.split('_sample_')[0]
            samples[client].append(os.path.join(IMAGES_DIR, fn))
    return samples


def build_profiles(processor, samples_by_client, holdout=True):
    profiles = []
    queries = []

    for client, files in samples_by_client.items():
        if len(files) < 2:
            continue
        # use last sample as query
        enroll_files = files[:-1]
        query_file = files[-1]

        sample_features = []
        for fpath in enroll_files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            feat = processor.extract_features(img)
            if feat is None:
                continue
            sample_features.append(feat)

        if not sample_features:
            continue

        profiles.append({'client_id': client, 'fingerprints': [{'descriptors': s['descriptors']} for s in sample_features]})
        queries.append((client, query_file))

    return profiles, queries


def brute_force_identify(processor, query_img, profiles):
    qfeat = processor.extract_features(query_img)
    if qfeat is None:
        return None, 0.0

    best_client = None
    best_score = 0.0

    # detailed check across all samples
    for profile in profiles:
        client_id = profile['client_id']
        samples = profile.get('fingerprints', [])
        scores = []
        for s in samples:
            enrolled = {'descriptors': s.get('descriptors')}
            score = processor.match_fingerprints_detailed(qfeat, enrolled)
            scores.append(score)
        if scores:
            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                best_client = client_id

    return best_client, best_score


def main():
    print("→ FAISS available:", FAISS_AVAILABLE)

    processor = ImprovedFingerprintProcessor()

    samples = find_samples()
    print(f"Found {len(samples)} clients with samples")

    profiles, queries = build_profiles(processor, samples)
    print(f"Built {len(profiles)} enrolled profiles, {len(queries)} queries")

    if len(profiles) == 0 or len(queries) == 0:
        print("Not enough data to run benchmark (need >=2 samples per client)")
        return

    # Build FAISS index if available
    if FAISS_AVAILABLE:
        print("Building FAISS index...")
        processor.build_faiss_index(profiles)

    # Run queries
    total = 0
    correct = 0
    times = []

    for true_client, qpath in queries:
        img = cv2.imread(qpath, cv2.IMREAD_GRAYSCALE)
        t0 = time.time()
        if FAISS_AVAILABLE:
            res = processor.identify_fingerprint_fast(img, profiles, threshold=0.7, top_k_candidates=5)
            if res:
                pred_client, _, conf = res
            else:
                pred_client, conf = None, 0.0
        else:
            pred_client, conf = brute_force_identify(processor, img, profiles)
        dt = time.time() - t0
        times.append(dt)
        total += 1
        ok = (pred_client == true_client)
        if ok:
            correct += 1
        print(f"Query {os.path.basename(qpath)} -> pred={pred_client} true={true_client} conf={conf:.3f} t={dt:.3f}s {'OK' if ok else 'FAIL'}")

    print("\n=== Summary ===")
    print(f"Queries: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Avg time/query: {sum(times)/len(times):.3f}s")


if __name__ == '__main__':
    main()
