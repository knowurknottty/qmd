#!/usr/bin/env python3
"""
QMD Embedding Benchmark

Measures CPU vs ONNX embedding speed for QMD's deep tier.
Run: python3 scripts/qmd_benchmark.py
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qmd_embeddings import AcceleratedEmbedder, benchmark_embedder


def run_benchmark():
    """Run CPU vs ONNX benchmark on sample texts."""
    # Sample texts — varied lengths and topics
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming software development",
        "Quantum computing uses qubits instead of bits",
        "The Riemann Hypothesis remains unsolved",
        "Post-quantum cryptography uses lattice-based algorithms",
        "Docker containers provide isolated execution environments",
        "Git is a distributed version control system",
        "Python is a high-level programming language",
        "Neural networks learn through backpropagation",
        "SQLite is a lightweight embedded database",
    ] * 20  # 200 texts total

    results = {}

    # Test 1: CPU (sentence-transformers)
    print("=" * 60)
    print("QMD Embedding Benchmark")
    print("=" * 60)
    print(f"\nSample size: {len(sample_texts)} texts")
    print()

    print("[1/2] Testing CPU backend (sentence-transformers)...")
    try:
        cpu_embedder = AcceleratedEmbedder(
            prefer_onnx=False,
            device="cpu",
        )
        cpu_result = benchmark_embedder(cpu_embedder, sample_texts)
        results["cpu"] = cpu_result
        print(f"  Backend: {cpu_result['backend']}")
        print(f"  Time: {cpu_result['total_seconds']}s")
        print(f"  Throughput: {cpu_result['texts_per_second']} texts/sec")
        print(f"  Shape: {cpu_result['embedding_shape']}")
    except Exception as e:
        print(f"  CPU backend failed: {e}")
        results["cpu"] = {"error": str(e)}

    print()

    # Test 2: ONNX (if available)
    print("[2/2] Testing ONNX backend (if available)...")
    try:
        onnx_embedder = AcceleratedEmbedder(
            prefer_onnx=True,
            device=None,
        )
        onnx_result = benchmark_embedder(onnx_embedder, sample_texts)
        results["onnx"] = onnx_result
        print(f"  Backend: {onnx_result['backend']}")
        print(f"  Time: {onnx_result['total_seconds']}s")
        print(f"  Throughput: {onnx_result['texts_per_second']} texts/sec")
        print(f"  Shape: {onnx_result['embedding_shape']}")

        # Speedup calculation
        if "cpu" in results and "total_seconds" in results.get("cpu", {}):
            speedup = results["cpu"]["total_seconds"] / onnx_result["total_seconds"]
            print(f"\n  Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"  ONNX backend not available: {e}")
        results["onnx"] = {"error": str(e)}

    print()
    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)

    # Save results
    results_path = os.path.expanduser("~/.hermes/memory/logs/qmd_benchmark.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
