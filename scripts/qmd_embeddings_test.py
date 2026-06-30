#!/usr/bin/env python3
"""
Tests for QMD Accelerated Embeddings.

Run with: python qmd_embeddings_test.py
"""

import os
import sys
import tempfile
import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qmd_embeddings import AcceleratedEmbedder, benchmark_embedder


def test_embedder_initialization():
    """Test embedder can be initialized."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.backend is not None
    print(f"✅ Embedder initialized with backend: {embedder.backend}")


def test_encode_single():
    """Test encoding a single text."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)
    text = "This is a test sentence for embedding."
    embedding = embedder.encode_single(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.all(embedding == 0)

    # Check normalization
    norm = np.linalg.norm(embedding)
    assert 0.99 <= norm <= 1.01, f"Embedding not normalized: {norm}"
    print("✅ Single text encoding works")


def test_encode_batch():
    """Test encoding a batch of texts."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]
    embeddings = embedder.encode(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)

    # Check all embeddings are different
    for i in range(len(texts) - 1):
        similarity = np.dot(embeddings[i], embeddings[i + 1])
        assert 0.0 <= similarity <= 1.0

    print("✅ Batch encoding works")


def test_empty_input():
    """Test encoding empty input."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)
    embeddings = embedder.encode([])
    assert embeddings.shape == (0, 384)
    print("✅ Empty input handled correctly")


def test_semantic_similarity():
    """Test that similar texts have higher similarity."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)

    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed today.",
    ]

    embeddings = embedder.encode(texts)

    sim_cat_feline = np.dot(embeddings[0], embeddings[1])
    sim_cat_stock = np.dot(embeddings[0], embeddings[2])

    assert sim_cat_feline > sim_cat_stock, \
        f"Similar texts should be more similar: {sim_cat_feline} vs {sim_cat_stock}"
    print(f"✅ Semantic similarity works (cat-feline: {sim_cat_feline:.3f}, cat-stock: {sim_cat_stock:.3f})")


def test_benchmark():
    """Test benchmark function."""
    embedder = AcceleratedEmbedder(prefer_onnx=False)
    texts = ["Test sentence number " + str(i) for i in range(10)]
    result = benchmark_embedder(embedder, texts)

    assert "backend" in result
    assert "num_texts" in result
    assert "total_seconds" in result
    assert "texts_per_second" in result
    assert "embedding_shape" in result
    print(f"✅ Benchmark works: {result['texts_per_second']} texts/sec via {result['backend']}")


def test_config_loading():
    """Test embedder config loading."""
    from qmd_embeddings import get_embedder

    # Create temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"embeddings": {"model": "all-MiniLM-L6-v2", "prefer_onnx": false}}')
        config_path = f.name

    try:
        embedder = get_embedder(config_path)
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.prefer_onnx is False
        print("✅ Config loading works")
    finally:
        os.unlink(config_path)


def run_all_tests():
    """Run all tests."""
    print("=== QMD Accelerated Embeddings Tests ===\n")

    tests = [
        test_embedder_initialization,
        test_encode_single,
        test_encode_batch,
        test_empty_input,
        test_semantic_similarity,
        test_benchmark,
        test_config_loading,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
