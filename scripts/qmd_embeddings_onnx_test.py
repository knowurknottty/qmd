#!/usr/bin/env python3
"""
QMD Embeddings ONNX Mock Test

Tests the ONNX code path using a mock ONNX session to verify
the accelerated embedding pipeline works without real GPU deps.
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestONNXMockPath(unittest.TestCase):
    """Test ONNX code path with mocked ONNX Runtime."""

    def test_config_singular_key(self):
        """Test that singular 'embedding' config key is read correctly."""
        from qmd_embeddings import get_embedder

        config = {
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "prefer_onnx": False,
                "device": None,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            embedder = get_embedder(config_path)
            self.assertFalse(embedder.prefer_onnx)
            self.assertEqual(embedder.model_name, "all-MiniLM-L6-v2")
        finally:
            os.unlink(config_path)

    def test_config_plural_key_backward_compat(self):
        """Test that plural 'embeddings' key still works for backward compat."""
        from qmd_embeddings import get_embedder

        config = {
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "prefer_onnx": False,
                "device": "cpu",
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            embedder = get_embedder(config_path)
            self.assertFalse(embedder.prefer_onnx)
            self.assertEqual(embedder.device, "cpu")
        finally:
            os.unlink(config_path)

    def test_device_passed_to_sentence_transformers(self):
        """Test that device parameter is stored and accessible."""
        from qmd_embeddings import AcceleratedEmbedder

        embedder = AcceleratedEmbedder(
            model_name="all-MiniLM-L6-v2",
            prefer_onnx=False,
            device="cuda",
        )
        self.assertEqual(embedder.device, "cuda")

    def test_onnx_provider_selection_mock(self):
        """Test ONNX provider selection logic with mocked onnxruntime."""
        from qmd_embeddings import AcceleratedEmbedder

        embedder = AcceleratedEmbedder(
            model_name="all-MiniLM-L6-v2",
            prefer_onnx=True,
            device=None,
        )

        # Mock onnxruntime module
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Mock optimum and transformers with fake modules
        mock_optimum = MagicMock()
        mock_optimum_ort = MagicMock()
        mock_ort_model = MagicMock()
        mock_optimum_ort.ORTModelForFeatureExtraction.from_pretrained.return_value = mock_ort_model
        mock_optimum.onnxruntime = mock_optimum_ort

        mock_transformers = MagicMock()
        mock_tokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict(sys.modules, {
            'onnxruntime': mock_ort,
            'optimum': mock_optimum,
            'optimum.onnxruntime': mock_optimum_ort,
            'transformers': mock_transformers,
        }):
            backend = embedder._try_onnx()
            self.assertEqual(backend, "onnx-cuda")

    def test_onnx_fallback_to_cpu_provider(self):
        """Test that ONNX falls back to CPU provider when GPU not available."""
        from qmd_embeddings import AcceleratedEmbedder

        embedder = AcceleratedEmbedder(
            model_name="all-MiniLM-L6-v2",
            prefer_onnx=True,
            device=None,
        )

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        mock_optimum = MagicMock()
        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForFeatureExtraction.from_pretrained.return_value = MagicMock()
        mock_optimum.onnxruntime = mock_optimum_ort

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {
            'onnxruntime': mock_ort,
            'optimum': mock_optimum,
            'optimum.onnxruntime': mock_optimum_ort,
            'transformers': mock_transformers,
        }):
            backend = embedder._try_onnx()
            self.assertEqual(backend, "onnx-cpu")

    def test_empty_text_encoding(self):
        """Test that empty text list returns correct shape."""
        from qmd_embeddings import AcceleratedEmbedder

        embedder = AcceleratedEmbedder(prefer_onnx=False)
        embedder._backend = "sentence-transformers"
        embedder._model = MagicMock()
        embedder._model.encode.return_value = []

        import numpy as np
        result = embedder.encode([])
        self.assertEqual(result.shape, (0, 384))


class TestBenchmark(unittest.TestCase):
    """Test benchmark function."""

    def test_benchmark_returns_dict(self):
        """Test that benchmark returns expected keys."""
        from qmd_embeddings import benchmark_embedder, AcceleratedEmbedder

        embedder = AcceleratedEmbedder(prefer_onnx=False)
        embedder._backend = "sentence-transformers"
        embedder._model = MagicMock()

        import numpy as np
        embedder._model.encode.return_value = np.zeros((3, 384), dtype=np.float32)

        result = benchmark_embedder(embedder, ["test1", "test2", "test3"])
        self.assertIn("backend", result)
        self.assertIn("num_texts", result)
        self.assertEqual(result["num_texts"], 3)
        self.assertIn("total_seconds", result)
        self.assertIn("texts_per_second", result)
        self.assertIn("embedding_shape", result)


if __name__ == "__main__":
    unittest.main()
