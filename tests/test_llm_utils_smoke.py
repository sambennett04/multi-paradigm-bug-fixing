"""Smoke tests for the retrieval portions of llm_utils."""

import tempfile
import unittest
import warnings

from src.llm_utils import (
    LLMConfig,
    _retrieve_samples,
    build_FAISS_index,
    load_knowledge_base,
    save_knowledge_base,
    train_embedding_model_and_generate_embeddings,
)


class TestLLMUtilsSmoke(unittest.TestCase):
    """Small end-to-end smoke tests for the embedding and retrieval pipeline."""

    def test_embedding_index_save_load_and_retrieve(self) -> None:
        raw_corpus = [
            [
                "public int add(int a, int b) { return a - b; }",
                "public int add(int a, int b) { return a + b; }",
            ],
            [
                "public int multiply(int a, int b) { return a + b; }",
                "public int multiply(int a, int b) { return a * b; }",
            ],
            [
                "this is not valid java {{{",
                "public void noop() {}",
            ],
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            embedder, embeddings, metadata = train_embedding_model_and_generate_embeddings(raw_corpus)

        self.assertGreaterEqual(len(caught), 1)
        self.assertEqual(embeddings.shape[0], len(metadata))
        self.assertEqual(embeddings.dtype.name, "float32")

        index = build_FAISS_index(embeddings)
        self.assertEqual(index.ntotal, len(metadata))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_knowledge_base(embedder, index, metadata, tmpdir)
            loaded_embedder, loaded_index, loaded_metadata, config = load_knowledge_base(tmpdir)

        self.assertEqual(config["num_examples"], len(loaded_metadata))
        self.assertEqual(loaded_index.ntotal, len(loaded_metadata))

        llm_config = LLMConfig(
            embedding_model=loaded_embedder,
            faiss_index=loaded_index,
            metadata=loaded_metadata,
            tokenizer=None,
            model=None,
            top_k=1,
        )

        results = _retrieve_samples(
            "public int add(int a, int b) { return a - b; }",
            llm_config,
        )

        self.assertLessEqual(len(results), 1)
        self.assertTrue(results)
        self.assertIn("buggy", results[0])
        self.assertIn("fixed", results[0])
        self.assertIn("similarity", results[0])


if __name__ == "__main__":
    unittest.main()
