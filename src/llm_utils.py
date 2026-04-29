"""Utilities for retrieval-augmented generation and zero-shot qwen workflows."""

from __future__ import annotations

import json
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import torch
from gensim.models import Word2Vec
from transformers import AutoModelForCausalLM, AutoTokenizer

from tree_sitter import Language, Parser
import tree_sitter_java as tsjava


def _build_java_parser() -> Parser:
    """Create a Java parser instance for AST extraction."""
    java_language = Language(tsjava.language())
    parser = Parser()
    parser.language = java_language
    return parser


def _warn_on_skipped_record(index: int, reason: str) -> None:
    """Emit a warning for a skipped training or retrieval sample."""
    warnings.warn(
        f"Skipping record {index}: {reason}",
        category=RuntimeWarning,
        stacklevel=2,
    )


def _extract_ast_nodes(code: str) -> list[str] | None:
    """Parse Java code into an AST node-type sequence."""
    return _extract_ast_nodes_with_parser(code, _build_java_parser())


def _extract_ast_nodes_with_parser(code: str, java_parser: Parser) -> list[str] | None:
    """Parse Java code into an AST node-type sequence with a provided parser."""
    try:
        tree = java_parser.parse(code.encode("utf-8"))
        root = tree.root_node
    except Exception:
        return None

    if root is None:
        return None

    has_error = getattr(root, "has_error", False)
    if has_error:
        return None

    nodes: list[str] = []

    def dfs(node: Any) -> None:
        nodes.append(node.type)
        for child in node.children:
            dfs(child)

    dfs(root)
    return nodes


def _prepare_training_records(raw_corpus: list[list[str]]) -> tuple[list[dict[str, Any]], int]:
    """Normalize raw buggy/fixed pairs and pre-parse buggy code ASTs."""
    prepared_records: list[dict[str, Any]] = []
    skipped_count = 0

    for index, pair in enumerate(raw_corpus):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            _warn_on_skipped_record(index, "expected [buggy, fixed] pair")
            skipped_count += 1
            continue

        buggy_code, fixed_code = pair
        ast_nodes = _extract_ast_nodes(str(buggy_code))
        if ast_nodes is None:
            _warn_on_skipped_record(index, "buggy method could not be parsed as Java")
            skipped_count += 1
            continue

        prepared_records.append(
            {
                "buggy": str(buggy_code),
                "fixed": str(fixed_code),
                "language": "java",
                "ast_nodes": ast_nodes,
            }
        )

    return prepared_records, skipped_count


class Code2VecEmbedder:
    """Structure-aware embeddings using AST node sequences + skip-gram."""

    def __init__(self, vector_size: int = 128, window: int = 5, min_count: int = 2, epochs: int = 10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model: Word2Vec | None = None
        self.num_skipped = 0
        self.java_parser = _build_java_parser()

    def _mean_pool_ast_nodes(self, ast_nodes: list[str]) -> np.ndarray | None:
        """Convert AST nodes into one pooled embedding vector."""
        if self.model is None:
            raise ValueError("Embedding model has not been trained or loaded.")

        vectors = [self.model.wv[node] for node in ast_nodes if node in self.model.wv]
        if not vectors:
            return None

        return np.mean(vectors, axis=0, dtype=np.float32)

    def train(self, records: list[dict[str, Any]]) -> None:
        """Train Word2Vec on prevalidated AST node sequences."""
        ast_sequences = [record["ast_nodes"] for record in records if record.get("ast_nodes")]
        if not ast_sequences:
            raise ValueError("No parseable training records were available to train the embedding model.")

        self.model = Word2Vec(
            sentences=ast_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=1,
            epochs=self.epochs,
        )

    def encode(self, records: list[dict[str, Any]]) -> np.ndarray:
        """Mean-pool AST node vectors into one embedding per record."""
        embeddings: list[np.ndarray] = []
        for index, record in enumerate(records):
            ast_nodes = record.get("ast_nodes")
            if not ast_nodes:
                _warn_on_skipped_record(index, "record did not contain precomputed AST nodes")
                continue

            pooled = self._mean_pool_ast_nodes(ast_nodes)
            if pooled is None:
                _warn_on_skipped_record(index, "buggy method produced no in-vocabulary AST nodes")
                continue

            embeddings.append(pooled)

        if not embeddings:
            return np.empty((0, self.vector_size), dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray | None:
        """Embed one buggy method for retrieval."""
        if self.model is None:
            raise ValueError("Embedding model has not been trained or loaded.")

        ast_nodes = _extract_ast_nodes_with_parser(query, self.java_parser)
        if ast_nodes is None:
            _warn_on_skipped_record(-1, "query buggy method could not be parsed as Java")
            return None

        pooled = self._mean_pool_ast_nodes(ast_nodes)
        if pooled is None:
            _warn_on_skipped_record(-1, "query buggy method produced no in-vocabulary AST nodes")
            return None

        return np.array([pooled], dtype=np.float32)


def train_embedding_model_and_generate_embeddings(
    raw_corpus: list[list[str]],
) -> tuple[Code2VecEmbedder, np.ndarray, list[dict[str, str]]]:
    """Train a Code2Vec-style embedder and encode parseable buggy methods."""
    prepared_records, skipped_count = _prepare_training_records(raw_corpus)
    if not prepared_records:
        raise ValueError("No parseable buggy methods were found in the raw corpus.")

    embedder = Code2VecEmbedder()
    embedder.num_skipped = skipped_count
    embedder.train(prepared_records)

    retained_records: list[dict[str, Any]] = []
    embeddings_list: list[np.ndarray] = []
    for index, record in enumerate(prepared_records):
        pooled = embedder._mean_pool_ast_nodes(record["ast_nodes"])
        if pooled is None:
            _warn_on_skipped_record(index, "buggy method produced no in-vocabulary AST nodes after training")
            embedder.num_skipped += 1
            continue

        embeddings_list.append(pooled)
        retained_records.append(record)

    if not embeddings_list:
        raise ValueError("No embeddings could be generated from the parseable buggy methods.")

    embeddings = np.array(embeddings_list, dtype=np.float32)
    metadata = [
        {
            "buggy": record["buggy"],
            "fixed": record["fixed"],
            "language": record["language"],
        }
        for record in retained_records
    ]
    return embedder, embeddings, metadata


def build_FAISS_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build an exact L2 FAISS index from 2D float32 embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty.")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_knowledge_base(
    embedding_model: Code2VecEmbedder,
    index: faiss.Index,
    metadata: list[dict[str, str]],
    output_dir: str | Path,
) -> None:
    """Persist the embedding model, FAISS index, metadata, and config."""
    if embedding_model.model is None:
        raise ValueError("Embedding model has not been trained or loaded.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_path / "faiss.index"))
    embedding_model.model.save(str(output_path / "code2vec.model"))

    with (output_path / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)

    config = {
        "embedding_method": "code2vec",
        "vector_size": embedding_model.vector_size,
        "dimension": index.d,
        "num_examples": len(metadata),
        "num_skipped": getattr(embedding_model, "num_skipped", 0),
        "language": "java",
    }
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_knowledge_base(
    knowledge_base_dir: str | Path,
) -> tuple[Code2VecEmbedder, faiss.Index, list[dict[str, str]], dict[str, Any]]:
    """Reload the saved embedding model, FAISS index, metadata, and config."""
    kb_path = Path(knowledge_base_dir)

    with (kb_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    index = faiss.read_index(str(kb_path / "faiss.index"))

    with (kb_path / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)

    embedder = Code2VecEmbedder(vector_size=config["vector_size"])
    embedder.model = Word2Vec.load(str(kb_path / "code2vec.model"))
    embedder.num_skipped = config.get("num_skipped", 0)

    return embedder, index, metadata, config


@dataclass
class LLMConfig:
    """Configuration and loaded assets for zero-shot and RAG generation."""

    embedding_model: Code2VecEmbedder
    faiss_index: faiss.Index
    metadata: list[dict[str, str]]
    tokenizer: Any
    model: Any
    num_samples: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 512
    top_k: int = 3
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _retrieve_samples(query: str, llm_config: LLMConfig) -> list[dict[str, Any]]:
    """Retrieve similar buggy/fixed training pairs from the knowledge base."""
    query_embedding = llm_config.embedding_model.encode_query(query)
    if query_embedding is None:
        return []

    distances, indices = llm_config.faiss_index.search(query_embedding, llm_config.top_k)

    results: list[dict[str, Any]] = []
    for distance, index in zip(distances[0], indices[0]):
        if index < 0 or index >= len(llm_config.metadata):
            continue

        item = llm_config.metadata[index]
        results.append(
            {
                "buggy": item["buggy"],
                "fixed": item["fixed"],
                "language": item.get("language", "java"),
                "similarity": 1 / (1 + float(distance)),
            }
        )

    return results


def _build_rag_context(retrieved_examples: list[dict[str, Any]]) -> str:
    """Format retrieved repair examples for prompt augmentation."""
    if not retrieved_examples:
        return ""

    parts = ["Here are similar Java bug-fixing examples for reference:\n"]
    for index, example in enumerate(retrieved_examples, start=1):
        parts.append(f"--- Example {index} ---")
        parts.append(f"Buggy method:\n```java\n{example['buggy']}\n```")
        parts.append(f"Fixed method:\n```java\n{example['fixed']}\n```")
    parts.append("Now fix the following buggy Java method:\n")
    return "\n".join(parts)


def _build_prompt(user_task: str, tokenizer: Any, rag_context: str = "") -> str:
    """Build a chat prompt for Qwen code generation."""
    system_content = (
        "You are an expert Java developer. Fix the buggy Java method and return only the corrected Java code."
    )
    if rag_context:
        system_content += " Use the provided repair examples as reference patterns."

    user_content = (
        f"{rag_context}\nBuggy Java method:\n```java\n{user_task}\n```"
        if rag_context
        else f"Buggy Java method:\n```java\n{user_task}\n```"
    )

    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return f"{system_content}\n\n{user_content}"


def _extract_code(text: str) -> str:
    """Extract code from markdown-wrapped model output."""
    pattern = r"```(?:python|java)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    if text.startswith("```java"):
        return text.replace("```java", "", 1).strip()
    if text.startswith("```python"):
        return text.replace("```python", "", 1).strip()
    if text.startswith("```"):
        return text.replace("```", "", 1).strip()

    return text.strip()


def _get_model_device(model: Any) -> torch.device:
    """Resolve a usable device for token tensors."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def load_qwen_generator(model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> tuple[Any, Any]:
    """Load the Qwen tokenizer and model with notebook-friendly defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return tokenizer, model


def generate_code(user_task: str, llm_config: LLMConfig, mode: Literal["zero_shot", "RAG"]) -> list[str]:
    """Generate repaired Java code with zero-shot or retrieval-augmented prompting."""
    rag_context = ""
    if mode == "RAG":
        rag_context = _build_rag_context(_retrieve_samples(user_task, llm_config))
    elif mode != "zero_shot":
        raise ValueError("mode must be either 'zero_shot' or 'RAG'")

    prompt = _build_prompt(user_task, llm_config.tokenizer, rag_context=rag_context)
    device = _get_model_device(llm_config.model)
    inputs = llm_config.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)
    input_length = inputs["input_ids"].shape[1]

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": llm_config.max_new_tokens,
        "do_sample": llm_config.temperature > 0.0,
        "pad_token_id": llm_config.tokenizer.pad_token_id,
        "eos_token_id": llm_config.tokenizer.eos_token_id,
    }
    if llm_config.temperature > 0.0:
        generation_kwargs["temperature"] = llm_config.temperature

    generations: list[str] = []
    for _ in range(llm_config.num_samples):
        with torch.no_grad():
            outputs = llm_config.model.generate(**generation_kwargs)

        generated_tokens = outputs[0][input_length:]
        decoded = llm_config.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if "Human:" in decoded:
            decoded = decoded.split("Human:")[0].strip()
        generations.append(_extract_code(decoded))

    return generations


__all__ = [
    "Code2VecEmbedder",
    "LLMConfig",
    "build_FAISS_index",
    "generate_code",
    "load_knowledge_base",
    "load_qwen_generator",
    "save_knowledge_base",
    "train_embedding_model_and_generate_embeddings",
]
