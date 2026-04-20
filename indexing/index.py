"""
RAPTOR-style indexing pipeline for Pinecone handoff.

What this file does:
1. Recursively reads supported files from an input directory.
2. Extracts raw text from each document.
3. Splits each document into token-based leaf chunks.
4. Embeds the leaf chunks with a local embedding model.
5. Clusters chunk embeddings with UMAP + Gaussian Mixture Models.
6. Summarizes each cluster into a parent node with a local instruct model.
7. Re-embeds the summaries and repeats the cluster -> summarize -> embed loop
   until the tree reaches the top level or the max depth.
8. Exports one Pinecone-ready JSON file containing:
   - namespace
   - vectors
   - metadata for each chunk/summary node
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".org"}


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    chunk_index: int
    level: int = 0


@dataclass
class Node:
    id: str
    text: str
    level: int
    source_files: List[str]
    child_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    node_type: str = "summary"  # chunk | summary | root


@dataclass
class ClusteringConfig:
    target_dim: int = 10
    max_clusters: int = 12
    random_state: int = 42
    umap_n_neighbors: Optional[int] = None


@dataclass
class ClusteringStats:
    reducer: str
    target_dim: int
    cluster_count: int


class Summarizer:
    def summarize(self, text: str, max_tokens: int = 100) -> str:
        raise NotImplementedError


class LocalTransformersSummarizer(Summarizer):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        # This model is used only for recursive parent-node summaries.
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # NOTE: I use a MAC so I optimized it for MAC GPUs
        if torch.backends.mps.is_available():
            self.device = "mps"
            torch_dtype = torch.float16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def summarize(self, text: str, max_tokens: int = 100) -> str:
        import torch

        # Prompt for Qwen to generate summaries for each cluster
        prompt = (
            "You are building a RAPTOR retrieval tree.\n"
            "Write a factual parent summary for the following cluster.\n"
            "Return one dense paragraph of about 3 to 6 sentences.\n"
            "Preserve the main topics, methods, findings, and technical terminology.\n"
            "Do not use bullets.\n\n"
            f"CLUSTER TEXT:\n{text}"
        )

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )
        else:
            model_inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )

        if isinstance(model_inputs, torch.Tensor):
            input_ids = model_inputs.to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            model_inputs = model_inputs.to(self.device)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        summary = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not summary:
            raise RuntimeError("Local summarizer returned an empty summary")
        return summary


class Embedder:
    provider = "base"
    dimension = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class SentenceTransformerEmbedder(Embedder):
    provider = "sentence-transformers"

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Nomic expects retrieval-specific prefixes for best embedding quality.
        if self.model_name.startswith("nomic-ai/nomic-embed-text"):
            texts = [f"search_document: {text}" for text in texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
def sanitize_id_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "doc"


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: python-docx") from exc

    document = Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def read_doc(path: Path) -> str:
    try:
        import textract  # type: ignore

        return textract.process(str(path)).decode("utf-8", errors="ignore")
    except Exception:
        # Fallback for systems with antiword installed
        result = subprocess.run(["antiword", str(path)], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    raise RuntimeError(
        "Could not parse .doc file. Install either `textract` (Python) or `antiword` (system tool)."
    )


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".org"}:
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".doc":
        return read_doc(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def discover_files(input_dir: Path, single_file: Optional[str]) -> List[Path]:
    if single_file:
        p = Path(single_file)
        if not p.is_absolute():
            p = input_dir / single_file
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {p.suffix}")
        return [p]

    # RAPTOR should see the full corpus, not just one folder level, so we walk
    # the input tree recursively and ignore hidden files/directories.
    files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.parts)
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def chunk_text(text: str, chunk_size_tokens: int = 100, overlap_tokens: int = 20) -> List[str]:
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: tiktoken") from exc

    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    encoder = tiktoken.get_encoding("cl100k_base")
    sentences = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if re.sub(r"\s+", " ", sentence).strip()
    ]
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0
    sentence_token_lengths: List[int] = []

    def flush_current() -> None:
        nonlocal current_sentences, current_tokens, sentence_token_lengths
        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())
        if overlap_tokens > 0 and current_sentences:
            kept_sentences: List[str] = []
            kept_token_lengths: List[int] = []
            kept_tokens = 0
            for sentence, token_len in zip(reversed(current_sentences), reversed(sentence_token_lengths)):
                kept_sentences.insert(0, sentence)
                kept_token_lengths.insert(0, token_len)
                kept_tokens += token_len
                if kept_tokens >= overlap_tokens:
                    break
            current_sentences = kept_sentences
            sentence_token_lengths = kept_token_lengths
            current_tokens = kept_tokens
        else:
            current_sentences = []
            sentence_token_lengths = []
            current_tokens = 0

    for sentence in sentences:
        sentence_tokens = encoder.encode(sentence)

        # Login from the RAPTOR paper: do not cut in the middle of a sentence
        # unless the sentence itself exceeds the chunk budget.
        if len(sentence_tokens) > chunk_size_tokens:
            flush_current()
            start = 0
            while start < len(sentence_tokens):
                end = min(start + chunk_size_tokens, len(sentence_tokens))
                chunk = encoder.decode(sentence_tokens[start:end]).strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(sentence_tokens):
                    break
                start = max(end - overlap_tokens, start + 1)
            continue

        if current_sentences and current_tokens + len(sentence_tokens) > chunk_size_tokens:
            flush_current()

        current_sentences.append(sentence)
        sentence_token_lengths.append(len(sentence_tokens))
        current_tokens += len(sentence_tokens)

    flush_current()
    return chunks


def build_leaf_chunks(
    files: Iterable[Path],
    chunk_size_tokens: int,
    overlap_tokens: int,
    skip_failed_files: bool = True,
) -> Tuple[List[Chunk], List[dict]]:
    chunks: List[Chunk] = []
    failed_files: List[dict] = []
    for f in files:
        try:
            text = extract_text(f)
        except Exception as exc:
            if not skip_failed_files:
                raise
            failed_files.append({"file": str(f), "error": str(exc)})
            continue
        doc_chunks = chunk_text(text, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
        doc_key = sanitize_id_part(f.stem)
        doc_hash = md5(str(f.resolve()).encode("utf-8")).hexdigest()[:8]
        for idx, chunk in enumerate(doc_chunks):
            chunk_id = f"chunk_{doc_key}_{doc_hash}_{idx:04d}"
            chunks.append(Chunk(id=chunk_id, text=chunk, source_file=str(f), chunk_index=idx, level=0))
    return chunks, failed_files


def pick_cluster_count(n_items: int) -> int:
    if n_items <= 3:
        return 1
    return max(2, min(int(math.sqrt(n_items)), max(2, n_items // 2)))


def reduce_embeddings(
    embeddings: List[List[float]],
    config: ClusteringConfig,
) -> Tuple[np.ndarray, str]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: umap-learn") from exc

    matrix = np.asarray(embeddings, dtype=float)
    if len(matrix) <= 2:
        return matrix, "identity"

    target_dim = min(config.target_dim, matrix.shape[1], len(matrix) - 1)
    if target_dim < 2:
        return matrix, "identity"
    if len(matrix) <= target_dim + 2:
        return matrix, "identity"

    # The paper-style path reduces the embedding space before GMM clustering.
    n_neighbors = config.umap_n_neighbors or max(2, min(15, len(matrix) - 1))
    reduced = umap.UMAP(
        n_components=target_dim,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=config.random_state,
    ).fit_transform(matrix)
    return reduced, "umap"


def get_optimal_cluster_count(reduced_embeddings: np.ndarray, config: ClusteringConfig) -> int:
    n_items = len(reduced_embeddings)
    if n_items <= 2:
        return 1

    max_clusters = min(config.max_clusters, pick_cluster_count(n_items), n_items - 1)
    if max_clusters <= 1:
        return 1

    best_k = 1
    best_bic = float("inf")

    # Select the GMM size with the best BIC score.
    for k in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, random_state=config.random_state)
        gmm.fit(reduced_embeddings)
        bic = gmm.bic(reduced_embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k


def cluster_embeddings(
    embeddings: List[List[float]],
    config: ClusteringConfig,
) -> Tuple[List[int], ClusteringStats]:
    if len(embeddings) <= 2:
        return [0] * len(embeddings), ClusteringStats(reducer="identity", target_dim=0, cluster_count=1)

    reduced_embeddings, reducer = reduce_embeddings(embeddings, config)
    cluster_count = get_optimal_cluster_count(reduced_embeddings, config)

    if cluster_count <= 1:
        return [0] * len(embeddings), ClusteringStats(
            reducer=reducer,
            target_dim=int(reduced_embeddings.shape[1]) if reduced_embeddings.ndim == 2 else 0,
            cluster_count=1,
        )

    gmm = GaussianMixture(n_components=cluster_count, random_state=config.random_state)
    labels = gmm.fit_predict(reduced_embeddings).tolist()
    return labels, ClusteringStats(
        reducer=reducer,
        target_dim=int(reduced_embeddings.shape[1]) if reduced_embeddings.ndim == 2 else 0,
        cluster_count=cluster_count,
    )


def build_raptor_tree(
    chunks: List[Chunk],
    summarizer: Summarizer,
    embedder: Embedder,
    clustering_config: ClusteringConfig,
    max_levels: int = 4,
    summary_max_tokens: int = 100,
) -> Tuple[List[Node], Dict[str, List[float]]]:
    # nodes: every chunk/summary/root node in the final RAPTOR tree
    # embeddings_by_id: vector for every exported node
    nodes: List[Node] = []
    embeddings_by_id: Dict[str, List[float]] = {}

    current_level_nodes: List[Node] = []
    for c in chunks:
        leaf = Node(
            id=c.id,
            text=c.text,
            level=0,
            source_files=[c.source_file],
            child_ids=[],
            parent_id=None,
            node_type="chunk",
        )
        nodes.append(leaf)
        current_level_nodes.append(leaf)

    # Step 1: embed the leaf chunks.
    leaf_embeddings = embedder.embed([n.text for n in current_level_nodes])
    for node, emb in zip(current_level_nodes, leaf_embeddings):
        embeddings_by_id[node.id] = emb

    level = 1
    while level <= max_levels and len(current_level_nodes) > 1:
        # Step 2: cluster the current layer's embeddings.
        level_embeddings = [embeddings_by_id[n.id] for n in current_level_nodes]
        labels, clustering_stats = cluster_embeddings(level_embeddings, clustering_config)
        grouped: Dict[int, List[Node]] = defaultdict(list)
        for node, label in zip(current_level_nodes, labels):
            grouped[label].append(node)

        # STOP building higher tree levels logic:
        # Stop when another level would not abstract anything further.
        if clustering_stats.cluster_count <= 1:
            current_level_nodes[0].node_type = "root"
            break
        if all(len(members) == 1 for members in grouped.values()):
            current_level_nodes[0].node_type = "root"
            break

        next_level_nodes: List[Node] = []
        for cluster_idx, members in grouped.items():
            # Step 3: summarize each cluster into a parent node.
            member_text = "\n\n".join(m.text for m in members)
            summary = summarizer.summarize(member_text, max_tokens=summary_max_tokens)
            source_files = sorted({sf for m in members for sf in m.source_files})

            node_type = "root" if len(grouped) == 1 else "summary"
            parent = Node(
                id=f"summary_l{level}_c{cluster_idx}",
                text=summary,
                level=level,
                source_files=source_files,
                child_ids=[m.id for m in members],
                parent_id=None,
                node_type=node_type,
            )

            for m in members:
                m.parent_id = parent.id

            nodes.append(parent)
            next_level_nodes.append(parent)

        # Step 4: embed the summaries so they can be clustered again at the next level.
        parent_embeddings = embedder.embed([n.text for n in next_level_nodes])
        for node, emb in zip(next_level_nodes, parent_embeddings):
            embeddings_by_id[node.id] = emb

        current_level_nodes = next_level_nodes
        level += 1

        if len(current_level_nodes) == 1:
            current_level_nodes[0].node_type = "root"
            break

    return nodes, embeddings_by_id


def build_pinecone_payload(nodes: List[Node], embeddings_by_id: Dict[str, List[float]]) -> List[dict]:
    # Every chunk and every summary node is exported as a Pinecone vector.
    payload = []
    for n in nodes:
        payload.append(
            {
                "id": n.id,
                "values": embeddings_by_id[n.id],
                "metadata": {
                    "text": n.text,
                    "level": n.level,
                    "node_type": n.node_type,
                    "parent_id": n.parent_id,
                    "child_ids": n.child_ids,
                    "source_files": n.source_files,
                },
            }
        )
    return payload


def build_stats(
    input_files: List[Path],
    failed_files: List[dict],
    nodes: List[Node],
    payload_data: List[dict],
) -> dict:
    exported_source_files = sorted(
        {
            source_file
            for node in nodes
            for source_file in node.source_files
        }
    )

    level_counts: Dict[str, int] = defaultdict(int)
    node_type_counts: Dict[str, int] = defaultdict(int)
    chunk_counts_by_source: Dict[str, int] = defaultdict(int)

    for node in nodes:
        level_counts[str(node.level)] += 1
        node_type_counts[node.node_type] += 1
        if node.node_type == "chunk":
            for source_file in node.source_files:
                chunk_counts_by_source[source_file] += 1

    missing_source_files = sorted({str(path) for path in input_files} - set(exported_source_files))

    return {
        "input_supported_file_count": len(input_files),
        "input_supported_files": [str(path) for path in input_files],
        "failed_file_count": len(failed_files),
        "failed_files": failed_files,
        "exported_source_file_count": len(exported_source_files),
        "exported_source_files": exported_source_files,
        "missing_source_file_count": len(missing_source_files),
        "missing_source_files": missing_source_files,
        "total_vectors": len(payload_data),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "level_counts": dict(sorted(level_counts.items(), key=lambda item: int(item[0]))),
        "chunk_counts_by_source_file": dict(sorted(chunk_counts_by_source.items())),
    }


def write_outputs(
    output_dir: Path,
    nodes: List[Node],
    embeddings_by_id: Dict[str, List[float]],
    input_files: List[Path],
    failed_files: List[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # handoff_path contains my json that will be uploaded to pinecone. Stats file contains a report of the run 
    # because the json file be very large and difficult to read through.
    handoff_path = output_dir / "pinecone_all_vectors.json"
    stats_path = output_dir / "pinecone_stats.json"
    payload_data = build_pinecone_payload(nodes, embeddings_by_id)
    handoff_data = {
        "namespace": "default",
        "vectors": payload_data,
    }
    stats_data = build_stats(input_files, failed_files, nodes, payload_data)

    handoff_path.write_text(json.dumps(handoff_data, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats_data, indent=2), encoding="utf-8")
    print(f"Saved Pinecone handoff: {handoff_path}")
    print(f"Saved Pinecone stats: {stats_path}")

    for path in output_dir.glob("*.json"):
        if path not in {handoff_path, stats_path}:
            path.unlink()


def build_summarizer(summary_model: str) -> Summarizer:
    try:
        return LocalTransformersSummarizer(model_name=summary_model)
    except ImportError as exc:
        raise RuntimeError("Transformers dependencies missing for local summarization.") from exc


def build_embedder(embedding_model: str) -> Embedder:
    try:
        model_name = embedding_model or "nomic-ai/nomic-embed-text-v1.5"
        return SentenceTransformerEmbedder(model_name=model_name)
    except ImportError as exc:
        raise RuntimeError("Embedding model dependencies are missing for local embeddings.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict RAPTOR-style indexing pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("documents"), help="Directory with source documents")
    parser.add_argument("--file", type=str, default=None, help="Optional single filename/path to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("indexing/output/latest"),
        help="Directory for generated artifacts",
    )
    parser.add_argument("--chunk-size-tokens", type=int, default=100)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=20)
    parser.add_argument("--max-levels", type=int, default=4)
    parser.add_argument("--summary-max-tokens", type=int, default=180)
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Fail immediately if any file extraction errors occur",
    )
    parser.add_argument("--summary-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-ai/nomic-embed-text-v1.5",
        help="Model name for the local embedding model",
    )
    parser.add_argument(
        "--cluster-reduction-dim",
        type=int,
        default=10,
        help="UMAP output dimension used before GMM clustering",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=12,
        help="Upper bound used when selecting GMM cluster count with BIC",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Pipeline entry point:
    # discover corpus -> build chunks -> build RAPTOR tree -> export Pinecone JSON
    files = discover_files(args.input_dir, args.file)
    if not files:
        raise RuntimeError(
            f"No supported documents found in {args.input_dir}. "
            "Supported: .pdf, .doc, .docx, .txt, .org"
        )

    summarizer = build_summarizer(args.summary_model)
    embedder = build_embedder(args.embedding_model)
    clustering_config = ClusteringConfig(
        target_dim=args.cluster_reduction_dim,
        max_clusters=args.max_clusters,
    )
    chunks, failed_files = build_leaf_chunks(
        files,
        args.chunk_size_tokens,
        args.chunk_overlap_tokens,
        skip_failed_files=not args.strict_files,
    )
    if not chunks:
        raise RuntimeError("No chunks generated. Check document contents.")

    nodes, embeddings_by_id = build_raptor_tree(
        chunks=chunks,
        summarizer=summarizer,
        embedder=embedder,
        clustering_config=clustering_config,
        max_levels=args.max_levels,
        summary_max_tokens=args.summary_max_tokens,
    )

    write_outputs(
        args.output_dir,
        nodes,
        embeddings_by_id,
        files,
        failed_files,
    )


if __name__ == "__main__":
    main()
