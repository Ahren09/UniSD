"""Embedding-based retrieval for selecting semantically similar donor examples.

All heavy compute (embedding, similarity, top-k) uses torch GPU tensors.
"""

import hashlib
import os
from pathlib import Path

import torch


def get_embedding_text(problem: dict) -> str:
    """Extract the text to embed from a formatted training example.

    Uses the student prompt content (contains task + signature + tests).
    """
    if isinstance(problem.get("prompt"), list):
        return problem["prompt"][0]["content"]
    return problem.get("prompt", "")


def compute_or_load_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: str | None = None,
    batch_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute sentence embeddings, L2-normalized, on `device`.

    Returns:
        (N, D) float32 tensor, L2-normalized, on `device`.

    Caching:
        If `cache_path` is provided and a valid cache exists (matching content hash),
        loads from disk. Otherwise computes and saves.
    """
    # Content hash for cache invalidation
    content_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]
    model_slug = model_name.replace("/", "_").replace("-", "_")

    if cache_path is not None:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"embeddings_{model_slug}_n{len(texts)}_{content_hash}.pt"

        if cache_file.exists():
            print(f"[Retrieval] Loading cached embeddings from {cache_file}")
            embeddings = torch.load(cache_file, map_location=device, weights_only=True)
            if embeddings.shape[0] == len(texts):
                return embeddings
            print(f"[Retrieval] Cache size mismatch ({embeddings.shape[0]} vs {len(texts)}), recomputing")

    # Compute embeddings
    from sentence_transformers import SentenceTransformer

    print(f"[Retrieval] Computing embeddings with {model_name} for {len(texts)} texts...")
    st_model = SentenceTransformer(model_name, device=device)
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalization
    )
    # Ensure float32 on target device
    embeddings = embeddings.float().to(device)

    # Free the sentence transformer model
    del st_model

    # Save cache
    if cache_path is not None:
        torch.save(embeddings.cpu(), cache_file)
        print(f"[Retrieval] Saved embeddings cache to {cache_file}")
        embeddings = embeddings.to(device)

    return embeddings


def retrieve_topk_donors(
    query_embeddings: torch.Tensor,  # (N, D) on GPU
    pool_embeddings: torch.Tensor,   # (M, D) on GPU
    k: int,
    self_indices: torch.Tensor | None = None,  # (N,) indices to mask out per row
    batch_size: int | None = 4096,  # chunk queries to bound peak memory; unbatched if N <= batch_size
) -> torch.Tensor:
    """Retrieve top-k most similar donors for each query via cosine similarity.

    Args:
        query_embeddings: (N, D) L2-normalized embeddings on GPU.
        pool_embeddings: (M, D) L2-normalized embeddings on GPU.
        k: Number of donors to retrieve per query.
        self_indices: (N,) int64 tensor of pool indices to mask out (self-exclusion).
            Only used when query and pool overlap. Each query[i] excludes pool[self_indices[i]].
        batch_size: If set, process queries in chunks of this size to reduce peak memory.
            None keeps the original unbatched code path.

    Returns:
        (N, k) int64 tensor of pool indices for each query.
        If k > available pool size, pads by repeating top indices via modular indexing.
    """
    N = query_embeddings.size(0)
    M = pool_embeddings.size(0)

    # Batched path: split queries into chunks to limit peak memory
    if batch_size is not None and batch_size < N:
        all_indices = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk_self = self_indices[start:end] if self_indices is not None else None
            chunk_indices = retrieve_topk_donors(
                query_embeddings[start:end], pool_embeddings, k,
                self_indices=chunk_self, batch_size=None,
            )
            all_indices.append(chunk_indices)
        return torch.cat(all_indices, dim=0)

    # Unbatched path (original logic)
    # Cosine similarity (embeddings are already L2-normalized)
    similarity = torch.mm(query_embeddings, pool_embeddings.T)  # (N, M)

    # Self-masking: exclude self from retrieval
    if self_indices is not None:
        row_indices = torch.arange(N, device=similarity.device)
        similarity[row_indices, self_indices] = -float('inf')

    # Available pool size after self-exclusion
    available = M - (1 if self_indices is not None else 0)
    actual_k = min(k, available)

    if actual_k <= 0:
        raise ValueError(f"Pool too small for retrieval: M={M}, k={k}, available={available}")

    # Top-k retrieval
    topk_result = torch.topk(similarity, k=actual_k, dim=1)
    indices = topk_result.indices  # (N, actual_k)

    # Pad if k > available by repeating with modular indexing
    if actual_k < k:
        repeat_count = (k + actual_k - 1) // actual_k  # ceil division
        indices = indices.repeat(1, repeat_count)[:, :k]

    return indices
