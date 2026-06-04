from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


EMBEDDING_DIM = 128


def image_to_embedding(image_bytes: bytes) -> list[float]:
    """Return a stable demo embedding.

    This module intentionally exposes the same shape we will use for FaceNet vectors
    in Supabase. If a production FaceNet package is added later, replace this
    function with model inference and keep the storage code unchanged.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.fit(image, (160, 160))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    channel_features = []
    for channel in range(3):
        values = pixels[:, :, channel]
        hist, _ = np.histogram(values, bins=32, range=(0.0, 1.0), density=True)
        channel_features.extend(hist.tolist())

    digest = hashlib.sha256(image_bytes).digest()
    hash_features = [byte / 255 for byte in digest]
    vector = np.array((channel_features + hash_features)[:EMBEDDING_DIM], dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    vector = np.mean(np.array(embeddings, dtype=np.float32), axis=0)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=np.float32)
    b = np.array(right, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_match(
    candidate: list[float],
    known_workers: list[dict],
    threshold: float = 0.8,
) -> tuple[dict | None, float]:
    best_worker = None
    best_score = 0.0
    for worker in known_workers:
        score = cosine_similarity(candidate, worker["embedding"])
        if score > best_score:
            best_worker = worker
            best_score = score

    if best_score >= threshold:
        return best_worker, best_score
    return None, best_score


def ranked_matches(
    candidate: list[float],
    known_workers: list[dict],
    limit: int = 3,
) -> list[dict]:
    matches = []
    for worker in known_workers:
        matches.append(
            {
                "worker": worker,
                "score": cosine_similarity(candidate, worker["embedding"]),
            }
        )
    return sorted(matches, key=lambda match: match["score"], reverse=True)[:limit]
