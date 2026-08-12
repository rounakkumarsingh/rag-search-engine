import random
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from cli.lib.config import CROSS_ENCODER_MODEL, EMBEDDING_MODEL


def seed_all(seed: int = 0) -> None:
    """Seed all RNG so embedding computation is reproducible on a device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    seed_all()
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    seed_all()
    return CrossEncoder(CROSS_ENCODER_MODEL)
