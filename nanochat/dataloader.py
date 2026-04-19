"""
Grain-based data loader for JAX pretraining.

Two modes are supported:

1. Pre-tokenized .bin shards (via ``pretokenized_distributed_data_loader``):
   Loads pre-tokenized binary shard files. Each shard has a 256 x int32 header
   followed by uint16 tokens. Uses the grain library for parallel,
   shared-memory shard loading and the BOSFinder helper for efficient batch
   construction across document boundaries.

2. Parquet-based streaming (via ``tokenizing_distributed_data_loader``):
   Streams raw text from parquet files, tokenizes on-the-fly, and yields
   numpy batches compatible with jax.device_put.
"""

from collections import deque
from pathlib import Path

import grain
import jax
import numpy as np
from grain.multiprocessing import SharedMemoryArray

from nanochat.dataset import list_parquet_files, parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOS_ID = 50256  # GPT-2 style endoftext token used as BOS delimiter


# ============================================================================
# BOSFinder — efficient batch slicing for pre-tokenized shards
# ============================================================================

class BOSFinder:
    """Index BOS positions in a token array and yield (start, end) slices
    that pack exactly ``max_seq_len + 1`` tokens per sequence in a batch."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.size = len(tokens)
        self.bos_idx = np.where(tokens == BOS_ID)[0]
        self.i = 0
        self.batch_iter = 0
        self.built_ready = False

    # -- pre-build a static index of all batches in this shard ------------
    def build(self, batch_size, max_seq_len):
        n = len(self.bos_idx)
        target_len = max_seq_len + 1
        starts, ends = [], []
        ptrs = [0]
        idx = 0

        while True:
            batch_pairs_begin = len(starts)
            full_batch = True
            for _ in range(batch_size):
                cur_len = 0
                while cur_len < target_len:
                    if idx >= n:
                        full_batch = False
                        break
                    cur = self.bos_idx[idx]
                    starts.append(cur)
                    remaining = target_len - cur_len
                    next_bos = self.bos_idx[idx + 1] if idx + 1 < n else self.size
                    end = min(next_bos, cur + remaining)
                    ends.append(end)
                    cur_len += end - cur
                    idx += 1
                if not full_batch:
                    break
                assert cur_len == target_len
            if not full_batch:
                del starts[batch_pairs_begin:]
                del ends[batch_pairs_begin:]
                break
            ptrs.append(len(starts))

        self.built_starts = np.asarray(starts, dtype=np.int32)
        self.built_ends = np.asarray(ends, dtype=np.int32)
        self.built_ptrs = np.asarray(ptrs, dtype=np.int64)
        self.built_batch_size = batch_size
        self.built_max_seq_len = max_seq_len
        self.built_ready = True
        self.i = 0
        self.batch_iter = 0
        return len(self.built_ptrs) - 1  # number of full batches

    # -- yield one batch worth of (starts, ends) pairs --------------------
    def next_batch(self, batch_size: int, max_seq_len: int):
        # Fast path: use prebuilt index
        if (
            self.built_ready
            and self.built_batch_size == batch_size
            and self.built_max_seq_len == max_seq_len
        ):
            b = self.batch_iter
            if b >= len(self.built_ptrs) - 1:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            p0 = int(self.built_ptrs[b])
            p1 = int(self.built_ptrs[b + 1])
            starts = self.built_starts[p0:p1].tolist()
            ends = self.built_ends[p0:p1].tolist()
            self.i += p1 - p0
            self.batch_iter += 1
            return starts, ends

        # Fallback: on-the-fly path
        n = len(self.bos_idx)
        starts, ends = [], []
        idx = self.i
        for _ in range(batch_size):
            cur_len = 0
            target_len = max_seq_len + 1
            while cur_len < target_len:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts.append(cur)
                remaining = target_len - cur_len
                next_bos = self.bos_idx[idx + 1] if idx + 1 < n else self.size
                end = min(next_bos, cur + remaining)
                ends.append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == target_len

        self.i = idx
        self.batch_iter += 1
        return starts, ends


# ============================================================================
# Grain data-source and transforms for pre-tokenized .bin shards
# ============================================================================

class CustomSharedMemoryDataSource(grain.sources.SharedMemoryDataSource):
    """Thin wrapper that resolves paths and stores file list."""

    def __init__(self, elements=None, *, name=None):
        if elements is not None:
            elements = [str(Path(p).resolve()) for p in elements]
        super().__init__(elements, name=name)
        self.files = [] if elements is None else elements
        self.name = name

    def __repr__(self):
        return f"Fineweb10BSharedMemoryData(name={self.name}, len={len(self.files)})"


class LoadShardTokens(grain.transforms.Map):
    """Grain map transform: read a .bin shard into shared memory."""

    def map(self, path):
        file = Path(path)

        header = np.fromfile(str(file), count=256, dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2])

        with file.open("rb", buffering=0) as f:
            f.seek(256 * 4)
            tokens = SharedMemoryArray((num_tokens,), dtype=np.uint16)
            nbytes = f.readinto(tokens)
            assert nbytes == 2 * num_tokens, (
                "number of tokens read does not match header"
            )

        bos_idx = np.flatnonzero(tokens == BOS_ID)
        return {
            "path": str(file),
            "tokens": tokens,
            "bos_idx": bos_idx,
            "size": num_tokens,
        }


def make_grain_shard_loader(files):
    """Build a grain iterator that loads .bin shards in parallel."""
    ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens())
    performance_config = grain.experimental.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024 * 10,
        max_workers=None,
        max_buffer_size=None,
    )
    ds = ds.to_iter_dataset(read_options=performance_config.read_options)
    return ds


# ============================================================================
# Pre-tokenized data loader  (grain + BOSFinder)
# ============================================================================

def pretokenized_distributed_data_loader(B, T, split, shard_files):
    """Yield (inputs, targets) as numpy int32/int64 arrays from .bin shards.

    Parameters
    ----------
    B : int
        Batch size (number of sequences per batch).
    T : int
        Sequence length (max_seq_len).
    split : str
        ``"train"`` or ``"val"``.
    shard_files : list[str | Path]
        Paths to ``.bin`` shard files.

    Yields
    ------
    inputs : np.ndarray, shape (B, T), dtype int32
    targets : np.ndarray, shape (B, T), dtype int64
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    needed = T + 1  # +1 for targets

    while True:
        shard_iter = make_grain_shard_loader(shard_files)
        for shard in shard_iter:
            tokens = np.asarray(shard["tokens"], dtype=np.int64)
            finder = BOSFinder(tokens)
            finder.build(B, T)

            try:
                while True:
                    starts, ends = finder.next_batch(B, T)
                    # Assemble the flat token stream for this batch
                    flat = np.empty(B * needed, dtype=np.int64)
                    pos = 0
                    for s, e in zip(starts, ends):
                        length = e - s
                        flat[pos:pos + length] = tokens[s:e]
                        pos += length

                    buf = flat[: B * needed].reshape(B, needed)
                    inputs = buf[:, :-1].astype(np.int32)
                    targets = buf[:, 1:].astype(np.int64)
                    yield inputs, targets
            except StopIteration:
                continue  # move to next shard


# ============================================================================
# Parquet-based tokenizing data loader (streaming, JAX-compatible)
# ============================================================================

def tokenizing_distributed_data_loader(
    B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
    process_index=0, process_count=1,
):
    """Stream pretraining text from parquet files, tokenize, yield training batches.

    This mirrors the original PyTorch dataloader but outputs plain numpy arrays
    that are directly consumable by JAX via ``jax.device_put``.

    Parameters
    ----------
    B : int
        Batch size.
    T : int
        Sequence length.
    split : str
        ``"train"`` or ``"val"``.
    tokenizer_threads : int
        Number of threads for the tokenizer.
    tokenizer_batch_size : int
        Number of documents to tokenize at once.
    process_index : int
        Current process rank (for multi-host JAX, replaces DDP rank).
    process_count : int
        Total number of processes (replaces DDP world size).

    Yields
    ------
    inputs : np.ndarray, shape (B, T), dtype int32
    targets : np.ndarray, shape (B, T), dtype int64
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    needed_tokens = B * T + 1

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    token_buffer = deque()
    scratch = np.empty(needed_tokens, dtype=np.int64)

    # Infinite iterator over document batches
    def document_batches():
        while True:
            for batch in parquets_iter_batched(
                split=split, start=process_index, step=process_count
            ):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size]

    batches = document_batches()

    while True:
        # Accumulate enough tokens
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)

        # Fill scratch buffer
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()

        inputs = scratch[:-1].astype(np.int32).reshape(B, T)
        targets = scratch[1:].astype(np.int64).reshape(B, T)
        yield inputs, targets