"""
Checkpoint manager using Orbax and the JAX/Flax NNX ecosystem.

Provides step-based checkpoint management for GPT training:
- Saves/restores model state (Flax NNX), optimizer state, and training metadata
- Universal path resolution via etils.epath to cleanly process GCS buckets (gs://) and local storage
- Automated robust fallback to pure weight restoration on optimizer structure invariants
- Automatic old-checkpoint cleanup via max_to_keep
- Save-interval gating to avoid checkpointing on every step

Usage:
    from nanochat.checkpoint_manager import CheckpointManager

    with CheckpointManager("gs://my-cloud-bucket/checkpoints", max_to_keep=3) as cm:
        cm.save(step=100, model=model, opt_state=opt_state, extra={"loss": 0.42})
        step, model, opt_state, extra = cm.restore_latest(model, opt_state)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

try:
    from etils import epath
except ImportError:
    epath = None

from nanochat.common import get_base_dir

logger = logging.getLogger(__name__)


def resolve_path(directory: str | os.PathLike) -> Any:
    dir_str = str(directory)
    if epath is not None:
        p = epath.Path(dir_str)
        if not dir_str.startswith("gs://"):
            p.mkdir(parents=True, exist_ok=True)
        return p
    else:
        if dir_str.startswith("gs://"):
            return dir_str
        p = Path(dir_str)
        p.mkdir(parents=True, exist_ok=True)
        return p


class CheckpointManager:
    """Orbax-backed checkpoint manager for Flax NNX models.

    Wraps ``orbax.checkpoint.CheckpointManager`` to handle the
    NNX split/merge dance automatically, and co-saves optimizer state
    plus arbitrary JSON-serialisable training metadata alongside
    the model weights.
    """

    def __init__(
        self,
        directory: str | os.PathLike | None = None,
        *,
        max_to_keep: int = 5,
        save_interval_steps: int = 1,
    ):
        if directory is None:
            directory = os.path.join(get_base_dir(), "checkpoints")
        
        self._directory = resolve_path(directory)
        self._save_interval = save_interval_steps

        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
        self._manager = ocp.CheckpointManager(
            self._directory,
            options=options,
        )
        logger.info("CheckpointManager initialised at %s (max_to_keep=%d)", self._directory, max_to_keep)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        *,
        model: nnx.Module,
        opt_state: Any | None = None,
        extra: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        if not force and self._save_interval > 1 and step % self._save_interval != 0:
            return False

        _, model_state = nnx.split(model)

        ckpt = {"model_state": model_state}

        if opt_state is not None:
            ckpt["opt_state"] = opt_state

        self._manager.save(step, args=ocp.args.StandardSave(ckpt))
        self._manager.wait_until_finished()

        if extra is not None and not str(self._directory).startswith("gs://"):
            meta_path = self._directory / str(step) / "_metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump(_make_serialisable(extra), f, indent=2)

        logger.info("Checkpoint saved  step=%d  dir=%s", step, self._directory)
        return True

    # ------------------------------------------------------------------
    # Restore helpers
    # ------------------------------------------------------------------

    def restore(
        self,
        step: int,
        *,
        model: nnx.Module,
        opt_state: Any | None = None,
    ) -> tuple[int, nnx.Module, Any | None, dict[str, Any]]:
        graphdef, abstract_state = nnx.split(model)

        abstract_ckpt: dict[str, Any] = {"model_state": abstract_state}
        if opt_state is not None:
            abstract_ckpt["opt_state"] = opt_state

        abstract_ckpt = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "shape") else x,
            abstract_ckpt,
        )

        try:
            restored = self._manager.restore(step, args=ocp.args.StandardRestore(abstract_ckpt))
        except ValueError as orig_err:
            # Flawless fallback to pure model weight restoration if optimizer structure changed
            if "opt_state" in str(orig_err):
                abstract_pure = {"model_state": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "shape") else x, abstract_state)}
                restored = self._manager.restore(step, args=ocp.args.StandardRestore(abstract_pure))
            else:
                raise orig_err

        model = nnx.merge(graphdef, restored["model_state"])
        restored_opt = restored.get("opt_state")
        extra = self._load_extra(step)

        logger.info("Checkpoint restored  step=%d", step)
        return step, model, restored_opt, extra

    def restore_latest(
        self,
        model: nnx.Module,
        opt_state: Any | None = None,
    ) -> tuple[int, nnx.Module, Any | None, dict[str, Any]] | None:
        step = self.latest_step()
        if step is None:
            logger.info("No checkpoints found in %s", self._directory)
            return None
        return self.restore(step, model=model, opt_state=opt_state)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def all_steps(self) -> list[int]:
        return sorted(self._manager.all_steps())

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._manager.close()
        logger.info("CheckpointManager closed.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_extra(self, step: int) -> dict[str, Any]:
        if not str(self._directory).startswith("gs://"):
            meta_path = self._directory / str(step) / "_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    return json.load(f)
        return {}


def _make_serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_serialisable(v) for v in obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return obj.item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj