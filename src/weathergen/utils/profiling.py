import logging
import os
import platform
import traceback
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import weathergen.common.config as config
from weathergen.utils.distributed import get_rank

logger: logging.Logger = logging.getLogger(__name__)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history(max_entries: int = MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT) -> None:
    """Start the allocator trace.

    `max_entries` is a ring buffer of events, not a time span: once it wraps, the snapshot
    covers only the tail of the run and everything allocated earlier loses its call site.
    Size it against the number of steps profiled rather than leaving it at the default.
    """

    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info(f"Starting snapshot record_memory_history (max_entries={max_entries})")
    torch.cuda.memory._record_memory_history(max_entries=max_entries)


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not stopping memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(cfg: dict | OmegaConf) -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    base_path = config.get_path_profiling_traces(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{get_rank()}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


class AutocastPromotionAudit(TorchDispatchMode):
    """Log operations that autocast promotes to float32, and what that costs in bytes.

    Autocast keeps a list of operations it runs in float32 whatever dtype it is handed
    (`layer_norm`, `sum`, `softmax`, `cumsum`, `norm`, ...). Handed a bfloat16 activation,
    such an operation materialises a float32 copy of the input *and* a float32 result --
    four times the bytes -- and the result then stays float32 until something casts it
    back. Three separate instances of this have dominated the training-step memory peak,
    each found only after it caused a crash or a confusing profile, so this maps them
    directly instead, charging each promotion to the operation and model call site
    responsible.

    Reports where promotion *happens*, not whether it propagates. A promoted result that is
    cast straight back down still appears here -- correctly, since it was still
    materialised -- so a site staying in the report after a fix is not a failed fix. What
    the fix changes is how long the float32 lives, which is a question for the allocator
    snapshot.

    Not free -- it intercepts every dispatch -- so it belongs in a profiling run over a
    couple of steps, not in training. Report is written by `write_report`, sorted by total
    bytes promoted, which is the order worth fixing them in.
    """

    #: below this an individual promoted result is noise, not an activation
    DEFAULT_MIN_BYTES: int = 1 << 20

    _REDUCED = (torch.bfloat16, torch.float16)

    def __init__(self, min_bytes: int = DEFAULT_MIN_BYTES):
        super().__init__()
        self.min_bytes = min_bytes
        # (op, call site) -> [count, total bytes]
        self.hits: dict[tuple[str, str], list[int]] = {}
        self._pending_bytes: int = 0
        self._pending_site: str = ""

    def _call_site(self) -> str:
        """The innermost weathergen frames that led here, outermost first."""

        frames = [
            f"{Path(f.filename).name}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()
            if f"{os.sep}weathergen{os.sep}" in f.filename and "profiling.py" not in f.filename
        ]
        return " > ".join(frames[-3:]) if frames else "<no weathergen frame>"

    def _float32_bytes(self, tensors) -> int:
        return sum(
            t.nbytes
            for t in tensors
            if isinstance(t, torch.Tensor)
            and t.dtype is torch.float32
            and t.nbytes >= self.min_bytes
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))

        name = str(func)
        flat_in = tree_flatten((args, kwargs))[0]
        flat_out = tree_flatten(out)[0]

        # A promotion reaches this mode as two dispatches: autocast's own `_to_copy` upcast,
        # then the operation it was inserted for. Naming the `_to_copy` would be useless --
        # the report needs to say `layer_norm` or `sum` -- so the upcast is held back and
        # charged to whatever consumes it next, together with that op's float32 result.
        if name.startswith("aten._to_copy"):
            if any(isinstance(t, torch.Tensor) and t.dtype in self._REDUCED for t in flat_in):
                upcast = self._float32_bytes(flat_out)
                if upcast:
                    self._pending_bytes += upcast
                    self._pending_site = self._pending_site or self._call_site()
            return out

        promoted = self._pending_bytes
        if promoted:
            promoted += self._float32_bytes(flat_out)
            entry = self.hits.setdefault((name, self._pending_site), [0, 0])
            entry[0] += 1
            entry[1] += promoted
            self._pending_bytes = 0
            self._pending_site = ""
        return out

    def write_report(self, cfg: dict | OmegaConf) -> None:
        base_path = config.get_path_profiling_traces(cfg)
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        path = base_path / f"{timestamp}_rank_{get_rank()}_autocast_promotions.txt"

        rows = sorted(self.hits.items(), key=lambda kv: -kv[1][1])
        total = sum(v[1] for v in self.hits.values())
        lines = [
            "Operations autocast promoted to float32 from reduced-precision inputs.",
            f"Threshold: results >= {self.min_bytes / 2**20:.1f} MiB. "
            f"Total promoted: {total / 2**30:.2f} GiB over {len(rows)} sites.",
            "",
            f"{'promoted MiB':>13}  {'n':>7}  {'op':<34} call site",
        ]
        lines += [
            f"{nbytes / 2**20:13.1f}  {count:7d}  {op.replace('aten.', '')[:34]:<34} {site}"
            for (op, site), (count, nbytes) in rows
        ]
        path.write_text("\n".join(lines) + "\n")
        logger.info(f"Wrote autocast promotion audit to {path} ({total / 2**30:.2f} GiB)")


def trace_handler(cfg: dict | OmegaConf, prof: torch.profiler.profile) -> None:
    # Prefix for file names.
    base_path = config.get_path_profiling_traces(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{get_rank()}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    on_aarch64 = platform.machine() == "aarch64"
    if not on_aarch64:
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    else:
        logger.info("[profiler] Memory distribution timeline skipped on aarch64")


def wrap_module_forward_with_profiling(model, prefix=""):
    """
    Recursively wrap all nn.Module forward methods with profiling context
    """
    for name, module in model.named_children():
        module_name = f"{prefix}.{name}" if prefix else name

        # Skip standard PyTorch modules (they're already traced)
        if type(module).__module__.startswith("torch.nn.modules"):
            # Still recurse into children
            wrap_module_forward_with_profiling(module, module_name)
            continue

        # Wrap custom modules, stashing the original so it can be restored later
        original_forward = module.forward
        module._original_forward = original_forward

        def make_profiled_forward(mod_name, orig_forward):
            def profiled_forward(*args, **kwargs):
                with record_function(f"nn.Module: {mod_name}"):
                    return orig_forward(*args, **kwargs)

            return profiled_forward

        module.forward = make_profiled_forward(module_name, original_forward)

        # Recurse into children
        wrap_module_forward_with_profiling(module, module_name)


def unwrap_module_forward_with_profiling(model) -> None:
    """
    Undo wrap_module_forward_with_profiling, restoring the original forward
    methods so the record_function wrappers no longer add overhead.
    """
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
