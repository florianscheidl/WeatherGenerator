import logging
import os
import platform
import traceback
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.overrides import TorchFunctionMode
from torch.profiler import record_function
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


class AutocastPromotionAudit(TorchFunctionMode):
    """Log the operations autocast promotes to float32, and what each costs in bytes.

    Autocast keeps a list of operations it runs in float32 whatever dtype it is handed
    (`layer_norm`, `sum`, `softmax`, `cumsum`, `norm`, ...). Handed a bfloat16 activation,
    such an operation materialises a float32 copy of the input *and* a float32 result --
    four times the bytes -- and the result then stays float32 until something casts it
    back. Three separate instances of this have dominated the training-step memory peak,
    each found only after it caused a crash or a confusing profile, so this maps them
    directly instead: any call handed reduced precision that hands back float32, charged to
    the model call site that made it.

    This is a `TorchFunctionMode`, which sits *above* autocast, so it sees the dtypes the
    model actually passed rather than the ones autocast rewrote them to. An earlier version
    used a `TorchDispatchMode`, below autograd, where a promotion appears only as an
    anonymous `_to_copy` and the backward pass -- which autocast never touches -- swamps the
    report with gradient casts attributed to whatever ran next. If this ever reports `add`,
    `detach` or `stack` against `<no weathergen frame>`, it has regressed to that.

    Consequences of the level it runs at, both intended:

    - Forward only. Autograd is C++ and does not route through `__torch_function__`, which
      is correct here because autocast only applies to the forward.
    - Python-level calls only. An operation issued from inside a C++ kernel is invisible;
      everything in the fp32 list is reachable from Python, so this has not mattered.

    Reports where promotion *happens*, not whether it propagates. A promoted result that is
    cast straight back down still appears -- correctly, it was still materialised -- so a
    site remaining after a fix is not a failed fix. Whether the float32 then survives is a
    question for the allocator snapshot.

    Report is written by `write_report`, sorted by bytes promoted, which is the order worth
    fixing them in.
    """

    #: below this an individual promoted result is noise, not an activation
    DEFAULT_MIN_BYTES: int = 1 << 20

    _REDUCED = (torch.bfloat16, torch.float16)
    #: deliberate casts: the caller asked for float32, so it is not a surprise worth logging
    _EXPLICIT_CASTS = frozenset({"to", "float", "double", "type", "type_as", "_to_copy"})

    def __init__(self, min_bytes: int = DEFAULT_MIN_BYTES):
        super().__init__()
        self.min_bytes = min_bytes
        # (op, call site) -> [count, total bytes]
        self.hits: dict[tuple[str, str], list[int]] = {}

    def _call_site(self) -> str:
        """The innermost weathergen frames that led here, outermost first."""

        frames = [
            f"{Path(f.filename).name}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()
            if f"{os.sep}weathergen{os.sep}" in f.filename and "profiling.py" not in f.filename
        ]
        return " > ".join(frames[-3:]) if frames else "<no weathergen frame>"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        if getattr(func, "__name__", "") in self._EXPLICIT_CASTS:
            return out

        flat_in = tree_flatten((args, kwargs))[0]
        if not any(isinstance(t, torch.Tensor) and t.dtype in self._REDUCED for t in flat_in):
            return out

        promoted = sum(
            t.nbytes
            for t in tree_flatten(out)[0]
            if isinstance(t, torch.Tensor)
            and t.dtype is torch.float32
            and t.nbytes >= self.min_bytes
        )
        if promoted:
            name = getattr(func, "__qualname__", None) or getattr(func, "__name__", str(func))
            entry = self.hits.setdefault((name, self._call_site()), [0, 0])
            entry[0] += 1
            entry[1] += promoted
        return out

    def write_report(self, cfg: dict | OmegaConf) -> None:
        base_path = config.get_path_profiling_traces(cfg)
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        path = base_path / f"{timestamp}_rank_{get_rank()}_autocast_promotions.txt"

        rows = sorted(self.hits.items(), key=lambda kv: -kv[1][1])
        total = sum(v[1] for v in self.hits.values())
        lines = [
            "Calls handed reduced precision that returned float32 (autocast fp32 policy).",
            f"Forward pass only. Threshold: results >= {self.min_bytes / 2**20:.1f} MiB. "
            f"Total promoted: {total / 2**30:.2f} GiB over {len(rows)} sites.",
            "",
            f"{'promoted MiB':>13}  {'n':>7}  {'op':<34} call site",
        ]
        lines += [
            f"{nbytes / 2**20:13.1f}  {count:7d}  {op[:34]:<34} {site}"
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
