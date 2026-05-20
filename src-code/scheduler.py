"""
N730 Layer Scheduler
====================

The conveyor belt. Reads a .n730 file and streams layers through a
3-stage pipeline so the GPU (GT 730) is never waiting:

  Stage 1 — Disk thread:    reads raw bytes from .n730 (ONE persistent handle)
  Stage 2 — Dequant thread: converts INT2/INT4/INT8 → float32 in parallel
  Stage 3 — GPU slot:       receives fully-ready tensors, never blocks

Key improvements over naive implementation:
  - Persistent file handle: open once, seek many — no per-layer open() overhead
  - Split disk + dequant threads: I/O and CPU work overlap completely
  - Read-ahead uses block_size reads to amortize seek cost
  - Hit rate tracking shows whether prefetch depth is sufficient

Usage (Python API):
    from scheduler import N730Scheduler

    scheduler = N730Scheduler("model.n730", prefetch=4)
    for layer in scheduler.stream():
        output = run_forward_pass(layer.weights, activations)

Usage (CLI):
    python scheduler.py --model model.n730 --benchmark
    python scheduler.py --model model.n730 --layer 42
    python scheduler.py --model model.n730 --benchmark --prefetch 8
"""

import json
import struct
import threading
import queue
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ─── N730 format constants ────────────────────────────────────────────────────

MAGIC       = b"N730\x00\x01\x00\x00"
LAYER_MAGIC = b"LYR\x00"
PAGE_SIZE   = 4096
LAYER_HDR_SIZE = 4 + 1 + 4 + 4 + 4 + 4 + 4   # magic + prec + rows + cols + scale + zp + data_size = 25 bytes


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RawLayerBlock:
    """Raw bytes off disk — not yet dequantized. Lives briefly between threads."""
    layer_idx: int
    layer_name: str
    precision: str
    prec_id: int
    rows: int
    cols: int
    scale: float
    zero_point: float
    raw_bytes: bytes
    disk_time_ms: float


@dataclass
class LayerBuffer:
    """A dequantized layer in RAM, ready for GPU transfer."""
    layer_idx: int
    layer_name: str
    weights: np.ndarray
    precision: str
    load_time_ms: float     # disk read time
    dequant_time_ms: float  # dequantization time


@dataclass
class SchedulerStats:
    total_layers: int = 0
    total_bytes_read: int = 0
    total_disk_ms: float = 0.0
    total_dequant_ms: float = 0.0
    gpu_wait_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    peak_ram_layers: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / max(total, 1)

    @property
    def avg_disk_ms(self) -> float:
        return self.total_disk_ms / max(self.total_layers, 1)

    @property
    def avg_dequant_ms(self) -> float:
        return self.total_dequant_ms / max(self.total_layers, 1)

    @property
    def throughput_mb_s(self) -> float:
        total_ms = self.total_disk_ms + self.total_dequant_ms
        if total_ms < 1:
            return 0.0
        return (self.total_bytes_read / (1024**2)) / (total_ms / 1000)


# ─── Dequantization ───────────────────────────────────────────────────────────

def dequantize(raw_bytes: bytes, prec_id: int, rows: int, cols: int,
               scale: float, zero_point: float) -> np.ndarray:
    total = rows * cols

    if prec_id == 8:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        w = (arr - zero_point) * scale

    elif prec_id == 4:
        packed = np.frombuffer(raw_bytes, dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        interleaved = np.empty(lo.size * 2, dtype=np.float32)
        interleaved[0::2] = lo
        interleaved[1::2] = hi
        w = (interleaved[:total] - zero_point) * scale

    elif prec_id == 2:
        packed = np.frombuffer(raw_bytes, dtype=np.uint8)
        b0 = (packed & 0x03).astype(np.float32)
        b1 = ((packed >> 2) & 0x03).astype(np.float32)
        b2 = ((packed >> 4) & 0x03).astype(np.float32)
        b3 = ((packed >> 6) & 0x03).astype(np.float32)
        interleaved = np.empty(b0.size * 4, dtype=np.float32)
        interleaved[0::4] = b0
        interleaved[1::4] = b1
        interleaved[2::4] = b2
        interleaved[3::4] = b3
        w = (interleaved[:total] - zero_point) * scale

    elif prec_id == 16:
        w = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)

    else:
        raise ValueError(f"Unknown precision id: {prec_id}")

    return w.reshape(rows, cols)


# ─── Core scheduler ───────────────────────────────────────────────────────────

class N730Scheduler:
    """
    3-stage pipeline scheduler for .n730 files.

    Stage 1 (disk thread)   — persistent file handle, sequential reads
    Stage 2 (dequant thread) — CPU float conversion, overlapped with next disk read
    Stage 3 (caller)        — consumes fully-ready float32 tensors

    The pipeline keeps prefetch layers buffered so the caller (GPU)
    never has to wait for disk I/O.
    """

    def __init__(self, model_path: str, prefetch: int = 4):
        self.model_path = Path(model_path)
        self.prefetch = prefetch
        self.stats = SchedulerStats()
        self._load_header()

    def _load_header(self):
        with open(self.model_path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"Not a .n730 file (magic: {magic!r})")
            version, header_size = struct.unpack(">II", f.read(8))
            header_json = f.read(header_size)
        self._header = json.loads(header_json)
        self._seek_table = self._header["seek_table"]
        print(f"  N730 scheduler ready")
        print(f"  Model   : {self._header['model_id']}")
        print(f"  Layers  : {self._header['total_layers']}")
        print(f"  Size    : {self._header['stored_mb']} MB  ({self._header['compression']}× compressed)")
        print(f"  Prefetch: {self.prefetch} layers")

    def stream(self, start: int = 0, end: Optional[int] = None) -> Iterator[LayerBuffer]:
        """
        Stream layers through the 3-stage pipeline.

        Yields LayerBuffer objects (float32 weights, ready for GPU).
        The pipeline runs ahead by `self.prefetch` layers.
        """
        if end is None:
            end = len(self._seek_table)

        # Two inter-thread queues
        raw_q: queue.Queue = queue.Queue(maxsize=self.prefetch + 2)
        ready_q: queue.Queue = queue.Queue(maxsize=self.prefetch + 1)

        # ── Stage 1: disk thread ──────────────────────────────────────────────
        def disk_thread():
            # ONE file handle for the entire run — eliminates per-layer open() cost
            with open(self.model_path, "rb") as f:
                for i in range(start, end):
                    entry = self._seek_table[i]
                    t0 = time.perf_counter()

                    f.seek(entry["file_offset"])
                    layer_magic = f.read(4)
                    if layer_magic != LAYER_MAGIC:
                        raise ValueError(f"Bad layer magic at layer {i}")

                    idx, prec_id, rows, cols, scale, zp, data_size = struct.unpack(
                        ">IBIIffi", f.read(LAYER_HDR_SIZE)
                    )
                    raw_bytes = f.read(data_size)
                    disk_ms = (time.perf_counter() - t0) * 1000

                    raw_q.put(RawLayerBlock(
                        layer_idx=idx,
                        layer_name=entry["layer_name"],
                        precision=entry["precision"],
                        prec_id=prec_id,
                        rows=rows,
                        cols=cols,
                        scale=scale,
                        zero_point=zp,
                        raw_bytes=raw_bytes,
                        disk_time_ms=disk_ms,
                    ))
            raw_q.put(None)  # sentinel

        # ── Stage 2: dequant thread ───────────────────────────────────────────
        def dequant_thread():
            while True:
                raw = raw_q.get()
                if raw is None:
                    ready_q.put(None)
                    return

                t0 = time.perf_counter()
                weights = dequantize(
                    raw.raw_bytes, raw.prec_id,
                    raw.rows, raw.cols,
                    raw.scale, raw.zero_point,
                )
                dequant_ms = (time.perf_counter() - t0) * 1000

                ready_q.put(LayerBuffer(
                    layer_idx=raw.layer_idx,
                    layer_name=raw.layer_name,
                    weights=weights,
                    precision=raw.precision,
                    load_time_ms=raw.disk_time_ms,
                    dequant_time_ms=dequant_ms,
                ))

                # Update shared stats (GIL keeps this safe)
                self.stats.total_layers += 1
                self.stats.total_disk_ms += raw.disk_time_ms
                self.stats.total_dequant_ms += dequant_ms
                self.stats.total_bytes_read += len(raw.raw_bytes)

        t1 = threading.Thread(target=disk_thread, daemon=True)
        t2 = threading.Thread(target=dequant_thread, daemon=True)
        t1.start()
        t2.start()

        # ── Stage 3: caller (GPU) ─────────────────────────────────────────────
        while True:
            t_wait = time.perf_counter()
            layer = ready_q.get()
            wait_ms = (time.perf_counter() - t_wait) * 1000

            if layer is None:
                break

            depth = ready_q.qsize()
            if depth > self.stats.peak_ram_layers:
                self.stats.peak_ram_layers = depth

            if wait_ms < 2.0:
                self.stats.prefetch_hits += 1
            else:
                self.stats.prefetch_misses += 1
                self.stats.gpu_wait_ms += wait_ms

            yield layer

        t1.join()
        t2.join()

    def get_layer(self, layer_idx: int) -> LayerBuffer:
        """Random access — O(1) seek to any single layer."""
        entry = self._seek_table[layer_idx]
        t0 = time.perf_counter()
        with open(self.model_path, "rb") as f:
            f.seek(entry["file_offset"])
            f.read(4)  # layer magic
            idx, prec_id, rows, cols, scale, zp, data_size = struct.unpack(
                ">IBIIffi", f.read(LAYER_HDR_SIZE)
            )
            raw_bytes = f.read(data_size)
        disk_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        weights = dequantize(raw_bytes, prec_id, rows, cols, scale, zp)
        dequant_ms = (time.perf_counter() - t0) * 1000

        return LayerBuffer(
            layer_idx=idx,
            layer_name=entry["layer_name"],
            weights=weights,
            precision=entry["precision"],
            load_time_ms=disk_ms,
            dequant_time_ms=dequant_ms,
        )

    def print_stats(self):
        s = self.stats
        bottleneck = "disk" if s.avg_disk_ms > s.avg_dequant_ms else "dequant"
        print("\n" + "═" * 55)
        print("  N730 SCHEDULER STATS")
        print("═" * 55)
        print(f"  Layers streamed   : {s.total_layers}")
        print(f"  Data read         : {s.total_bytes_read / (1024**2):.1f} MB")
        print(f"  Avg disk time     : {s.avg_disk_ms:.2f} ms/layer")
        print(f"  Avg dequant time  : {s.avg_dequant_ms:.2f} ms/layer")
        print(f"  Throughput        : {s.throughput_mb_s:.1f} MB/s")
        print(f"  Bottleneck        : {bottleneck}")
        print(f"  Prefetch hit rate : {s.hit_rate * 100:.1f}%")
        print(f"  GPU wait total    : {s.gpu_wait_ms:.1f} ms")
        print(f"  Peak RAM layers   : {s.peak_ram_layers}")
        if s.prefetch_misses == 0:
            print(f"  Pipeline stalls   : 0  ✓ GPU never waited")
        else:
            print(f"  Pipeline stalls   : {s.prefetch_misses}")
            if bottleneck == "disk":
                print(f"  Tip: SSD limited. Try --prefetch {self.prefetch + 4}")
            else:
                print(f"  Tip: CPU dequant limited. Try --prefetch {self.prefetch + 2}")
        print("═" * 55)


# ─── Simulated forward pass ───────────────────────────────────────────────────

def simulated_forward_pass(layer: LayerBuffer, activations: np.ndarray,
                            simulate_ms: float = 0.0) -> np.ndarray:
    """
    Placeholder for the CUDA forward pass kernel (Phase 4).
    simulate_ms adds artificial delay to mimic real GPU compute time,
    which lets us properly test whether the prefetch pipeline stays ahead.
    """
    if simulate_ms > 0:
        time.sleep(simulate_ms / 1000)

    w = layer.weights
    in_dim = w.shape[1]
    if activations.shape[0] != in_dim:
        activations = np.ones(in_dim, dtype=np.float32) * 0.01
    return np.tanh(w @ activations)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def run_benchmark(model_path: str, prefetch: int, n_layers: Optional[int],
                  simulate_gpu_ms: float = 0.0):
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║  N730 Scheduler  •  Project Bombakla  ║")
    print(f"╚══════════════════════════════════════╝\n")

    scheduler = N730Scheduler(model_path, prefetch=prefetch)
    total = len(scheduler._seek_table)
    end = min(n_layers, total) if n_layers else total

    gpu_label = f" + {simulate_gpu_ms:.0f}ms simulated GPU" if simulate_gpu_ms else ""
    print(f"\n  Benchmarking {end}/{total} layers  prefetch={prefetch}{gpu_label}\n")

    activations = np.ones(1024, dtype=np.float32) * 0.01
    bar_width = 40
    t_total = time.perf_counter()

    for layer in scheduler.stream(end=end):
        activations = simulated_forward_pass(layer, activations, simulate_gpu_ms)

        done = layer.layer_idx + 1
        filled = int(bar_width * done / end)
        bar = "█" * filled + "░" * (bar_width - filled)
        hit_pct = scheduler.stats.hit_rate * 100
        status = "✓" if hit_pct >= 80 else ("~" if hit_pct >= 40 else "⚠")
        print(
            f"  [{bar}] {done:>3}/{end}  "
            f"[{layer.precision:<4}]  "
            f"disk={layer.load_time_ms:5.1f}ms  "
            f"dq={layer.dequant_time_ms:4.1f}ms  "
            f"hit={hit_pct:.0f}%{status}",
            end="\r"
        )

    elapsed = time.perf_counter() - t_total
    print(f"\n\n  Total wall time: {elapsed:.2f}s")
    scheduler.print_stats()


def read_single_layer(model_path: str, layer_idx: int):
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║  N730 Scheduler  •  Project Bombakla  ║")
    print(f"╚══════════════════════════════════════╝\n")
    scheduler = N730Scheduler(model_path, prefetch=1)
    print(f"\n  Reading layer {layer_idx}...")
    layer = scheduler.get_layer(layer_idx)
    print(f"  Name         : {layer.layer_name}")
    print(f"  Precision    : {layer.precision}")
    print(f"  Shape        : {layer.weights.shape}")
    print(f"  Mean / Std   : {layer.weights.mean():.6f} / {layer.weights.std():.6f}")
    print(f"  Min / Max    : {layer.weights.min():.4f} / {layer.weights.max():.4f}")
    print(f"  Disk read    : {layer.load_time_ms:.2f} ms")
    print(f"  Dequant      : {layer.dequant_time_ms:.2f} ms")
    print(f"  Total        : {layer.load_time_ms + layer.dequant_time_ms:.2f} ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="N730 Layer Scheduler — Project Bombakla Phase 3")
    parser.add_argument("--model", type=str, default="model.n730")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--layers", type=int, help="Limit to first N layers")
    parser.add_argument("--layer", type=int, help="Inspect single layer by index")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--simulate-gpu-ms", type=float, default=0.0,
                        help="Simulate GPU compute time per layer (ms) to test pipeline balance")
    args = parser.parse_args()

    if args.layer is not None:
        read_single_layer(args.model, args.layer)
    else:
        run_benchmark(args.model, args.prefetch, args.layers, args.simulate_gpu_ms)