import os
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import psutil
import thop
import torch
import torch.nn as nn

def profile_model_efficiency(
    model: nn.Module,
    model_name: str,
    device: torch.device = torch.device("cpu"),
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, Any]:
    """Profile FLOPs, parameters, peak RAM, inference latency (bs=1, bs=16), and throughput on CPU."""
    model.eval()
    model.to(device)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs calculation using thop
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    try:
        flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"[Profiler Warning] thop profiling error: {e}")
        flops = 0.0

    # 3. Memory usage tracking
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    # 4. Latency BS=1
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    latencies_bs1 = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_bs1.append((t1 - t0) * 1000.0) # in ms

    mean_lat_bs1 = float(np.mean(latencies_bs1))
    p95_lat_bs1 = float(np.percentile(latencies_bs1, 95))
    throughput_bs1 = float(1000.0 / mean_lat_bs1)

    # 5. Latency BS=16
    dummy_bs16 = torch.randn(16, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_bs16)

    latencies_bs16 = []
    with torch.no_grad():
        for _ in range(25):
            t0 = time.perf_counter()
            _ = model(dummy_bs16)
            t1 = time.perf_counter()
            latencies_bs16.append((t1 - t0) * 1000.0)

    mean_lat_bs16 = float(np.mean(latencies_bs16))
    p95_lat_bs16 = float(np.percentile(latencies_bs16, 95))
    throughput_bs16 = float((16 * 1000.0) / mean_lat_bs16)

    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = max(ram_before_mb, ram_after_mb)

    stats = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_million": round(total_params / 1e6, 3),
        "gmacs": round(flops / 1e9, 4),
        "mflops": round(flops / 1e6, 2),
        "bs1_latency_mean_ms": round(mean_lat_bs1, 2),
        "bs1_latency_p95_ms": round(p95_lat_bs1, 2),
        "bs16_latency_mean_ms": round(mean_lat_bs16, 2),
        "bs16_latency_p95_ms": round(p95_lat_bs16, 2),
        "throughput_img_per_s": round(throughput_bs16, 2),
        "ram_usage_mb": round(peak_ram_mb, 2)
    }
    return stats

if __name__ == "__main__":
    from src.models.model_factory import create_model
    for m in ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0"]:
        net = create_model(m)
        stats = profile_model_efficiency(net, m)
        print(f"[{m}] Params: {stats['params_million']}M | GMACs: {stats['gmacs']} | Latency BS=1: {stats['bs1_latency_mean_ms']}ms (p95: {stats['bs1_latency_p95_ms']}ms) | Throughput: {stats['throughput_img_per_s']} img/s")
