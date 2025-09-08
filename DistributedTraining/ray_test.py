import time
import torch
import ray
import numpy as np

def heavy_compute_cpu(x):
    start = time.time()
    size = 10240
    a = torch.rand(size, size)
    b = torch.rand(size, size)
    for _ in range(5):
        a = torch.matmul(a, b)
    result = (a[0, 0] + x).item()
    compute_time = time.time() - start
    return result, compute_time

@ray.remote(num_gpus=1, runtime_env={"pip": ["torch"]})
def heavy_compute_gpu(x):
    start = time.time()
    device = torch.device("cuda")
    size = 10240
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)
    for _ in range(5):
        a = torch.matmul(a, b)
    result = (a[0, 0] + x).item()
    compute_time = time.time() - start
    return result, compute_time

def run_locally(n_tasks=4):
    print("Running on local CPU...")
    start = time.time()
    results_with_times = [heavy_compute_cpu(i) for i in range(n_tasks)]
    duration = time.time() - start
    print(f"Local wall-clock time: {duration:.2f}s")
    results, times = zip(*results_with_times)
    print(f"Average local CPU compute time: {np.mean(times):.2f}s")
    return results

def run_ray(n_tasks=4):
    print("Running on Ray cluster with GPU tasks...")
    start = time.time()
    futures = [heavy_compute_gpu.remote(i) for i in range(n_tasks)]
    results_with_times = ray.get(futures)
    duration = time.time() - start
    print(f"Ray wall-clock time: {duration:.2f}s")
    results, times = zip(*results_with_times)
    print(f"Average GPU compute time: {np.mean(times):.2f}s")
    return results

if __name__ == "__main__":
    ray.init("ray://localhost:10001")  # Change URI to your cluster address

    n_tasks = 4

    local_results = run_locally(n_tasks)
    ray_results = run_ray(n_tasks)

    diff = np.mean(np.abs(np.array(local_results) - np.array(ray_results)))
    print("Result difference (avg):", diff)

    ray.shutdown()
