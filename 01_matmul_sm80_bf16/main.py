# Benchmark harness adapted from gau-nernst/learn-cuda 02_matmul_sm80/main.py
# (commit afcb20b). Changes: runs through ../modal/run.py instead of its own Modal
# launcher; kernel list comes from --kernels; cutedsl kernels dropped.
#
#   uv run modal run ../modal/run.py --target main.py --gpu A100-40GB --args "--kernels cublas matmul_v1"
#   ... --args "--sweep 2048 4096 8192"
#   ... --args "--profile matmul_v1"     # single call, for ncu

import argparse
import math
import multiprocessing as mp
import subprocess
from pathlib import Path

import cuda.bench
import pandas as pd
import torch
import torch._inductor.config
import torch.utils.cpp_extension

CURRENT_DIR = Path(__file__).parent


def get_module(path):
    torch.utils.cpp_extension.load(
        "my_module",
        sources=list(Path(path).glob("matmul*")),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        is_python_module=False,
        verbose=True,
    )
    return torch.ops.my_module


def get_sol():
    gpu_name = torch.cuda.get_device_name()
    sol = 1e9
    if "A100" in gpu_name:
        sol = 312
    return sol


def get_kernel(name: str, source_dir: str):
    if name == "cublas":
        f = torch.mm
    elif name == "inductor":
        torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        f = torch.compile(torch.mm, mode="max-autotune-no-cudagraphs", dynamic=False)
    else:
        module = get_module(source_dir)
        f = getattr(module, name)
    return f


def to_torch_stream(s: cuda.bench.CudaStream, device: int | None):
    return torch.cuda.ExternalStream(stream_ptr=s.addressof(), device=device)


def torch_bench(state: cuda.bench.State) -> None:
    # state.set_throttle_threshold(0.25)
    device = state.get_device()

    # select kernel
    f = get_kernel(state.get_string("kernel"), state.get_string("source_dir"))

    # problem shape
    M = state.get_int64("M")
    N = state.get_int64("N")
    K = state.get_int64("K")

    stream = to_torch_stream(state.get_stream(), device)
    with torch.cuda.stream(stream):
        scale = K**-0.5  # make sure output doesn't explode
        A = torch.randn(M, K, device="cuda").mul(scale).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(scale).bfloat16().T

        # compute in FP32 to avoid split-K
        out_ref = torch.mm(A.float(), B.float()).bfloat16()
        out = f(A, B)
        torch.testing.assert_close(out, out_ref)

        # distinct inputs per call, so repeated calls don't hit a warm L2
        inputs_list = []
        for _ in range(state.get_int64("num_inputs")):
            A = torch.randn(M, K, device=device).mul(scale).bfloat16()
            B = torch.randn(N, K, device=device).mul(scale).bfloat16().T
            inputs_list.append((A, B))

    def launcher(launch: cuda.bench.Launch) -> None:
        stream = to_torch_stream(launch.get_stream(), device)
        with torch.cuda.stream(stream):
            for A, B in inputs_list:
                f(A, B)

    state.exec(launcher, sync=True)


def run_nvbench(M: int, N: int, K: int, source_dir: str, kernels_list: list[str]):
    # duplicate inputs to make sure each measurement is at least 10ms
    SOL = get_sol()
    min_latency_ms = 2 * M * N * K / (SOL * 1e12) * 1e3
    num_inputs = math.ceil(10 / min_latency_ms)

    bench = cuda.bench.register(torch_bench)
    bench.add_string_axis("source_dir", [source_dir])
    bench.add_string_axis("kernel", kernels_list)
    bench.add_int64_axis("M", [M])
    bench.add_int64_axis("N", [N])
    bench.add_int64_axis("K", [K])
    bench.add_int64_axis("num_inputs", [num_inputs])

    result_path = "/tmp/result.csv"
    cuda.bench.run_all_benchmarks(["--csv", result_path])

    df = pd.read_csv(result_path)
    df["GPU Time (sec)"] /= num_inputs
    df["latency (us)"] = df["GPU Time (sec)"] * 1e6
    df["TFLOPS"] = 2 * M * N * K / df["GPU Time (sec)"] * 1e-12
    df["% SOL"] = df["TFLOPS"] / SOL

    # apply formatting
    df["latency (us)"] = df["latency (us)"].map("{:.2f}".format)
    df["TFLOPS"] = df["TFLOPS"].map("{:.2f}".format)
    df["% SOL"] = df["% SOL"].map("{:.2%}".format)
    df["Noise"] = df["Noise"].map("{:.2%}".format)

    print()
    print(df[["kernel", "latency (us)", "Noise", "TFLOPS", "% SOL"]].to_markdown(index=False))


def main(args: argparse.Namespace):
    source_dir = str(CURRENT_DIR)

    subprocess.run(["nvidia-smi"], check=True)

    if args.profile is not None:
        M, N, K = args.shape
        scale = K**-0.5  # make sure output doesn't explode
        A = torch.randn(M, K, device="cuda").mul(scale).bfloat16()
        B = torch.randn(N, K, device="cuda").mul(scale).bfloat16().T

        f = get_kernel(args.profile, source_dir)
        f(A, B)
        return

    if args.sweep is not None:
        shapes = [(s, s, s) for s in args.sweep]
    else:
        assert len(args.shape) == 3
        shapes = [args.shape]

    mp_context = mp.get_context("spawn")

    for M, N, K in shapes:
        # nvbench can only be run once per process. hence, we spawn subprocess for each shape.
        proc = mp_context.Process(target=run_nvbench, args=(M, N, K, source_dir, args.kernels))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096, 4096])
    parser.add_argument("--sweep", type=int, nargs="+")
    parser.add_argument("--kernels", nargs="+", default=["cublas", "matmul_v1"])
    main(parser.parse_args())
