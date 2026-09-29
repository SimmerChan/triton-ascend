#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vector_add 全流程演示:编译 → 运行 → 列出每阶段交付件。

用法(NPU 机器):
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python run_vector_add_dump.py

脚本做什么:
  1. import triton 前开启 TRITON_KERNEL_DUMP=1,使各阶段 IR/二进制 dump 到 ~/.triton/dump;
  2. 运行 vector_add kernel 并与 torch 对拍;
  3. 扫描 ~/.triton/cache 与 ~/.triton/dump,打印本次编译的全部交付件及各 IR 文件头。
"""

import os

# 必须在 import triton 之前设置(knobs 惰性读取环境变量,但提前设置最稳妥)
os.environ.setdefault("TRITON_KERNEL_DUMP", "1")

import glob
import time

import torch
import torch_npu  # noqa: F401  注册 npu 设备

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def dump_file_head(path: str, max_lines: int = 25) -> None:
    print(f"\n----- {os.path.basename(path)} ({os.path.getsize(path)} bytes) -----")
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    print(f"  ... (截断,完整内容请查看 {path})")
                    break
                print("  " + line.rstrip())
    except OSError as e:
        print(f"  <无法读取: {e}>")


def newest_cache_entries(root: str, since: float):
    """返回 root 下 mtime 晚于 since 的文件,按目录分组。"""
    result = {}
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.getmtime(path) >= since:
            result.setdefault(os.path.dirname(path), []).append(path)
    return result


def main() -> None:
    print_header("STEP 1. 运行 vector_add(kernel 编译 + NPU 执行)")
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    t0 = time.time()
    output_triton = add(x, y)
    torch.npu.synchronize()
    print(f"编译+首次运行耗时: {time.time() - t0:.2f}s")

    output_torch = x + y
    torch.testing.assert_close(output_triton, output_torch)
    print(f"与 torch 对拍通过, max diff = "
          f"{torch.max(torch.abs(output_torch - output_triton)).item()}")

    print_header("STEP 2. 缓存目录 ~/.triton/cache 中的交付件")
    cache_root = os.path.expanduser(os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache"))
    groups = newest_cache_entries(cache_root, t0 - 5)
    if not groups:
        print("(本次未新增缓存 —— 可能命中旧缓存;可设 TRITON_ALWAYS_COMPILE=1 重跑)")
    for d in sorted(groups):
        print(f"\n[{d}]")
        for f in sorted(groups[d]):
            print(f"  {os.path.basename(f):<48} {os.path.getsize(f):>10} bytes")

    print_header("STEP 3. dump 目录 ~/.triton/dump 中各阶段 IR")
    dump_root = os.path.expanduser(os.environ.get("TRITON_DUMP_DIR", "~/.triton/dump"))
    dump_groups = newest_cache_entries(dump_root, t0 - 5)
    stage_order = ["kernel.ttir.mlir", "kernel.ttadapter.mlir", "kernel.mlirbc",
                   "kernel.mlir", "kernel.npuir.mlir", "kernel.o"]
    shown = False
    for d in sorted(dump_groups):
        files = sorted(dump_groups[d], key=lambda f: stage_order.index(os.path.basename(f))
                       if os.path.basename(f) in stage_order else 99)
        print(f"\n[{d}]  ->  {', '.join(os.path.basename(f) for f in files)}")
        for f in files:
            shown = True
            if f.endswith((".mlir", ".ttir", ".ttadapter")) or os.path.basename(f) in stage_order:
                dump_file_head(f)
    if not shown:
        print("(未发现 dump —— 检查 TRITON_KERNEL_DUMP 是否在 import triton 前生效)")

    print_header("完成:TTIR → ttadapter → mlirbc → bcmlir → npubin 全链路交付件已生成")


if __name__ == "__main__":
    main()
