# Triton DSL → NPU 算子二进制:编译全流程与交付件分析

> 基于 triton-ascend 本地代码仓(commit e3ca11f30,Triton 3.6.0 基线)逐文件核实。
> 所有路径相对仓库根目录;`ascend/compiler.py` 指 `third_party/ascend/backend/compiler.py`。

## 0. 全流程总览

```
@triton.jit Python 函数
    │  jit.py: JITFunction.run — binder 绑定参数 / 计算 cache key
    ▼
triton.compile (python/triton/compiler/compiler.py)
    │  ASTSource.make_ir → ast_to_ttir (Python AST → TTIR)
    ▼
┌─ stage 1  ttir      make_ttir                    → {name}.ttir        优化后 TTIR (MLIR 文本)
├─ stage 2  ttadapter ttir_to_linalg              → {name}.ttadapter   Ascend Linalg/HIVM IR (MLIR 文本)
├─ stage 3  mlirbc    linalg_to_bc_by_triton_mlir_opt → {name}.mlirbc   MLIR 字节码 (二进制)
├─ stage 4  bcmlir    bc_to_linalg_by_bishengir_opt → {name}.bcmlir     毕昇侧回读 MLIR 文本
└─ stage 5  npubin    linalg_to_bin_enable_npu_compile_* → {name}.npubin ★算子二进制 (kernel.o 字节)
    │  CompiledKernel._init_handles
    ▼
加载:  npu_utils.so → aclrtBinaryLoadFromData / rtDevBinaryRegister (设备侧注册)
启动:  launcher so (__triton_launcher) → aclrtLaunchKernelWithHostArgs / rtKernelLaunch
```

- 纯 SIMT 分支(仅 Ascend910_95/950 且 `compile_mode="simt_only"`):stage 2-4 被跳过,`ttir → npubin` 由 `ttir_to_npubin` 直接完成(ascend/compiler.py:1589-1591)。
- 本流水线**没有** NVIDIA 路径的 `.ttgir/.llir/.ptx/.cubin`,也没有 `.npucbin`;二进制交付件集合为 `binary_extensions = {"npubin", "mlirbc"}`(ascend/compiler.py:1514)。

## 1. 阶段 A:JIT 入口与参数绑定(python/triton/runtime/jit.py)

| 步骤 | 位置 | 说明 |
|------|------|------|
| `kernel[grid](...)` | jit.py:364-370 | `KernelInterface.__getitem__` 转为 `run(grid=..., warmup=False, ...)` |
| 取设备/流 | jit.py:700-701 | `driver.active.get_current_device()/get_current_stream()` |
| binder 绑定 | jit.py:710 | `create_function_from_signature`(jit.py:391-448)动态生成的 binder 把实参分为 constexpr 与非常量参数,非常量参数经 `native_specialize_impl` 产出`(类型串, 特化值)`(对齐 16、值=1 等特化) |
| cache key | jit.py:712-713, 563-584 | `str(specialization)+str(options)`;命中 `kernel_cache` 则直接复用 `CompiledKernel` |
| 打包编译输入 | jit.py:671-693 | `_pack_args`:`backend.parse_options`(构造 `NPUOptions`)、`signature`、`constexprs`、`attrs` |
| 触发编译 | jit.py:826-853 | `ASTSource(self, signature, constexprs, attrs)` → `triton.compile` |

**本阶段交付件(内存对象,不落盘)**:signature、constexprs、attrs(特化)、NPUOptions、cache key。
cache key = `triton_key() - src.hash() - backend.hash() - options.hash() - env_vars`(compiler.py:245-248),其中 `options.hash()` 额外混入 CANN 版本文件哈希(ascend/compiler.py:1360-1363)。

## 2. 阶段 B:triton.compile 主控(python/triton/compiler/compiler.py)

1. `make_backend(target)`(compiler.py:363-368):`target.backend == "npu"` 时命中 `AscendBackend`。
2. `src.make_ir(...)`(compiler.py:304 → 78-81):`code_generator.ast_to_ttir` 把 Python AST 翻译为 TTIR。
3. **初始 TTIR 落盘**:`{file_name}.source`(compiler.py:313-314)。
4. `backend.add_stages(stages, options, language)`(compiler.py:287-288)注册 6 个阶段;从 `src.ext` 对应阶段开始顺序执行(compiler.py:323-349),每个阶段产物写 `{file_name}.{ext}`。
5. metadata 写盘:`{file_name}.json` + 组索引 `__grp__{file_name}.json`(compiler.py:350-353)。
6. 返回 `CompiledKernel`(compiler.py:360)。

## 3. 编译五阶段详解(ascend/compiler.py `add_stages`,1586-1605)

### Stage 1 `ttir` — TTIR 优化
- 实现:`make_ttir`(ascend/compiler.py:297-321),在进程内用 `ir.pass_manager` 跑通用 pass:inliner → combine → canonicalizer → reorder_broadcast → CSE → LICM → symbol-dce → loop-unroll,可选 `add_graph_optimize`(Ascend 图优化,ub 预算感知)。
- 交付件:**`{name}.ttir`** — 优化后 TTIR,MLIR 文本。示例骨架:

```mlir
tt.func public @add_kernel(%arg0: memref<...f32> ..., %arg1: ..., %arg2: ..., %arg3: i32) {
  %0 = tt.get_program_id x : i32
  // tt.load / arith.addf / tt.store ...
}
```

### Stage 2 `ttadapter` — 降到 Ascend Linalg/HIVM IR(核心 fork 阶段)
- 实现:`ttir_to_linalg`(ascend/compiler.py:324-441)。虽然函数内构建了完整 pass pipeline,但实际通过进程内 C++ 绑定 `ascend.passes.ttir.*` 执行,等价命令行为 `triton-adapter-opt --pass-pipeline=...`(debug 模式会打印该命令,415-427):
  `TritonControlFlowOpt → TritonToStructure → DiscreteMaskAccessConversion → TritonToAnnotation → TritonToUnstructure → TritonToHIVM → TritonToHFusion → TritonToLLVM → BubbleUpOperation → TritonToStructure → TritonToLinalg`(可选 `DynamicCVPipeline`/merge_concat_load_buffer,386-398)。
- 之后 `_parse_linalg_metadata`(527-599)用正则从 IR 提取 `mix_mode`(`aiv`/`mix`/`aic`)、`parallel_mode`、`kernel_name`、`tensor_kinds`、bitcodes 等运行期元数据;`_export_program_grid_metadata`(163-190)导出 launcher 所需的 grid 协变信息(`hacc.coalesce_factor` 等)。
- 交付件:**`{name}.ttadapter`** — linalg/hivm 方言 MLIR 文本,含 `mix_mode = "aiv"` 等模块属性。这是毕昇编译器的输入格式。

### Stage 3 `mlirbc` — 序列化为字节码
- 实现:`linalg_to_bc_by_triton_mlir_opt`(ascend/compiler.py:442-477),外部命令 `triton-mlir-opt input --emit-bytecode -o kernel.mlirbc`。
- 交付件:**`{name}.mlirbc`** — MLIR Bytecode(二进制)。它也是 `binary_extensions` 之一,可跨版本缓存放置。

### Stage 4 `bcmlir` — 毕昇侧回读
- 实现:`bc_to_linalg_by_bishengir_opt`(ascend/compiler.py:480-515),外部命令 `bishengir-opt kernel.mlirbc --mlir-print-debuginfo -o kernel.mlir`。
- 交付件:**`{name}.bcmlir`** — 由毕昇工具链转回的 MLIR 文本(可能含毕昇侧标注),作为 stage 5 的实际输入。

### Stage 5 `npubin` — 生成算子二进制 ★
- 实现:`linalg_to_bin_enable_npu_compile_A2_A3`(ascend/compiler.py:924-1122,Atlas A2/A3)或 `..._910_95`(697-921,Ascend 910_95/950)。
- 外部命令(1083/1457):`bishengir-compile kernel.mlir --target=<arch> --enable-hfusion-compile=true --enable-triton-kernel-compile=true [--reg-based=true|--enable-hivm-compile=true] -o kernel`。
  工具定位:`_get_npucompiler_path`(backend/utils.py:495-509),按 `$ASCEND_HOME_PATH/bishengir/bin` → `PATH` → `$TRITON_NPU_COMPILER_PATH` 查找。
- 关键产物判定:`kernel.o`(新版 bishengir-compile)或 `kernel_reloc.o`(旧版,由 `_check_bishengir_api_change` 探测,utils.py:535-556);以 `Path(bin_path).read_bytes()` 返回(1122)。
- 附带效果:bishengir-compile 同目录若生成 `libkernel.so`,则 `ctypes.CDLL` 加载并调用其导出函数补充元数据(`_infer_task_type_function`→`bs_task_type`、`_infer_workspace_shape_function`→`workspace_size`、sync_block_lock 布局,913-919);stdout 中的 `UB size = N bits` 写入 `required_ub_bits`(898-902)。
- 交付件:**`{name}.npubin`** — NPU 设备可执行 ELF 二进制(kernel.o 字节),**这就是"算子二进制"**。

### 分支:纯 SIMT 路径(910_95/950 专属)
`compile_mode="simt_only"` 时 `ttir_to_npubin`(ascend/compiler.py:1412-1464)直接把 TTIR 交给 bishengir-compile,附加 `--pure-simt --num-warps --threads-per-warp --simt-stack-limit ...`,跳过 stage 2-4。适用 `al.parallel`/SIMT 编程模型 kernel。

## 4. 阶段 C:加载与启动(运行期)

| 步骤 | 实现 | 说明 |
|------|------|------|
| host 对象初始化 | `CompiledKernel._init_handles`(compiler.py:436-471) | `launcher_cls = NPULauncher`;`utils.load_binary`(patch 后传 `kernel_name, kernel, shared, device, mix_mode`,见 patch/triton-ascend-3.6.0.patch:700-702) |
| 设备侧注册 | `NPUUtils.load_binary` → npu_utils.cpp:202-217 `loadKernelBinary` | CANN ≥9.1:`aclrtBinaryLoadFromData` + `aclrtBinaryGetFunction`(magic 按 aiv/aicore 选 ELF 类型,npu_utils.cpp:94-113);旧 CANN:`rtDevBinaryRegister` + `rtFunctionRegister`(156-187)。得到 function handle → `kernel.function` |
| launcher 编译 | `make_npu_launcher_stub`(backend/driver.py:328-374) | 生成 `__triton_launcher` C++ 源码(`generate_npu_header_src` 445-524 + `make_launcher` 737 起),cache key=sha256(源码),用 host 编译器(`$CC`/clang++/g++,utils.py:623-631)编成 Python 扩展 so,**链接 `-lruntime -lascendcl`** |
| kernel 启动 | launcher so 内 `launch`(METH_FASTCALL)→ `_launch`(driver.py:1477-1508 打包 ffts/workspace/args)→ `cann_launch_kernel`(478-480) | CANN 9.0+:`aclrtLaunchKernelWithHostArgs`;旧:`rtKernelLaunchWithFlagV2`/`rtKernelLaunch`。grid 展开为 1D `blockNum = gridX*gridY*gridZ` |

**host 侧交付件(两个独立缓存条目)**:
1. `launcher_cxx11abi{0|1}<py-ext-suffix>.so` — `__triton_launcher` 扩展(缓存 key 为源码 sha256);
2. `npu_utils.so` — 由 `third_party/ascend/backend/npu_utils.cpp` 编出,负责设备属性查询、kernel 注册、workspace/syncBlockLock 分配(driver.py:48-106;缓存 key=`md5(cann_version+torch_npu_version+源码)`,70-92)。

## 5. 磁盘交付件完整清单

一次全新编译后,`$TRITON_CACHE_DIR`(默认 `~/.triton/cache`)下的 `<base32(sha256)>/` 目录:

| 文件 | 阶段 | 格式 | 写入点 |
|------|------|------|--------|
| `{name}.source` | ast_to_ttir | 未优化 TTIR(MLIR 文本) | compiler.py:313-314 |
| `{name}.ttir` | stage 1 | 优化后 TTIR | compiler.py:335-336 |
| `{name}.ttadapter` | stage 2 | linalg/hivm MLIR 文本 | 同上 |
| `{name}.mlirbc` | stage 3 | MLIR 字节码(二进制) | 同上 |
| `{name}.bcmlir` | stage 4 | 毕昇回读 MLIR 文本 | 同上 |
| `{name}.npubin` | stage 5 | **设备 ELF 二进制** | 同上 |
| `{name}.json` | — | metadata(hash/target/options/kernel_name/mix_mode/tensor_kinds/…) | compiler.py:261, 350-352 |
| `__grp__{name}.json` | — | 组索引(child_paths,cache 命中判定) | cache.py:91-96 |

另有两个独立 hash 目录存放 host so(launcher、npu_utils)。
`TRITON_KERNEL_DUMP=1` 时,各阶段产物会额外以 `kernel.*.mlir`、`kernel.o` 等名字写入 `$TRITON_DUMP_DIR/<hash>/`(默认 `~/.triton/dump`;ascend/compiler.py:317-319, 435-437, 473-475, 511-513, 911;driver.py:339-344)。

## 6. 实际例子:vector_add 全流程验证

配套脚本 [run_vector_add_dump.py](run_vector_add_dump.py) 在 tutorials/01-vector-add.py 的 kernel 基础上:
1. 编译前设置 `TRITON_KERNEL_DUMP=1`(IR/二进制 dump 到 `~/.triton/dump`);
2. 运行 kernel 并与 torch 对拍;
3. 扫描 `~/.triton/cache` 与 `~/.triton/dump`,列出本次编译产生的全部交付件并打印各 IR 文件头。

### 远程执行步骤(NPU 机器)

```bash
# 环境:CANN 9.1.0 + torch_npu 2.7.1 + Python 3.11(参考 docs/en/installation_guide.md)
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 必须,提供 ASCEND_HOME_PATH 与 bishengir-*
python run_vector_add_dump.py
# 或运行官方教程 / pytest:
python third_party/ascend/tutorials/01-vector-add.py
pytest third_party/ascend/unittest/pytest_ut/test_01_vector_add.py
```

成功后控制台会打印 cache 目录文件表;`~/.triton/dump/<hash>/` 下可用 `less` 逐阶段查看
`kernel.ttir.mlir → kernel.ttadapter.mlir → kernel.mlirbc(二进制) → kernel.mlir → kernel.o(ELF)`。

### IR 对照(同一行 `output = x + y` 在各阶段的样子)

| 阶段 | 该运算的表示 |
|------|--------------|
| `.ttir` | `%12 = arith.addf %10, %11 : tensor<1024xf32>`(张量语义,与硬件无关) |
| `.ttadapter` | linalg/hivm 块内 `hivm.*` 指令序列(已绑定 AIV/UB、tile 形状与 `mix_mode = "aiv"`) |
| `.npubin` | Vector 核机器码(ELF,`aclrtBinaryGetFunction` 按名 `add_kernel` 取入口) |

## 7. 常用环境变量

| 变量 | 作用 |
|------|------|
| `TRITON_CACHE_DIR` | 内核/launcher/npu_utils 缓存根(默认 `~/.triton/cache`) |
| `TRITON_KERNEL_DUMP=1` | 各阶段 IR/二进制 dump 到 `TRITON_DUMP_DIR` |
| `TRITON_ALWAYS_COMPILE=1` | 跳过 json 缓存命中,强制重编译 |
| `MLIR_ENABLE_DUMP=1` | 打印每个 MLIR pass 后的 IR(注意:会改变 cache key) |
| `TRITON_KERNEL_OVERRIDE` | 用同名 IR 文件替换某阶段产物(ir_override) |
| `TRITON_COMPILE_ONLY=1` | 只编译不启动(Ascend 扩展) |
| `ASCEND_HOME_PATH` | 必须 source `set_env.sh` 设置,定位 bishengir 工具链(backend/utils.py:582-586) |
| `TRITON_NPU_COMPILER_PATH` | bishengir-opt/compile/bisheng 的备用查找路径 |
