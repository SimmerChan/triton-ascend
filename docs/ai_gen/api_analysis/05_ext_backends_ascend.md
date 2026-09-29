# 05 · 扩展 API:`triton.backends.ascend` + kernels + 主包 fork 新增 + 补丁注入

## 1. `triton.backends.ascend`(后端包,来源 `third_party/ascend/backend/`)

### 1.1 基准测试(`triton.backends.ascend.testing`)

| API | 功能备注 |
|---|---|
| `do_bench_npu(funcs, warmup=5, active=30, clear_l2_cache=..., ...)` | NPU kernel 基准测试,返回毫秒均值;mspti 优先,回退 `torch_npu.profiler`;是 GPU `do_bench` 的昇腾替代 |
| `do_bench_npu_profiler(...)` | 基于 `torch_npu.profiler` 的底层计时路径 |
| `do_bench_npu_mspti(...)` | 基于 mspti 的底层计时路径 |
| `ProfilerResultMismatchError` | profiler 结果校验异常 |
| `triton.backends.ascend.do_bench_npu` | 顶层 re-export(`__init__.py`) |

### 1.2 自动调优(`triton.backends.ascend.runtime`)

| API | 功能备注 |
|---|---|
| `import triton.backends.ascend.runtime`(副作用) | 自动把 `triton.autotune` 替换为昇腾版、给 `triton.Config` 增加 `ubtune_cfg` 磁盘缓存读写、把 `TRITON_ENABLE_UBTUNER` 加入缓存失效 env |
| `autotune(...)`(昇腾版) | 在原生 autotune 基础上新增 `auto_prof_dir=`(最优 config 的 profiling 输出目录)与 `hints=`(传给 AutoTilingTuner 的提示 dict) |
| `max_autotune(...)` | 在 autotune 之上按 tuning_params(每个值为 list)做笛卡尔积展开 config;`kernel_type ∈ {"cube","mixcv","vector"}` |
| `get_max_configs(...)` | 单个 base Config × 调参列表展开;可调参数集:`num_stages/unit_flag/multibuffer/limit_auto_multi_buffer_only_for_local_buffer/limit_auto_multi_buffer_of_local_buffer/set_workspace_multibuffer/enable_hivm_auto_cv_balance/tile_mix_vector_loop/tile_mix_cube_loop/enable_ubuf_saving` |
| `get_autotune_cube_config` / `get_autotune_cv_config` / `get_autotune_vector_config` | 按算子类型(Cube/CV 融合/Vector)生成默认调参 config |
| `AutoTilingTuner` | 继承 Autotuner 的自动切分调优器 |
| `@ubtuner(...)`(`runtime/ubtuner.py`) | UB 溢出自动调优装饰器(须放在 `@autotune` 外层;env `TRITON_ENABLE_UBTUNER` 启用);贪心搜索 bisheng 编译选项组合(auto_multi_buffer/ubuf_saving/tile_mix_vector_loop/tile_mix_cube_loop/enable_hivm_auto_cv_balance/sync_solver/unit_flag/vf_fusion_mode) |
| `ubtuner.get_origin_fn` / `get_sorted_configs` / `print_sorted_configs` | ubtuner 辅助:取原始函数 / 排序 / 打印调优结果 |
| `cv_autotune.*` | CV 融合算子调优框架(`CVTileGenerator` 配置生成、`ParamSpace` 参数空间、`HeuristicPruner` 启发式剪枝、`HardwareConstraints` 硬件约束、参数名归一化),偏内部 |

### 1.3 工具函数(`triton.backends.ascend.utils`)

| API | 功能备注 |
|---|---|
| `is_compile_on_910_95(arch=None)` | 当前是否按 Ascend 910_95(A5)目标编译 |
| `get_cann_version()` / `is_cann_version_at_least(major, minor, patch)` | 查询/比较 CANN 版本 |
| `cann_version_compile_args()` | 按 CANN 版本生成编译参数 |
| `triton_enable_libdevice_simt(arch=None)` | libdevice 是否启用 SIMT 路径 |
| `get_ascend_arch_from_env()` / `get_machine_arch()` | 从环境/机器读取昇腾架构 |
| `ub_size_in_kbytes_for_arch` / `graph_ub_budget_bytes_for_arch` | 按架构查询 UB 容量/图级预算 |
| `is_ffts_supported` / `force_disable_ffts` | FFTS 调度支持/禁用判断 |
| `downgrade_llir` / `get_backend_func` / `get_logger` | LLIR 降级 / 后端函数注册表 / 日志 |

### 1.4 编译(`triton.backends.ascend.compiler`)

| API | 功能备注 |
|---|---|
| `NPUOptions` | 昇腾编译选项类(对应 `triton.compile` 的 options;含 multibuffer/unit_flag/enable_hivm_auto_cv_balance 等) |
| `make_ttir(fn, ...)` | 生成 TTIR 的流水线入口(测试常用) |
| `ttir_to_linalg(...)` | TTIR → Linalg 的 lowering 入口 |
| `min_dot_size(a, b)` | 查询 dot 的最小 M/N/K 约束 |
| `get_module_map()` | 返回 `{"triton.language.extra.libdevice": cann.libdevice}` 重映射表 |

### 1.5 其他

| API | 功能备注 |
|---|---|
| `_apply_ascend_patch()` | 安装 4 个运行时猴补丁(CodeGenerator 注入 hacc.target、`compiler.parse` 支持 npubin/bcmlir、`TritonSemantic.dot` tf32→hf32 回退、dtype_guard);测试环境用 |
| `ascend_interpreter.AscendInterpreterBuilder` 等 | interpreter 的昇腾算子实现(内部) |
| `program_grid` / `dtype_guard` / `op_dtype_config` / `debug_line_rewriter` / `cpu_driver` / `driver` | 内部:网格映射、dtype 守卫、算子 dtype 配置、调试行号、CPU/NPU driver |

## 2. `triton.language.extra.kernels`(高层 kernel,来源 `third_party/ascend/language/kernels/`)

| API | 功能备注 |
|---|---|
| `gather_2d_simd(src_ptr, idx_ptr, out_ptr, M, N, K, XBLOCK, XBLOCK_SUB)` | 2D 沿 axis=1 的 SIMD gather 优化 kernel;**已 deprecated**(带 static_print 警告),示例见 `third_party/ascend/tutorials/10-gather-2d-simd.py` |

## 3. 主包内 fork 新增(源码可见,`git diff 85400f80b` 确认)

| API | 功能备注 |
|---|---|
| `triton.extension.buffer.*` | buffer 语言(见 03 文件),主包内新增目录 `python/triton/extension/buffer/`(Python 585 行 + C++ 183 行) |
| `triton.runtime.libentry`(`libentry` / `libtuner` / `LibEntry` / `LibTuner`) | FlagGems(BAAI)风格的持久化自动调优入口:`libtuner` sqlite 记录调优结果、`LibEntry` grid 常量化启动;已适配 `torch_device_fn = torch.npu`。未导出到 `triton.runtime.__init__`,需显式 `from triton.runtime.libentry import libentry` |
| `triton.runtime.code_cache`(`cache_dir_path` / `cache_dir` / `code_cache_dir` / `config_cache_dir` / `clear_cache`) | 编译缓存目录管理工具(FlagGems 风格) |

## 4. 补丁注入(装后包生效,源码树不含;`third_party/ascend/patch/triton-ascend-3.6.0.patch`)

| API/行为 | 功能备注 |
|---|---|
| `triton.compiler.errors.MLIRCompilationError` | **fork 新增异常类**:MLIR/后端阶段编译错误(dev 补丁让 autotuner `_bench` 捕获它) |
| `tl.max(..., propagate_nan=...)` | fork 给原生 `tl.max` 扩展的形参(`standard.py` 新增 `_elementwise_max_propagate_nan`) |
| `tl.math.tanh` | 由 `cann/__init__.py` 猴补恢复(上游 3.6 已删) |
| `tl.device_assert(..., mask=...)` | fork 扩展的 `mask` 形参(`semantic.py`) |
| `tl.exp` 支持 fp16 | `math.py` 的 `_check_dtype` 放宽 |
| `compiler.compile()` / `CompiledKernel` / `parse` 扩展 | 支持 npubin/bcmlir、CodeGenerator builder 注入(供 cann extension 独立 builder) |
| `runtime/interpreter.py` Ascend 扩展 | `_try_import_ascend()`、`create_auto_overflow_assert`、GridExecutor 适配(昇腾 interpreter) |
| `jit.create_function_from_signature` 调整 | kwargs 传播(配合 propagate_nan 等) |

## 5. 非 Python 的配套接口

| API | 功能备注 |
|---|---|
| `triton_launch_kernel` | launcher `.so` 导出的 `extern "C"` C 接口:绕过 `@triton.jit` 直接以 CANN runtime 句柄起 kernel(13 参数),供推理引擎集成;文档见 `docs/en/triton_api_extension/bl/triton_launch_kernel.md` |

## 6. 环境变量与编译选项(非 API,但与扩展配置强相关)

| 项 | 说明 |
|---|---|
| `TRITON_ENABLE_UBTUNER` | 启用 UB 自动调优 |
| `TRITON_ENABLE_LIBDEVICE_SIMT` | libdevice 走 SIMT 实现 |
| autotune 编译选项 | `multibuffer`(默认 true)/`unit_flag`/`tile_mix_vector_loop`/`tile_mix_cube_loop`/`enable_hivm_auto_cv_balance`/`enable_ubuf_saving` 等,位于 `third_party/ascend/backend/compiler.py` 的 `NPUOptions` |
