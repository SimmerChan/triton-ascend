# 01 · 原生 API(Triton 社区原生,与上游 v3.6.0 一致)

> 判定:`python/triton/` 主包源码与上游基线 `85400f80b`(triton-lang/triton 3.6.0)diff 后,
> `language/`、`knobs.py`、`compiler/`、`testing.py`、`experimental/`、`tools/` 均无差异。
> 本文件每个 API 均附功能备注;"⚠️约束"列表示该 API 在 Ascend 后端上有特殊限制(详见 README)。

## 1. 顶层接口 `triton.*`(`python/triton/__init__.py`)

| API | 类型 | 功能备注 |
|---|---|---|
| `triton.jit` | 装饰器 | 把 Python 函数编译为 GPU/NPU kernel;签名参数 version/repr/launch_metadata/do_not_specialize/do_not_specialize_on_alignment/debug/noinline 与上游一致 |
| `triton.JITFunction` | 类 | 被 `@triton.jit` 包装后的 kernel 对象,负责编译缓存与 launch |
| `triton.KernelInterface` | 类 | kernel 基类,提供 `__getitem__(grid)` 启动语法 |
| `triton.autotune` | 装饰器 | 按多个 `Config` 基准测试并缓存最优配置(Ascend 上被后端扩展版替换,见 05) |
| `triton.Config` | 类 | autotune 配置项(meta 参数 + num_warps/num_stages);⚠️ num_stages/num_ctas/maxnreg 在 Ascend 不适用 |
| `triton.heuristics` | 装饰器 | 按 lambda 在 launch 时动态计算 meta 参数 |
| `triton.Heuristics` | 类 | `heuristics` 的包装类 |
| `triton.Autotuner` | 类 | autotune 的运行时实现类 |
| `triton.compile` | 函数 | 编译入口:传入 `ASTSource`/`IRSource` + `GPUTarget` 得到 `CompiledKernel` |
| `triton.ASTSource` | 类 | 以 `@jit` 函数 + 签名 + constexpr 为输入的编译源 |
| `triton.IRSource` | 类 | 以已有 MLIR IR 为输入的编译源 |
| `triton.CompiledKernel` | 类 | 编译产物:持有二进制、可 `__getitem__` 直接 launch |
| `triton.CompilationError` | 异常 | 前端(kernel Python 源码)编译错误 |
| `triton.MLIRCompilationError` | 异常 | **fork 新增(装后补丁注入)**:MLIR/后端阶段编译错误 |
| `triton.InterpreterError` | 异常 | interpreter 模式错误 |
| `triton.OutOfResources` | 异常 | 共享内存等资源超出限制 |
| `triton.TritonError` | 异常 | 所有 triton 异常的基类 |
| `triton.cdiv` | 函数 | 向上取整除 `(x + y - 1) // y` |
| `triton.next_power_of_2` | 函数 | 返回 ≥ n 的最小 2 的幂 |
| `triton.constexpr` | 类 | 编译期常量包装(kernel 参数不参与 specialize 的显式标注) |
| `triton.constexpr_function` | 函数/装饰器 | 把 Python 函数标记为编译期求值函数 |
| `triton.reinterpret` | 函数 | 把 tensor 指针按另一 dtype 重新解释 |
| `triton.TensorWrapper` | 类 | 携带 dtype 元信息的 host 侧张量包装(interpreter 用) |
| `triton.MockTensor` | 类 | 仅含 dtype/shape 的假张量(签名推断用) |
| `triton.set_allocator` | 函数 | 注册 host 侧分配器(供 TMA workspace / profile scratch) |
| `triton.must_use_result` | 装饰器 | 标记函数返回值必须被消费(否则报错) |
| `triton.AsyncCompileMode` | 类 | 异步编译模式开关(上游实验特性) |
| `triton.FutureKernel` | 类 | 异步编译的 kernel 句柄 |
| `triton.runtime` | 模块 | 运行时(jit/driver/autotuner/cache 等) |
| `triton.language` | 模块 | `tl`,kernel 内语言(见下) |
| `triton.testing` | 模块 | 基准测试与数值校验(见下) |
| `triton.tools` | 模块 | 离线编译 `compile.py`、反汇编 `disasm.py`、`link.py`、`mxfp.py`、`tensor_descriptor.py` 等 |
| `triton.experimental` | 包 | `gluon`(上游实验性低层语言;未做昇腾适配) |
| `triton.knobs` | 对象 | 运行环境配置分组:build/cache/compilation/autotuning/runtime/language/nvidia/amd/proton;**无 ascend 分组**,昇腾开关走环境变量/后端 |

## 2. `triton.language`(`tl.*`)— 与上游 `__all__` 完全一致

### 2.1 编程模型

| API | 功能备注 |
|---|---|
| `tl.tensor` | kernel 内张量值类型;成员方法见 2.10 |
| `tl.program_id(axis)` | 当前程序(逻辑核)在网格中的坐标 |
| `tl.num_programs(axis)` | 网格在给定轴上的程序总数 |
| `tl.tensor_descriptor` | TMA 风格张量描述符类型(配合 host 侧 descriptor) |
| `tl.map_elementwise(fn, *args, pack=1)` | 把标量函数逐元素映射到张量,允许 if 分支;⚠️ Ascend 不支持 while、pack 无效果 |

### 2.2 创建 / 转换

| API | 功能备注 |
|---|---|
| `tl.arange(start, end)` | 创建 `[start, end)` 的连续 int32 一维张量 |
| `tl.full(shape, value, dtype)` | 创建以标量 value 填充的张量 |
| `tl.zeros(shape, dtype)` | 创建全 0 张量 |
| `tl.zeros_like(input)` | 按输入 shape/dtype 创建全 0 张量 |
| `tl.cat(input, other, can_reorder=False)` | 沿第一维拼接两个一维张量 |
| `tl.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)` | 类型转换(bitcast=True 时按位重解释) |
| dtype 常量 | `int1/8/16/32/64`, `uint8/16/32/64`, `float16`, `bfloat16`, `float32`, `float64`, `float8e4nv/e4b8/e5/e4b15/e5b16`, `pi32_t`, `void` |
| `tl.constexpr` / `tl.const` | 编译期常量类型 |
| `tl.tuple` / `tl.slice` | 元组值类型 / 切片描述(`slice(None, None, 2)` 风格) |
| `tl.PropagateNan` | 归约中 NaN 传播策略枚举(NONE/ALL) |
| `tl.TRITON_MAX_TENSOR_NUMEL` | 单个张量允许的最大元素数(常量) |
| `tl.pointer_type` / `tl.block_type` / `tl.tuple_type` / `tl.tensor_descriptor_type` | 内部类型对象 |
| `tl.dtype` | dtype 描述类(`is_block/is_ptr/to_ir` 等) |

### 2.3 形状操作

| API | 功能备注 |
|---|---|
| `tl.broadcast(input, other)` | 把两个张量广播到共同形状 |
| `tl.broadcast_to(input, shape)` | 广播到指定形状 |
| `tl.expand_dims(input, axis)` | 插入长度 1 的新轴 |
| `tl.permute(input, dims)` | 按维度重排(同 `trans` 泛化) |
| `tl.trans(input, *dims)` | 转置/维度交换 |
| `tl.reshape(input, shape, can_reorder=False)` | 改变形状 |
| `tl.view(input, shape)` | 改变形状(要求内存布局兼容,不可重排) |
| `tl.ravel(input, can_reorder=False)` | 展平为一维 |
| `tl.join(a, b)` | 沿新末维拼接两个形状相同的一维张量 |
| `tl.split(input)` | 把末维为 2 的张量拆成两个张量 |
| `tl.interleave(a, b)` | 沿末维交错拼接两个一维张量 |
| `tl.item(input)` | 单元素张量取标量 |

### 2.4 线性代数

| API | 功能备注 |
|---|---|
| `tl.dot(a, b, acc=None, ...)` | 矩阵乘(A2/A3 支持 int4/int8/fp16/bf16/fp32;950 另支持 fp8;tf32→hf32 回退) |
| `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, ...)` | 带缩放的低精度矩阵乘(mxfp/nvfp4 等);⚠️ K 须为 64 倍数、k_pack 不支持 |

### 2.5 内存 / 指针

| API | 功能备注 |
|---|---|
| `tl.load(pointer, mask=None, other=None, ...)` | 从内存加载张量;⚠️ eviction_policy 无效、cache_modifier 仅 950 SIMT |
| `tl.store(pointer, value, mask=None, ...)` | 写入内存;约束同上 |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | 构造分块指针(多维内存块的抽象) |
| `tl.advance(base, offsets)` | 移动 block_ptr 的偏移 |
| `tl.make_tensor_descriptor(base, shape, strides, block_shape)` | 构造 host 侧 TMA 风格描述符(NPU 上走非 TMA 降级路径) |
| `tl.load_tensor_descriptor(desc, offsets)` | 经描述符加载 |
| `tl.store_tensor_descriptor(desc, offsets, value)` | 经描述符存储 |

### 2.6 索引 / 选择

| API | 功能备注 |
|---|---|
| `tl.where(cond, a, b)` | 按条件逐元素选择 |
| `tl.gather(src, index, axis)` | 沿轴按索引收集(上游 #5262) |
| `tl.flip(input, dim=None)` | 沿指定维翻转 |
| `tl.swizzle2d(x, size_y, size_x, size_g)` | 二维下标重排(shared memory bank 冲突优化) |

### 2.7 基础数学 `tl.math`(`python/triton/language/math.py`)

| API | 功能备注 |
|---|---|
| `tl.abs(x)` | 绝对值 |
| `tl.exp(x)` | e^x;⚠️ Ascend 补丁放宽允许 fp16 |
| `tl.exp2(x)` | 2^x |
| `tl.log(x)` | 自然对数 |
| `tl.log2(x)` | 以 2 为底对数 |
| `tl.sin(x)` / `tl.cos(x)` | 正弦 / 余弦 |
| `tl.sqrt(x)` | 平方根 |
| `tl.sqrt_rn(x)` | 平方根(最近偶舍入) |
| `tl.rsqrt(x)` | 平方根倒数 |
| `tl.erf(x)` | 误差函数 |
| `tl.floor(x)` / `tl.ceil(x)` | 向下 / 向上取整 |
| `tl.fdiv(x, y, ieee_rounding=False)` | 浮点除法(可要求 IEEE 舍入) |
| `tl.div_rn(x, y)` | IEEE 最近偶舍入除法 |
| `tl.fma(x, y, z)` | 融合乘加 x*y+z |
| `tl.umulhi(x, y)` | 无符号乘积的高 32 位 |
| `tl.math.tanh` | **注**:上游 3.6 已删除;本 fork 由 `cann/__init__.py` 猴补恢复为昇腾实现 |

### 2.8 逻辑 / 比较 / 算术

| API | 功能备注 |
|---|---|
| `tl.maximum(x, y)` / `tl.minimum(x, y)` | 逐元素最大 / 最小 |
| `tl.clamp(x, min, max)` | 截断到 [min, max] |
| `tl.add` / `tl.sub` / `tl.mul` | 显式逐元素加 / 减 / 乘(内建版) |
| 运算符重载 | `+ - * / // % ** ` 及 `& \| ^ << >>`、比较 `> >= < <= == !=`、`~ -`、`tensor.logical_and/logical_or`、`__getitem__`、`T` 等,均为上游原生 |

### 2.9 归约 / 扫描 / 排序

| API | 功能备注 |
|---|---|
| `tl.sum(input, axis=None, dtype=None)` | 求和归约(Ascend 补丁含 dtype 提升逻辑) |
| `tl.max(input, axis=None, return_indices=False)` | 最大值归约;⚠️ 装后包 fork 扩展 `propagate_nan` 形参 |
| `tl.min(input, axis=None, return_indices=False)` | 最小值归约 |
| `tl.argmax(input, axis, tie_break_left=True)` | 最大值索引 |
| `tl.argmin(input, axis, tie_break_left=True)` | 最小值索引 |
| `tl.xor_sum(input, axis=None)` | 异或归约 |
| `tl.reduce_or(input, axis=None)` | 逐位或归约 |
| `tl.reduce(fn, axis, combine_fn=None)` | 用自定义 combine 函数做归约 |
| `tl.cumsum(input, axis=0, reverse=False)` | 前缀和 |
| `tl.cumprod(input, axis=0, reverse=False)` | 前缀积 |
| `tl.associative_scan(input, axis, combine_fn, reverse=False)` | 自定义结合性扫描 |
| `tl.histogram(input, num_bins)` | 直方图统计 |
| `tl.sort(x, dim=None, descending=False)` | 双调排序(沿末维) |
| `tl.topk(x, k, dim=None)` | 取前 k 大 |
| `tl.bitonic_merge(x, dim, k, ascending, ...) ` | 双调归并基元(sort/topk 内部使用,亦可直接调用) |
| `tl.sigmoid(x)` | 1/(1+e^-x) |
| `tl.softmax(x, dim=None)` | softmax(单程序维度内) |

### 2.10 原子操作

| API | 功能备注 |
|---|---|
| `tl.atomic_add/atomic_max/atomic_min/atomic_and/atomic_or/atomic_xor(ptr, val, mask)` | 原子读改写 |
| `tl.atomic_cas(ptr, cmp, val, mask)` | 原子比较交换 |
| `tl.atomic_xchg(ptr, val, mask)` | 原子交换 |

### 2.11 随机数 `tl.random`(`python/triton/language/random.py`)

| API | 功能备注 |
|---|---|
| `tl.rand(seed, offset)` | [0,1) 均匀分布随机数(philox) |
| `tl.rand4x(seed, offsets)` | 一次生成 4 个随机数 |
| `tl.randn(seed, offset)` | 标准正态随机数 |
| `tl.randn4x(seed, offsets)` | 一次生成 4 个正态随机数 |
| `tl.randint(seed, offset, dtype)` | 随机整数 |
| `tl.randint4x(seed, offsets, dtype)` | 一次生成 4 个随机整数 |
| `tl.philox` / `tl.philox_impl` | philox 计数器随机数生成内部实现 |
| `tl.pair_uniform_to_normal` | 均匀对→正态变换 |
| `tl.uint_to_uniform_float` | 无符号整数→[0,1) 浮点 |

### 2.12 迭代 / 控制流

| API | 功能备注 |
|---|---|
| `tl.range(start, end, step, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, disable_licm=False)` | 可带编译提示的循环迭代器;⚠️ 后三个提示在 Ascend 功能不完整 |
| `tl.static_range(start, end, step)` | 编译期展开的循环 |
| `tl.condition(cond, disable_licm=True)` | while 条件包装(禁止 LICM,上游 #7733) |
| `tl.static_assert` / `tl.static_print` | 编译期断言 / 编译期打印 |

### 2.13 编译提示

| API | 功能备注 |
|---|---|
| `tl.multiple_of(input, values)` | 提示各维度是 values 的倍数 |
| `tl.max_contiguous(input, values)` | 提示各维度连续段长度 |
| `tl.max_constancy(input, values)` | 提示各维度取值恒定段长度 |
| `tl.assume(cond)` | 向编译器声明恒真条件(上游 #4396) |
| `tl.debug_barrier()` | 原生版:程序内同步屏障(无数据语义) |

### 2.14 调试 / 其他

| API | 功能备注 |
|---|---|
| `tl.device_print(args...)` | 设备端打印 |
| `tl.device_assert(cond, msg, mask)` | 设备端断言;⚠️ Ascend 补丁扩展了 `mask` 形参 |
| `tl.inline_asm_elementwise(...)` | 内联 PTX/汇编;⚠️ Ascend 仅支持 s64/f32、`'l'` 约束、1D |

### 2.15 `tensor` 成员方法(上游原生,`x.method(...)` 形式)

`x.T`、`x.to(ty)`、`x.broadcast_to`、`x.trans/permute/split/view/reshape/expand_dims`、`x.store/advance`、`x.atomic_cas/xchg/add/max/min/and/or/xor`、`x.exp/log/cos/sin/sqrt/rsqrt/abs`、`x.reduce/associative_scan/gather/histogram`、`x.cdiv`、`x.sigmoid/softmax/ravel`、`x.max/argmax/min/argmin/sum/xor_sum/reduce_or/cumsum/cumprod/sort/flip` — 功能与对应内建函数一致。

## 3. `triton.testing`(`python/triton/testing.py`,上游原生)

| API | 功能备注 |
|---|---|
| `do_bench(fn, ...)` | CUDA 计时基准(Ascend 上不可用,用扩展 `do_bench_npu`) |
| `do_bench_cudagraph(fn, ...)` | CUDA Graph 基准 |
| `assert_close(actual, expected, ...)` | 数值容差比较 |
| `perf_report(bench)` / `Benchmark` / `Mark` | 基准报告框架 |
| `nvsmi(attrs)` | 读取 nvidia-smi 信息 |
| `get_dram_gbps` / `get_max_tensorcore_tflops` / `get_max_simd_tflops` | 硬件峰值指标估算 |
| `cuda_memcheck(fn)` | CUDA memcheck 运行 |
| `set_gpu_clock()` | 锁频(仅 GPU) |

## 4. 原生 runtime / compiler(`python/triton/runtime`、`compiler/`)

| API | 功能备注 |
|---|---|
| `triton.runtime.jit.*` | `KernelParam`、`mangle_type`、`create_function_from_signature`、`DependenciesFinder` 等内部机制(上游) |
| `triton.runtime.interpreter.GridExecutor` | interpreter 模式下以 numpy 逐网格执行 kernel |
| `triton.runtime.driver` | 后端 driver 发现与选择(Ascend 下自动选中 `triton.backends.ascend`) |
| `triton.runtime.cache` | 编译缓存管理(含 RemoteCacheBackend/RedisRemoteCacheBackend) |
| `triton.compiler.parse(...)` | 解析 MLIR 文本为 IR(装后由补丁扩展支持 npubin/bcmlir) |
| `triton.compiler.make_backend(target)` | 按 target 构造后端实例 |
| `triton.backends.GPUTarget` / `BaseBackend` / `DriverBase` / `GPUDriver` | 目标描述与后端基类(上游) |

> 来源:`python/triton/__init__.py`、`python/triton/language/{__init__,core,standard,math,random}.py`、`python/triton/testing.py`。
