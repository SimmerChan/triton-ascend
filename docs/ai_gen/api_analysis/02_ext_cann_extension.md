# 02 · 扩展 API:`al` = `triton.language.extra.cann.extension`

> 昇腾语言扩展,上游 Triton 没有。来源:`third_party/ascend/language/cann/extension/`
>(安装后挂载为 `triton.language.extra.cann.extension`)。导出表见该目录 `__init__.py:84-167`,共 63 项。
> 官方文档:`docs/en/triton_api_extension/al/`、`docs/zh/python-api/triton.language.extra.cann.extension.rst`。

## 1. 执行单元控制

| API | 功能备注 |
|---|---|
| `al.scope` | 类,上下文管理器。`with al.scope(core_mode="cube"/"vector"):` 把 with 块内算子显式绑定到 Cube 核或 Vector 核执行;支持 `noinline`、`vec_mode`、`disable_auto_sync` 等属性;编译期把块转为 scope region(`hivm.tcore_type`)。约束:每 kernel 各 1 个 cube scope + 1 个 vector scope,二者并行执行,跨 scope 需显式同步 |
| `al.ascend_address_space` | 片上地址空间常量组(映射 `hivm::AddressSpace`):`.UB`(统一缓冲)/`.L1`/`.L0A`/`.L0B`/`.L0C`;须与 `bl.alloc` 配合使用 |
| `al.sub_vec_id()` | 返回当前 AI Core 上 Vector 子核的索引(i16,∈ [0, N));仅 AIC+AIV 混合内核合法,纯 Cube/纯 Vector 内核会编译失败(映射 `hivm.hir.get_sub_block_idx`) |
| `al.sub_vec_num` | 返回每个 AI Core 的 Vector 子核数量(运行时查 NPUUtils) |
| `al.CORE` | 执行单元枚举:`VECTOR / CUBE / CUBE_OR_VECTOR / CUBE_AND_VECTOR`(custom op 必填字段) |
| `al.MODE` | 执行模式枚举:`SIMD / SIMT / MIX`(custom op 非 CUBE 时必填) |
| `al.int64` | 类。`al.int64(x)` 让 custom op 的 Python int 按设备侧 int64 传递 |

## 2. 数据搬运

| API | 功能备注 |
|---|---|
| `al.copy(src, dst)` | 片上数据拷贝(生成 `hivm.hir.copy`):支持 UB→UB 或 UB→L1;src/dst 可为 `tl.tensor` 或 `bl.buffer`,须同 dtype/shape。是 `copy_from_ub_to_l1` 的超集 |
| `al.copy_from_ub_to_l1(src, dst)` | UB→L1 直拷(走 A5 硬件直达通路,避免经 GM 两跳);已标 deprecated,改用 `al.copy` |
| `al.fixpipe(src, dst=None, dma_mode=NZ2ND, dual_dst_mode=NO_DUAL, pre_quant_mode=..., pre_relu_mode=...)` | Cube 累加结果从 L0C 直接搬出到 UB(仅 Ascend910_95);src 必须是 `dot` 结果,dst 必须是 UB memscope buffer;`dst=None` 时返回新建 tensor。配 4 个枚举:`FixpipeDMAMode`(NZ2DN/NZ2ND/NZ2NZ)、`FixpipeDualDstMode`、`FixpipePreQuantMode`(NO_QUANT/F322BF16/F322F16/S322I8)、`FixpipePreReluMode` |

## 3. 跨核同步

| API | 功能备注 |
|---|---|
| `al.sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)` | 跨核 set-flag:生产者核发同步信号(映射 CrossCoreSetFlag 计数器模型);sender/receiver ∈ {"cube","vector"} 且必须不同;event_id ∈ [0,15];默认 pipe:cube 侧 FIX/MTE2,vector 侧 MTE3/MTE2 |
| `al.sync_block_wait(sender, receiver, event_id, ...)` | 与 `sync_block_set` 配对的等待端(映射 CrossCoreWaitFlag),参数语义相同 |
| `al.sync_block_all(mode, event_id)` | 全核同步;mode ∈ {"all_cube","all_vector","all","all_sub_vector"};event_id ∈ [0,15] |
| `al.debug_barrier(sync_mode)` | Vector 单元内细粒度手动屏障(在 vector/scaler load/store 指令间插同步);`sync_mode: al.SYNC_IN_VF` 提供 12 种模式(VV_ALL/VST_VLD/VLD_VST/VST_VST/VS_ALL/VST_LD/VLD_ST/VST_ST/SV_ALL/ST_VLD/LD_VST/ST_VST);只能在 scope 内使用 |
| `al.PIPE` | 硬件流水线枚举:`PIPE_S/V/M/MTE1/MTE2/MTE3/ALL/FIX` |
| `al.SYNC_IN_VF` | VF 内屏障模式枚举(见 `debug_barrier`) |
| `al.EVENT_ID` | 同步事件 ID 辅助(EVENT_ID0–EVENT_ID7) |

## 4. Cube 计算

| API | 功能备注 |
|---|---|
| `al.dot(a, b, format_a="", format_b="", format_c="")` | 昇腾扩展矩阵乘:format 支持 `"fractal"`(zN 4D 分形布局)与 `"nd"`/`""`(2D ND);dtype f16/bf16/f32/int8;fractal 约束 block_col=32/block_row=16;输出 f32/i32。与原生 `tl.dot` 相比暴露了分形布局控制 |
| `al.conv1d(input, weight, bias, stride, padding, dilation, groups)` | 1D Cube 卷积,参数风格仿 `torch.nn`;支持 'same'/'valid' padding |
| `al.conv2d(input, weight, bias, stride, padding_size, dilation, groups)` | 2D Cube 卷积,参数风格仿 `torch.nn`;支持 'same'/'valid' padding |

## 5. 向量 / 内存扩展算子

| API | 功能备注 |
|---|---|
| `al.insert_slice(ful, sub, offsets, sizes, strides)` | 把子张量 `sub` 写入大张量 `ful` 的指定位置(UB 内拼装;FlashAttention/CV fusion 分块大 accumulator 用) |
| `al.extract_slice(ful, offsets, sizes, strides)` | 从大张量按 offsets/sizes/strides 切出子张量(与 insert_slice 配对) |
| `al.get_element(input, indices)` | 按索引取出张量中的单个元素(标量) |
| `al.sort(ptr, dim=-1, descending=False)` | 扩展排序:沿指定维排序;不支持 uint8/int32/int64/fp64/bool 等 dtype(见 constraints) |
| `al.flip(ptr, dim=-1)` | 沿指定维翻转张量(SIMD/SIMT 双实现) |
| `al.cast(input, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None)` | 扩展类型转换:比 `tl.cast` 多 `overflow_mode ∈ {"trunc","saturate"}`(整型降级溢出处理) |
| `al.index_put(dest, indices, value, dim, ...)` | 向 GM 目标按索引写入(dest 在 GM,index/value 在 UB;支持 index_boundary、start/end_offset、dst_stride);仅 950PR&950DT,fp16/bf16/f32 |
| `al.gather_out_to_ub(src, index, ...)` | GM 源按索引 gather 到 UB(支持 1–5 维 index、越界默认值 other);仅 950PR&950DT |
| `al.scatter_ub_to_out(src, index, ...)` | UB tile 按索引 scatter 写回 GM;仅 950PR&950DT |
| `al.index_select_simd(src, dim, index, src_shape, src_offset, read_shape)` | GM→UB 零拷贝并行 index_select;int32/int64 索引、1D index;约束:`read_shape[dim]` 必须为 -1,`dim` 不能是最后一维 |

## 6. 编译提示

| API | 功能备注 |
|---|---|
| `al.compile_hint(ptr, hint_name, hint_val=None)` | 向 tensor 附加编译提示 MLIR 注解(如 `"hivm.multi_buffer"`);SIMT 模式下无效 |
| `al.multibuffer(src, size)` | 为 tensor 设置多缓冲(`size` 目前只支持 2);内部即 `compile_hint("hivm.multi_buffer")`;等价 `tl.range` 的 `disallow_acc_multi_buffer` 反向操作 |
| `al.parallel(start, end, num_stages=..., loop_unroll_factor=..., bind_sub_block=False)` | 类(继承 `tl.range`)。显式多核循环迭代器:`for s in al.parallel(0, 2):`;`bind_sub_block=True` 让多个 Vector 子核参与循环(910B 混合 kernel,最多 2 个);⚠️ 并非所有硬件/编译配置可用 |

## 7. 自定义算子框架

| API | 功能备注 |
|---|---|
| `al.custom(name, *args, out=...)` | 在 kernel 内按名字调用已注册的 AscendC 自定义算子;单 pipe 生成 `hivm.hir.custom`,双 pipe 序列生成 custom_macro;支持字面量/元组/列表参数(配 `al.int64`) |
| `al.register_custom_op` | 装饰器:注册类为自定义算子。类字段:`name`(默认类名)、`core: CORE`(必填)、`pipe: PIPE`(必填,双 PIPE 列表表示 macro)、`mode: MODE`(非 CUBE 必填)、可选 `symbol`/`bitcode`/`source`/`compile`/`extra_attr`/`align_dim`/`indexing_map`(al.affine_map 列表)/`iterator_types`/`extra_buffers`/`sync_event_slots`;`__init__` 签名即算子参数签名 |
| `al.custom_semantic(name, ..., out=..., _semantic=...)` | 内部语义入口(用户一般不用),供 extern 函数包装 |
| `al.SyncEventSlot` | macro custom op 的 `sync_event_slots` 条目(set_pipe/wait_pipe/sync/event) |
| `al.builtin` / `al.is_builtin` | 昇腾 buffer-language 内建装饰器与判别函数(须在 `@triton.jit` 内调用) |
| `__builtin_index_select`(内置 custom op) | SIMT 模板 gather(2D–5D src、1D/2D index);已 deprecated,建议直接 `al.custom('__builtin_index_select', ...)` |

## 8. MLIR Affine 类型(供 custom op `indexing_map` 使用)

| API | 功能备注 |
|---|---|
| `al.affine_expr` / `al.AffineExpr` | Affine 表达式基类型 |
| `al.affine_constant_expr` / `al.AffineConstantExpr` | 常量表达式 |
| `al.affine_dim_expr` / `al.AffineDimExpr` | 维度表达式 |
| `al.affine_symbol_expr` / `al.AffineSymbolExpr` | 符号表达式 |
| `al.affine_binary_op_expr` / `al.AffineBinaryOpExpr` | 二元运算表达式(+ - * floordiv ceildiv mod) |
| `al.affine_map` / `al.AffineMap` | 仿射映射(维度/符号 → 结果表达式) |
| `al.IteratorType` | custom op 迭代器类型枚举:Parallel/Broadcast/Transpose/Reduction/…/Opaque |

## 9. 其他导出

| API | 功能备注 |
|---|---|
| `al.is_compile_on_910_95(arch=None)` | 判断当前是否按 Ascend 910_95(A5)目标编译(来自 `triton.backends.ascend.utils`) |
| `al.atan2 / isfinited / finitef`(math_ops) | 已 deprecated,指向 libdevice 同名函数 |
| `al.static_range / ascend_cast_impl` | 内部辅助(编译期 range / cast 实现) |

## 典型用法(测试/教程中验证)

```python
import triton.language.extra.cann.extension as al

with al.scope(core_mode="vector"):
    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)
al.sync_block_set("cube", "vector", 5); al.sync_block_wait("cube", "vector", 5)
al.sync_block_all("all_vector", 3);     al.debug_barrier(al.SYNC_IN_VF.VST_VLD)
blk = al.extract_slice(acc, (s*SUB_M, 0), (SUB_M, BLOCK_N), (1, 1))
acc = al.insert_slice(acc, val, (s*SUB_M, 0), (SUB_M, BLOCK_N), (1, 1))
for s in al.parallel(0, 2): ...
y = al.custom("my_op", x, al.int64(0), out=y)
```
