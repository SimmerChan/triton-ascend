# 03 · 扩展 API:`bl` = `triton.extension.buffer.language`

> buffer 语言:以 `bl.buffer`(片上内存块)为一等值的编程接口,与 `tl.tensor` 语义隔离,
> 二者须经 `to_buffer` / `to_tensor` 显式转换。上游 Triton 没有。
> 来源:`python/triton/extension/buffer/language/`(**主包内 fork 新增目录**,
> 含 Python 层 585 行 + `buffer/src/buffer_ir.cc` 183 行 C++ IR 绑定)。
> 官方文档:`docs/en/triton_api_extension/bl/`、`docs/zh/python-api/triton.language.extra.extension.buffer.language.rst`。

## API 清单

| API | 功能备注 |
|---|---|
| `bl.buffer` | 值类型:表示一段片上内存(UB/L1/L0A/L0B/L0C),持有 element type / shape / strides / space;不能与 `tl.tensor` 互相赋值。成员方法:`.subview(offsets, sizes, strides)`(取零拷贝视图)、`.to_tensor(writable=True, target_shape=None)`(转回 tl.tensor) |
| `bl.buffer_type` | `bl.buffer` 的类型对象(`tl.dtype` 子类,携带 element_ty/shape/strides/space) |
| `bl.address_space` | 抽象地址空间基类(与地址空间相关的类型/常量基类);昇腾具体地址空间用 `al.ascend_address_space.{UB,L1,L0A,L0B,L0C}` |
| `bl.alloc(etype, shape, _address_space=None, is_mem_unique=False)` | 在指定地址空间分配 buffer(硬件无关,映射 `memref.alloc`)。例:`bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)`。dtype 支持(A2/A3/950):int8/int16/int32/uint8/uint64/int64/fp32/bf16/bool;**不支持 uint16/uint32/fp16** |
| `bl.to_buffer(tensor, space=None, bind_buffer=None)` | 核心转换入口:`tl.tensor` → `bl.buffer`;`space` 指定目标地址空间(UB/L1/L0A/L0B/L0C),`bind_buffer` 绑定到已有 buffer(绑定时直接返回该 buffer,shape/etype 必须一致,一个 tensor 不能绑多个 buffer) |
| `bl.to_tensor(memref, writable=True, target_shape=None)` | `bl.buffer` → `tl.tensor`(映射 bufferization.to_tensor;`writable` 控制可写性) |
| `bl.subview(src, offsets, sizes, strides)` | 零拷贝取 buffer 视图(映射 `memref.subview`);约束:offset 非负且 32 字节对齐、stride 全为 1、sizes/strides 必须 `List[tl.constexpr]` |
| `bl.builtin` / `bl.is_builtin` | buffer-language 内建装饰器与判别(标记函数须在 `@triton.jit` 内调用) |
| `bl.allocate_local_buffer` | 内部/约束条目引用的本地分配接口(`to_buffer`/`to_tensor` 的约束说明以它为基准),无独立公开文档页 |

## 与 al 的关系

- `bl` 负责内存(buffer 抽象、分配、视图、与 tensor 互转);
- `al` 负责 Ascend 硬件语义(scope 绑核、地址空间枚举、搬运 fixpipe/copy、同步 sync_block_*、Cube 计算 dot/conv);
- 典型组合:`ub = bl.alloc(tl.float32, [M,N], al.ascend_address_space.UB)` → `bl.to_buffer(t)` → `al.copy(t, ub)` → `bl.to_tensor(ub)`。

## 附:C 接口 `triton_launch_kernel`(与 bl 同目录文档,非 Python API)

| 接口 | 功能备注 |
|---|---|
| `triton_launch_kernel`(launcher `.so` 导出的 `extern "C"` 函数) | 绕过 `@triton.jit` Python 路径,直接以 CANN runtime 句柄启动已编译 kernel,供推理引擎/高级部署集成;13 个参数:kernelName/func/stream/gridXYZ/shapes_data/shape_dims/num_tensors/tensor_kinds/kernel_args/arg_sizes/num_args。见 `docs/en/triton_api_extension/bl/triton_launch_kernel.md` |
