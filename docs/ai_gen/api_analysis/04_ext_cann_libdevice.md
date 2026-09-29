# 04 · 扩展 API:`triton.language.extra.cann.libdevice`(昇腾版 libdevice,177 个公开函数)

> 上游 Triton 只提供 libdevice 桩文件(`python/triton/language/extra/libdevice.py`,函数体为 `...`,
> 依赖各后端 module-map 提供实现)。本 fork 在 `third_party/ascend/language/cann/libdevice.py` 提供
> 昇腾实现(177 个公开函数,按 `is_compile_on_910_95` / `triton_enable_libdevice_simt` 在 SIMD/SIMT
> 双路径间选择),并把 `tl.extra.libdevice` 重映射到本模块。
>
> 舍入方向后缀约定:`_rn` 最近偶舍入 / `_rz` 向零 / `_rd` 向 -∞ / `_ru` 向 +∞。
> 用法:`import triton.language.extra.cann.libdevice as libdevice` 或 `tl.extra.libdevice.xxx`。

## 1. 昇腾特有 / 特化入口

| API | 功能备注 |
|---|---|
| `flip(input, dim)` | 沿指定维翻转张量(SIMD 走 create_flip,SIMT 走 xor-swap)。**CUDA libdevice 无此函数,昇腾新增** |
| `reciprocal(x)` | 倒数 1/x |
| `relu(x)` | max(x, 0) |
| `tanh(x)` | 双曲正切;同时被 `cann/__init__.py` 猴补恢复为 `tl.math.tanh`(上游 3.6 已删除该函数) |

## 2. 指数 / 对数 / 幂

| API | 功能备注 |
|---|---|
| `exp(x)` | e^x |
| `exp2(x)` | 2^x |
| `exp10(x)` | 10^x |
| `expm1(x)` | e^x − 1(小值精确) |
| `log(x)` | 自然对数 |
| `log2(x)` | 以 2 为底对数 |
| `log10(x)` | 以 10 为底对数 |
| `log1p(x)` | log(1 + x)(小值精确) |
| `pow(x, y)` | x^y |
| `ldexp(x, e)` | x · 2^e |
| `scalbn(x, n)` | x · 2^n |
| `ilogb(x)` | x 的指数部分的整数值 |
| `logb(x)` | x 的指数部分(浮点) |
| `nan()` | 生成 NaN |

## 3. 快速低精度函数(`fast_*` 族,牺牲精度换速度)

| API | 功能备注 |
|---|---|
| `fast_dividef(x, y)` | 快速浮点除法 |
| `fast_expf(x)` | 快速 e^x(float) |
| `fast_exp10f(x)` | 快速 10^x |
| `fast_sinf(x)` / `fast_cosf(x)` / `fast_tanf(x)` | 快速正弦 / 余弦 / 正切 |
| `fast_tanhf(x)` | 快速双曲正切 |
| `fast_logf(x)` / `fast_log2f(x)` / `fast_log10f(x)` | 快速对数 |
| `fast_powf(x, y)` | 快速幂 |

## 4. 三角 / 双曲 / 反三角

| API | 功能备注 |
|---|---|
| `sin(x)` / `cos(x)` | 正弦 / 余弦 |
| `tan(x)` | 正切 |
| `asin(x)` / `acos(x)` / `atan(x)` | 反正弦 / 反余弦 / 反正切 |
| `atan2(y, x)` | y/x 的象限感知反正切 |
| `sinpi(x)` / `cospi(x)` | sin(πx) / cos(πx) |
| `sinh(x)` / `cosh(x)` / `tanh(x)` | 双曲正弦 / 余弦 / 正切 |
| `asinh(x)` / `acosh(x)` / `atanh(x)` | 反双曲正弦 / 余弦 / 正切 |

## 5. 取整 / 绝对值 / 符号 / 比较

| API | 功能备注 |
|---|---|
| `abs(x)` | 绝对值 |
| `ceil(x)` / `floor(x)` | 向上 / 向下取整 |
| `trunc(x)` | 向零截断取整 |
| `round(x)` | 四舍五入(半值远离零) |
| `rint(x)` / `nearbyint(x)` | 按当前舍入模式取整 |
| `llrint(x)` / `llround(x)` | 取整并返回 int64 |
| `signbit(x)` | 符号位是否为 1 |
| `copysign(x, y)` | 取 x 的幅值、y 的符号 |
| `saturatef(x)` | 截断到 [0.0, 1.0] |
| `max(x, y)` / `min(x, y)` | 浮点最大 / 最小(NaN 语义与算术比较不同) |
| `fmod(x, y)` | 浮点取模(符号随 x) |
| `remainder(x, y)` | IEEE 754 余数(符号随最近整数倍) |
| `fdim(x, y)` | max(x − y, 0) |

## 6. 带舍入方向的算术(`_rn/_rz/_rd/_ru` 四方向)

| API | 功能备注 |
|---|---|
| `add_rn/rz/rd/ru(x, y)` | 加法,显式舍入方向 |
| `sub_rn/rz/rd/ru(x, y)` | 减法,显式舍入方向 |
| `mul_rn/rz/rd/ru(x, y)` | 乘法,显式舍入方向 |
| `div_rn(x, y)` / `div_rz/rd/ru` | 除法,显式舍入方向 |
| `rcp_rn/rz/rd/ru(x)` | 倒数,显式舍入方向 |
| `sqrt_rn/rz/rd/ru(x)` | 平方根,显式舍入方向 |
| `rsqrt_rn(x)` | 平方根倒数(最近偶舍入) |
| `fma_rn/rz/rd/ru(x, y, z)` | 融合乘加 x·y+z,显式舍入方向 |
| `fma(x, y, z)` | 融合乘加(默认舍入) |
| `div_rn(x, y)`(末尾组重复定义) | IEEE 除法变体 |
| `fdiv(x, y, ieee_rounding)` | 浮点除法,可选 IEEE 舍入 |

## 7. 特殊函数

| API | 功能备注 |
|---|---|
| `hypot(x, y)` | sqrt(x²+y²)(避免溢出) |
| `rhypot(x, y)` | 1 / hypot(x, y) |
| `cbrt(x)` | 立方根 |
| `rcbrt(x)` | 立方根倒数 |
| `norm3d(x, y, z)` | sqrt(x²+y²+z²) |
| `rnorm3d(x, y, z)` | 1 / norm3d |
| `norm4d(x, y, z, w)` | sqrt(x²+y²+z²+w²) |
| `rnorm4d(x, y, z, w)` | 1 / norm4d |
| `nextafter(x, y)` | x 朝 y 方向的下一个可表示浮点数 |

### 7.1 贝塞尔函数

| API | 功能备注 |
|---|---|
| `j0(x)` / `j1(x)` / `jn(n, x)` | 第一类贝塞尔函数(0 阶 / 1 阶 / n 阶) |
| `y0(x)` / `y1(x)` / `yn(n, x)` | 第二类贝塞尔函数(诺伊曼函数) |
| `cyl_bessel_i0(x)` / `cyl_bessel_i1(x)` | 第一类修正柱贝塞尔函数(0/1 阶) |

### 7.2 误差函数 / 正态分布 / 伽马函数

| API | 功能备注 |
|---|---|
| `erf(x)` | 误差函数 |
| `erfc(x)` | 补余误差函数 1−erf(x) |
| `erfcx(x)` | 缩放补余误差函数 e^(x²)·erfc(x) |
| `erfcinv(x)` | erfc 的反函数 |
| `erfinv(x)` | erf 的反函数 |
| `normcdf(x)` | 标准正态分布累积函数 Φ(x) |
| `normcdfinv(x)` | Φ 的反函数 |
| `gamma(x)` | ln|Γ(x)|(lgamma 的历史别名) |
| `tgamma(x)` | Γ(x)(真正的伽马函数) |
| `lgamma(x)` | ln|Γ(x)| |

## 8. 位操作 / 整数运算

| API | 功能备注 |
|---|---|
| `clz(x)` | 前导 0 计数 |
| `popc(x)` | 1 位计数(popcount) |
| `brev(x)` | 位反转 |
| `byte_perm(x, y, s)` | 按 8 位选择器 s 重排 x/y 的 8 个字节 |
| `ffs(x)` | 最低位 1 的位置(1 起,无则为 0) |
| `mulhi(x, y)` | 32×32 位乘积的高 32 位(有符号) |
| `mul24(x, y)` | 24 位快速乘法 |
| `sad(x, y, c)` | |x−y|+c(绝对差累加) |
| `hadd(x, y)` | (x+y)/2(不溢出) |
| `rhadd(x, y)` | 舍入平均 (x+y)/2 |
| `uhadd/umul24/umulhi/urhadd/usad` | 上述运算的无符号版本(umulhi:无符号高位乘) |

## 9. 位重解释(不改变位模式)

| API | 功能备注 |
|---|---|
| `float_as_int(x)` / `int_as_float(x)` | float32 位模式 ↔ int32 |
| `float_as_uint(x)` / `uint_as_float(x)` | float32 位模式 ↔ uint32 |
| `half2float(x)` | fp16 → fp32 |
| `float2half_rn(x)` | fp32 → fp16(最近偶舍入) |

## 10. 数值类型转换(8 族 × 4 舍入方向,共 32 个)

| API | 功能备注 |
|---|---|
| `float2int_rn/rz/rd/ru(x)` | float32 → int32,四种舍入方向 |
| `float2uint_rn/rz/rd/ru(x)` | float32 → uint32,四种舍入方向 |
| `float2ll_rn/rz/rd/ru(x)` | float32 → int64,四种舍入方向 |
| `float2ull_rn/rz/rd/ru(x)` | float32 → uint64,四种舍入方向 |
| `int2float_rn/rz/rd/ru(x)` | int32 → float32,四种舍入方向 |
| `uint2float_rn/rz/rd/ru(x)` | uint32 → float32,四种舍入方向 |
| `ll2float_rn/rz/rd/ru(x)` | int64 → float32,四种舍入方向 |
| `ull2float_rn/rz/rd/ru(x)` | uint64 → float32,四种舍入方向 |

## 判定说明

- 本模块整体属于**昇腾扩展**(上游只有桩,无通用实现);函数名以 CUDA libdevice 为蓝本,
  上游 NVIDIA 后端的同名函数(`extra/cuda/libdevice.py`)由 CUDA 提供,二者 API 兼容。
- 上游 `tl.extra.libdevice` → 本模块的重映射逻辑在 `third_party/ascend/backend/compiler.py:1613-1615`
  (`get_module_map()` 返回 `{"triton.language.extra.libdevice": cann.libdevice}`)。
