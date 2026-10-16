# 🧭 稀疏自编码器并不保证局部特征

🌍 Languages: [English](README.md) | [中文](README_cn.md)

👉 一句话总结。即使稀疏自编码器重建的是完全相同的表示，其特征也可能依赖不同的输入。我们提出**最优输入作用域曲线（Optimal Input Scope Curves）**，测量需要多少受控输入依赖才能解释 SAE 特征的激活。

<p align="center">
  <b>📄 arXiv 论文即将上线</b> &nbsp; | &nbsp;
  <a href="https://htmlpreview.github.io/?https://github.com/OpenSAE/SAE-Input-Localization/blob/main/docs/index.html"><b>🎞️ 5 分钟讲解</b></a> &nbsp; | &nbsp;
  <a href="#-demo"><b>🧪 Demo</b></a>
</p>

## 🎥 概览

<p align="center"><img src="assets/figures/fig1.png" alt="相同表示和 SAE，不同输入依赖" width="900"/></p>

**Figure 1.** 两个可逆输入映射向同一个冻结的 SAE 提供完全相同的表示样本，但解释特征激活所需的输入不同。在五个训练得到的 SAE 上，混合输入映射让解释 90% 激活方差所需的预算增加了 **41% 和 49%**。

---

## 🎞️ 5 分钟讲解

<p align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/OpenSAE/SAE-Input-Localization/blob/main/docs/index.html"><img src="assets/tour_preview.svg" alt="7 页交互式 5 分钟教程" width="900"/></a>
</p>

💻 点击打开浏览器中的 [交互式教程](https://htmlpreview.github.io/?https://github.com/OpenSAE/SAE-Input-Localization/blob/main/docs/index.html)

---

## ✨ 亮点

- 🧩 **相同表示，不同依赖。** 在相同 SAE 和表示分布下，r 阶输入映射所需的作用域预算可以恰好变为 1 阶的 **r 倍**，即使稀疏编码目标唯一确定了共享的字典。
- 📐 **最优作用域曲线。** 给定总输入依赖预算，计算 SAE 特征激活最少还剩多少方差无法解释，区分稀疏性与局部性。
- 📈 **五个 learned SAE 验证了这一差异。** 解释 90% 激活方差所需的平均预算从 **136.0** 增加到 **192.4** 和 **202.8**。
- 🎲 **成对干预恢复曲线。** 每个候选作用域的采样对数从 32 增加到 2,048 时，平均最大归一化曲线误差从 **0.123** 降至 **0.0109**。
- 🛡️ **局部化成本的理论下界。** 兼容性和谱界将联合输入效应、解码方向与输入特征依赖数联系起来。
- ⚡ **轻量实现。** 核心算法仅依赖 NumPy；五个 SAE 的 Demo 可以直接读取公开激活数据运行，无需重新训练或下载模型。

---

## 📐 Localize 测量什么

<details>
<summary>🧩 输入作用域与 SAE 特征</summary>

令 $`x=(x_1,\ldots,x_n)`$ 为受控输入，$`h(x)`$ 是模型表示，$`z(x)=\mathrm{Enc}(h(x))`$ 是冻结 SAE 的特征激活。为特征 $`j`$ 指定作用域 $`A_j\subseteq[n]`$，表示用于解释该特征的输入变量。总预算为

$$B(A)=\sum_j|A_j|.$$

SAE 的稀疏性计算激活了多少特征，而 $`B(A)`$ 计算输入与特征之间的依赖数量。

</details>

<details>
<summary>📉 最优输入作用域曲线</summary>

使用输入子集 $`A_j`$ 预测特征 $`z_j`$ 时，最佳预测是条件均值。遗漏的激活方差为

$$\Gamma_z(A)=\sum_j\mathbb E\left[\left(z_j-\mathbb E[z_j\mid x_{A_j}]\right)^2\right].$$

定义最优作用域曲线及其反函数：

$$\mathcal L_z(B)=\min_{B(A)\le B}\Gamma_z(A),\qquad \mathcal B_z(\tau)=\min\{B:\mathcal L_z(B)\le\tau\}.$$

$`\mathcal L_z(B)`$ 表示预算为 $`B`$ 时最少遗漏多少激活方差；$`\mathcal B_z(\tau)`$ 表示达到容差 $`\tau`$ 所需的最小预算。

</details>

<details>
<summary>🎲 成对干预估计</summary>

从乘积分布独立抽取 $`x,x'`$。对于输入作用域 $`A`$，构造 $`y=(x_A,x'_{-A})`$，则

$$\mathbb E\left[\left(z_j(x)-\mathbb E[z_j\mid x_A]\right)^2\right]=\frac12\mathbb E\left[\left(z_j(x)-z_j(y)\right)^2\right].$$

成对重采样估计给定作用域的遗漏方差；再结合作用域优化得到经验曲线。论文还给出了完成作用域选择之后仍然有效的有限样本界。

</details>

<details>
<summary>🛡️ 兼容性与谱下界</summary>

联合输入效应要求某些 SAE 特征同时依赖相应输入，并且这些特征的 decoder 方向能够表达该效应。我们据此推导出关于输入依赖数的兼容性下界和谱下界。

</details>

---

## 🔎 方法对比

受控实验固定表示样本和训练好的 SAE，只改变表示样本与输入变量的对应关系。

| 指标 | 衡量内容 | Identity 与 XOR 映射 |
|---|---|:---:|
| 重建误差 | SAE 对表示的重建质量 | 相同 |
| 激活稀疏度 | SAE 激活的特征数量 | 相同 |
| 表示与激活样本 | $`h`$ 和 $`z`$ 的样本分布 | 相同 |
| **最优输入作用域曲线** | 解释特征激活需要哪些输入 | **不同** |
| **成对输入干预** | 指定作用域遗漏的特征激活方差 | **不同** |

因此，即使 SAE 的重建与稀疏性完全一致，特征对输入的依赖结构仍然可以不同。

---

## ⚙️ 安装

```bash
git clone https://github.com/AtheavenMa/SAE-Input-Localization.git
cd SAE-Input-Localization
pip install -e .
```

核心方法仅需要 NumPy（Python 3.10+）。测试用 `pip install -e ".[dev]"`；训练实验用 `pip install -e ".[experiments]"`。

---

## 🚀 快速上手

```python
import numpy as np
from sae_localize import exact_scope_curve

# 三个二元受控输入，两个 SAE 特征。
x = np.arange(8)
z = np.column_stack([x & 1, (x >> 1) & 1]).astype(float)

# 将前两个输入混合，同时保持 SAE 激活样本完全一致。
mixed_ids = ((x & 1) ^ ((x >> 1) & 1)) | (x & ~1)
identity = exact_scope_curve(z, n_factors=3)
mixed = exact_scope_curve(z[mixed_ids], n_factors=3)

print(identity.budget(0), mixed.budget(0))
# 2 3
```

💡 相同的激活样本、相同的 SAE，但在混合输入映射下，一个特征需要同时依赖两个输入，精确作用域预算从 2 增加到 3。

🧰 **使用自己的 SAE。** 输入冻结 SAE 在完整二元受控网格上的激活数组，形状为 `(2**n_factors, n_features)`，各行按二进制输入配置排列。使用 `exact_scope_curve(z, n_factors=n)`，或者运行：

```bash
python demo/custom_activations.py my_codes.npy --n-factors 8
```

---

## 📊 主要结果

<details>
<summary>🧠 Figure 2. 同一 SAE 下的精确分离</summary>

<p align="center"><img src="assets/figures/fig2.png" alt="一阶到四阶的最优作用域曲线" width="850"/></p>

在 $`\mathbb F_7`$ 上取四个输入因子。四种输入映射保持相同表示分布和 SAE，但改变交互阶数。精确局部化所需预算随阶数 $`r`$ 成比例增加。

</details>

<details>
<summary>📈 Figure 3. Learned SAE 与有限样本恢复</summary>

<p align="center"><img src="assets/figures/fig3.png" alt="五个 SAE 的作用域曲线与成对重采样" width="850"/></p>

五组独立训练的 representation–SAE 模型，在八个二元受控输入的全部 256 种配置上评估。

| 输入映射 | 解释 90% 激活方差所需的平均预算 | 变化 |
|---|---:|---:|
| Identity | 136.0 | — |
| XOR，arity 3 | **192.4** | **+41.5%** |
| XOR，arity 5 | **202.8** | **+49.1%** |

成对干预估计随样本增加逐渐恢复精确曲线：

| 每个候选作用域的采样对数 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|
| 平均最大归一化曲线误差 | 0.123 | 0.0472 | 0.0216 | **0.0109** |

</details>

<details>
<summary>🛡️ Appendix Figure 4. 谱下界与 Feature-order 下界</summary>

<p align="center"><img src="assets/figures/fig4.png" alt="局部化预算与两种下界" width="820"/></p>

| 输入映射 | 精确预算 | 谱下界 | Feature-order 下界 |
|---|---:|---:|---:|
| Identity | 136.0 | 4.2 | 71.2 |
| XOR，arity 3 | 192.4 | 12.6 | 145.0 |
| XOR，arity 5 | 202.8 | 20.6 | 150.2 |

Feature-order 下界利用了 SAE 特征层面的交互信息，因此比只使用表示层面协方差的谱下界更紧。

</details>

---

## 🧪 Demo

使用公开的五个 SAE 激活数据运行完整对比：

```bash
python demo/demo.py
```

```text
Map          Five seed budgets            Mean ± sample SD
Identity     [128, 147, 168, 129, 108]   136.0 ± 22.6
XOR arity 3  [206, 197, 197, 186, 176]   192.4 ± 11.6
XOR arity 5  [215, 204, 205, 192, 198]   202.8 ±  8.6
```

<p align="center"><img src="assets/demo_scope_budgets.png" alt="五个 SAE 的 Demo 结果" width="900"/></p>

✅ Demo 重新优化最优作用域曲线，复现论文核心数值。重新训练全部 SAE 或重新生成图表的方法见[复现指南](docs/REPRODUCIBILITY.md)。

---

## 🗺️ 图表索引

| 图 | 主要发现 | 复现入口 |
|---|---|---|
| [Figure 1](assets/figures/fig1.png) | 相同表示和 SAE，不同输入依赖 | Overview |
| [Figure 2](assets/figures/fig2.png) | 不同交互阶数下的精确分离 | [`verify_family.py`](experiments/exact_construction/verify_family.py) |
| [Figure 3](assets/figures/fig3.png) | Learned SAE 作用域曲线与成对估计 | [`pilot.py`](experiments/learned_sae_validation/pilot.py)、[`audit.py`](experiments/learned_sae_validation/audit.py) |
| [Appendix Figure 4](assets/figures/fig4.png) | 谱下界与 Feature-order 下界 | [`code_order_bound.py`](experiments/learned_sae_validation/code_order_bound.py) |

---

## 🙏 致谢

本项目属于 [OpenSAE](https://github.com/OpenSAE) 研究体系，与 [Behavioral Compositionality (SAR)](https://github.com/OpenSAE/SAE-Behavioral-Compositionality)、[Representation Redistribution (RSS)](https://github.com/OpenSAE/Redistribution-Sensitivity-Score) 和 [SAEBase](https://github.com/OpenSAE/SAEBase) 基础设施互补。


