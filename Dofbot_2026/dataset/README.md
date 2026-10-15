# Dofbot 学习数据集说明

## 1. 数据采集环境

数据由 `collect_dofbot_dataset()` 在 PyBullet `DIRECT` 模式下采集。每个工作进程
使用独立的 `DofbotEnv`，并启用 `kinematic_only=True`：

- 环境中只加载机械臂；
- 不加载地面、桌子和任务物体；
- 重力为零；
- 关闭机械臂碰撞掩码；
- 直接设置关节状态，不执行动力学轨迹；
- 所有样本均由当前 URDF 中固定的 `tool_center` TCP 计算。

采样关节范围与 URDF 物理范围一致：

```text
下限：[-pi, 0, -1.05, -0.16, -pi]
上限：[ pi, pi, 4.19,  3.30,  pi]
```

## 2. 输出文件

每次采集都会创建一个时间戳目录：

```text
dataset/YYYYMMDD_HHMMSS/
    dofbot_fk_N_raw.csv
    dofbot_fk_N_norm.csv
    dofbot_fk_N_norm_stats.json
```

- `raw.csv`：原始关节角和 TCP 位姿；
- `norm.csv`：训练使用的编码、归一化数据；
- `norm_stats.json`：位置归一化所需的统计信息。

原始数据列包括：

```text
q1～q5
x、y、z
qx、qy、qz、qw
roll、pitch、yaw
nx～az（旋转矩阵按列展开）
```

## 3. 关节表示

FK 输入和 IK 输出统一使用七维关节特征：

```text
[sin(q1), cos(q1), q2_scaled, q3_scaled, q4_scaled,
 sin(q5), cos(q5)]
```

- 周期关节 `q1`、`q5` 使用正余弦对，避免 `-pi` 与 `pi` 处的不连续；
- 有界关节 `q2`～`q4` 按各自物理范围线性映射到 `[-1, 1]`；
- IK 网络输出中的正余弦对会重新单位化；
- IK 网络输出中的 `q2`～`q4` 通过 `tanh` 投影到有效特征范围；
- 解码过程直接恢复物理关节角，不进行额外软件裁剪。

## 4. 位姿表示

FK 输出和 IK 输入统一使用十二维 TCP 位姿：

```text
[x, y, z, nx, ny, nz, ox, oy, oz, ax, ay, az]
```

其中 `[n, o, a]` 是旋转矩阵的三列。XYZ 使用数据集统计值缩放到
`[-1, 1]`，旋转矩阵元素不缩放。

## 5. 模型输入输出

```text
FK：7维关节特征 → 12维 TCP 位姿
IK：12维 TCP 位姿 → 2 × 7维关节候选特征
```

IK 固定输出两个七维关节候选。实际使用时，先由学习得到的 FK 模型重建两个候选的
TCP 位姿，再选择物理位姿损失较小的候选作为最终关节解。双候选模型不兼容单候选
IK 权重，修改候选数量后必须重新训练 IK 模型。

FK 和 IK 均使用 `FlexibleMLP`，隐藏层宽度默认为 `(128, 128, 64)`。网络由
线性层、`LayerNorm`、`SiLU` 和残差块组成。

## 6. 损失函数

`PoseLossConfig` 集中定义：

```text
位置容差：10 mm
姿态容差：5°
旋转有效性权重：0.01
```

位置误差和姿态误差分别按上述物理容差归一化后等权组合。位置误差在还原 XYZ
缩放后按米计算；姿态误差使用旋转矩阵弦距离；旋转有效性项约束预测矩阵的正交性和
行列式。

FK 直接最小化预测位姿与 PyBullet 位姿之间的物理位姿损失。

IK 使用冻结的最优 FK 模型分别重建两个候选关节对应的 TCP 位姿。每个样本只取两个
候选中总损失较小的一项反向传播，即 best-of-2 损失。辅助关节监督由
`JointSupervisionSchedule` 控制：初始权重为 `0.01`，在前 20% 训练轮次内线性
衰减到零，后续训练和全部验证只使用位姿损失。

## 7. 训练配置

当前默认配置为：

```text
训练集/验证集：80% / 20%
优化器：AdamW
初始学习率：1e-3
最小学习率：1e-5
学习率调度：CosineAnnealingLR
默认 batch size：128
提前停止耐心轮数：20
全局梯度范数上限：100.0
```

训练时先生成 FK 模型，再将最优 FK 权重传给 IK 训练。最佳验证模型始终保存为
`best_model.pt`，并可按 `--checkpoint-every` 定期保存中间权重。

## 8. 验证方式

验证阶段重新从物理关节范围内生成未参与训练的可达样本：

1. 使用 PyBullet 生成真实 TCP 目标位姿；
2. FK 网络根据关节角预测位姿，并与 PyBullet 位姿比较；
3. IK 网络根据目标位姿输出两个关节候选；
4. 学习 FK 分别预测两个候选的位姿，并选择预测位姿损失较小的候选；
5. 将两个候选分别设置到 PyBullet，计算真实到达误差，并以 PyBullet 最优候选作为
   诊断用 oracle；
6. 输出最终选择结果的平均/最大误差、候选使用次数、各候选有效率、分支重复率，以及
   学习 FK 选择器与 PyBullet oracle 的一致率；
7. 使用 `10 mm / 5°` 容差判断候选是否有效，使用 `10°` 最大逐关节差判断两个有效
   候选是否属于同一关节分支。

结果图片默认保存为：

```text
results/model_results/fk_error.png
results/model_results/ik_error.png
results/model_results/ik_candidate_diagnostics.csv
```

## 9. 推荐运行命令

数据 CSV 是可重新生成的大文件，仓库只保留此说明和归一化统计参数。首次克隆后，在 `Dofbot_2026` 目录下完整执行数据采集、FK/IK 训练和验证：

```powershell
python .\dofbot_learn_student.py --collect
```

使用本地已有的 60,000 条原始数据重新训练：

```powershell
python .\dofbot_learn_student.py --raw-path "dataset\60000\dofbot_fk_60000_raw.csv"
```

只对仓库中保存的测试数据重新绘制评估图：

```powershell
python .\dofbot_learn_student.py --evaluation-only "results\learn_model\mlp_fk_20261014_225410\evaluation_data.npz"
```
