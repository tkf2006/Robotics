# Dofbot 仿真作业

本仓库保存机器人学课程第 1、2 次大作业的 Dofbot 代码与实验报告。任务 0—3 使用改进 DH 模型和 Robotics Toolbox for Python；任务 4—6 使用 PyBullet 数据训练、评估 FK/IK 网络；任务 7 在 PyBullet 中抓取并放置物块。

## 项目文件

| 路径 | 用途 |
| --- | --- |
| `Dofbot_2026/dh_kine_student.py` | 任务 0—3：DH 建模、正逆运动学、工作空间 |
| `Dofbot_2026/dofbot_learn_student.py` | 任务 4—6：数据处理、训练、测试 |
| `Dofbot_2026/main_student.py` | 任务 7：物块抓放 |
| `Dofbot_2026/dofbot.py`、`models/` | PyBullet 环境与 URDF/STL 资源 |
| `Dofbot_2026/utils_learn/` | 网络、归一化、采集、训练和评估函数 |
| `Dofbot_2026/results/实验报告.pdf` | 实验报告；TeX 源码和插图位于同一目录 |

## 环境

建议使用 Python 3.10，在 `Dofbot_2026` 目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows 下 PyBullet 对含中文的资源绝对路径可能处理不稳。运行仿真时，建议把 `Dofbot_2026` 放在纯英文路径下，并以该目录为当前工作目录。

## 运行

以下命令均在 `Dofbot_2026` 目录执行。

```powershell
# 任务 0—3：分别运行。任务 3 会保存 results/task3_workspace.png。
python .\dh_kine_student.py --task0-only
python .\dh_kine_student.py --task1-only
python .\dh_kine_student.py --task2-only
python .\dh_kine_student.py --task3-only

# 任务 4—6：自行采集新数据并训练、测试（默认采集 120000 条）。
python .\dofbot_learn_student.py --collect

# 仅重绘本次保存的 1000 条测试数据的评估结果。
python .\dofbot_learn_student.py --evaluation-only "results\learn_model\mlp_fk_20261014_225410\evaluation_data.npz"

# 任务 7：GUI 演示；--headless 使用 DIRECT 模式。
python .\main_student.py
python .\main_student.py --headless
```

报告使用的是 60,000 条样本。原始和归一化 CSV 未放入 GitHub，可在课程提交压缩包的 `Dofbot_2026/dataset/60000/` 中找到。将其放回同名目录后，可用下列命令重新训练：

```powershell
python .\dofbot_learn_student.py --raw-path "dataset\60000\dofbot_fk_60000_raw.csv"
```

重新采集或训练会生成新的随机结果和时间戳目录，报告数值对应仓库保留的训练记录、最优权重及评估数据。任务 7 视频也保存在课程提交压缩包中，不纳入 Git 仓库。

## 编译报告

在 `Dofbot_2026/results` 目录执行：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

编译输出位于 `build/main.pdf`。仓库中的 `实验报告.pdf` 为整理后的提交版本。
