"""Dofbot 任务 0-3：改进 DH 建模、运动学计算与工作空间采样。"""

# --------------------- 导入常用库 ---------------------
import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from dofbot import PHYSICAL_LOWER, PHYSICAL_UPPER

# --------------------- 连杆参数（m） ---------------------
pi = np.pi
task0_only = "--task0-only" in sys.argv
task1_only = "--task1-only" in sys.argv
task2_only = "--task2-only" in sys.argv
task3_only = "--task3-only" in sys.argv
l1 = 0.1045
l2 = 0.08285
l3 = 0.08285
l4 = 0.12842

# ===================== 任务 0：改进 DH 建模 =====================
# 坐标系约定：O0 取底座最下方，O5 取机械臂末端最上方；z 轴沿
# 对应转动关节轴，x 轴沿相邻 z 轴的公法线，基坐标系 z0 沿 J1 轴。
# 使用改进 DH（MDH）参数：
# RevoluteMDH(a, alpha, d, offset)，关节变量 theta = q + offset。
# 几何尺寸均以米为单位，关节角及固定角偏置均以弧度为单位。
# --------------------- Demo ---------------------
DH_demo = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d=l1),  # J1
        rtb.RevoluteMDH(a=l2),  # J2
        rtb.PrismaticMDH(a=l3, alpha=pi),  # J3
        rtb.RevoluteMDH(d=l4),  # J4
    ],
    name="DH_demo",
)
print("========== DH_demo 改进 DH 参数表 ==========")
print(DH_demo)

# 数学零位展示，不代表实机复位姿态。
q_zero = [0, 0, 0, 0]
print("\n零位齐次变换矩阵：")
print(DH_demo.fkine(q_zero))
if not (task1_only or task2_only or task3_only):
    DH_demo.plot(q_zero, block=True)
# --------------------------------------------------

# 结构尺寸对应课件中的 L1-L4：L1=104.5 mm，L2=L3=82.85 mm，
# L4=128.42 mm。L2、L3 写入相邻关节轴的公法线距离 a；L1、L4
# 写入沿关节轴方向的距离 d。J2/J4 的固定 offset 用来对齐模型
# 坐标系与 Dofbot 关节的零位定义。
DH_PARAMETERS = [
    # name, a (m), alpha (rad), d (m), theta offset (rad)
    ("J1", 0.0, 0.0, l1, 0.0),
    ("J2", 0.0, -0.5 * pi, 0.0, -0.5 * pi),
    ("J3", l2, 0.0, 0.0, 0.0),
    ("J4", l3, 0.0, 0.0, 0.5 * pi),
    ("J5", 0.0, 0.5 * pi, l4, 0.0),
]

dofbot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(a=a, alpha=alpha, d=d, offset=offset)
        for _, a, alpha, d, offset in DH_PARAMETERS
    ],
    name="Dofbot",
)

print("========== Dofbot 改进 DH 参数表 ==========")
print("关节      a (m)    alpha (rad)       d (m)  offset (rad)")
for joint, a, alpha, d, offset in DH_PARAMETERS:
    print(f"{joint:<6} {a:>10.5f} {alpha:>14.5f} {d:>11.5f} {offset:>13.5f}")
print("几何尺寸：", {"L1 (m)": l1, "L2 (m)": l2, "L3 (m)": l3, "L4 (m)": l4})
print(dofbot)

# 数学零位展示，不代表实机复位姿态。
q_zero = [0, 0, 0, 0, 0]
T_zero = dofbot.fkine(q_zero).A
print("\n零位下从基座到 DH 末端坐标系的齐次变换矩阵：")
print(np.array2string(T_zero, precision=6, suppress_small=True))
if not (task1_only or task2_only or task3_only):
    dofbot.plot(q_zero, block=True)

# 允许单独检查任务 0，避免尚未完成的任务 1-3 Demo 混入本次输出。
if __name__ == "__main__" and task0_only:
    raise SystemExit(0)

# ===================== 任务 1：正运动学 =====================
# 给定角度用于数学计算，部分角度超出物理关节范围。

def show_fk_case(label, joint_angles):
    """打印一组关节角的齐次变换矩阵并显示对应姿态。"""
    transform = dofbot.fkine(joint_angles).A
    print(f"\n========== 任务 1：{label} 正解 ==========")
    print("关节角（rad）：", np.asarray(joint_angles, dtype=float))
    print("基座到 DH 末端坐标系的齐次变换矩阵：")
    print(np.array2string(transform, precision=6, suppress_small=True))
    if not (task2_only or task3_only):
        dofbot.plot(joint_angles, block=True)
    return transform

# --------------------- Demo ---------------------
q_demo = [0, pi / 3, pi / 4, pi / 5, 0]
T_demo = show_fk_case("Demo", q_demo)

# --------------------- 第 1 题 ---------------------
q_1 = [pi / 2, pi / 5, pi / 5, pi / 5, pi]
T_1 = show_fk_case("第 1 题", q_1)

# --------------------- 第 2 题 ---------------------
q_2 = [pi / 3, pi / 4, -pi / 3, -pi / 4, pi / 2]
T_2 = show_fk_case("第 2 题", q_2)

# --------------------- 第 3 题 ---------------------
q_3 = [-pi / 2, pi / 3, -2 * pi / 3, pi / 3, pi / 3]
T_3 = show_fk_case("第 3 题", q_3)

if __name__ == "__main__" and task1_only:
    raise SystemExit(0)

# ===================== 任务 2：逆运动学 =====================
# ik_LM 返回值的第 0 项为关节解，第 1 项为成功状态。
# 同时报告完整矩阵、位置和旋转矩阵残差；失败时记录常见排查方向。
q0 = [0, pi / 2, pi / 2, pi / 2, pi / 2]

def show_ik_case(label, target_transform, initial_guess):
    """求逆解，使用 FK 复算残差，并在求解成功时绘制姿态。"""
    target_transform = np.asarray(target_transform, dtype=float)
    target_rotation = target_transform[:3, :3]
    target_rotation_orthogonality_error = np.linalg.norm(
        target_rotation.T @ target_rotation - np.eye(3)
    )
    try:
        result = dofbot.ik_LM(target_transform, q0=initial_guess)
    except (ValueError, RuntimeError) as error:
        print(f"\n========== 任务 2：{label} 逆解 ==========")
        print("求解成功：False")
        print(f"求解器异常：{type(error).__name__}: {error}")
        print(
            "目标旋转矩阵正交性误差："
            f"{target_rotation_orthogonality_error:.8g}"
        )
        print("求解器未返回关节解，无法进行 FK 残差复算。")
        print(
            "请检查目标是否可达、姿态是否与 5 自由度约束相容、"
            "旋转矩阵是否因小数舍入而不再严格正交。"
        )
        return {
            "joint_solution": None,
            "success": False,
            "within_physical_limits": False,
            "reached_transform": None,
            "position_error_mm": None,
            "rotation_matrix_error": None,
            "matrix_error": None,
            "target_rotation_orthogonality_error": target_rotation_orthogonality_error,
            "exception": f"{type(error).__name__}: {error}",
        }

    joint_solution = np.asarray(result[0], dtype=float)
    success = bool(result[1])
    within_physical_limits = bool(
        np.all(joint_solution >= PHYSICAL_LOWER)
        and np.all(joint_solution <= PHYSICAL_UPPER)
    )
    reached_transform = dofbot.fkine(joint_solution).A
    residual = reached_transform - target_transform
    position_error_mm = np.linalg.norm(residual[:3, 3]) * 1000.0
    rotation_matrix_error = np.linalg.norm(residual[:3, :3])
    matrix_error = np.linalg.norm(residual)
    print(f"\n========== 任务 2：{label} 逆解 ==========")
    print("目标位姿：")
    print(np.array2string(target_transform, precision=6, suppress_small=True))
    print("关节角（rad）：", np.array2string(joint_solution, precision=8))
    print("求解成功：", success)
    print("关节解满足物理限位：", within_physical_limits)
    print("FK 复算位姿：")
    print(np.array2string(reached_transform, precision=6, suppress_small=True))
    print(f"位置残差（mm）：{position_error_mm:.6f}")
    print(f"旋转矩阵残差（Frobenius 范数）：{rotation_matrix_error:.8g}")
    print(f"齐次矩阵残差（Frobenius 范数）：{matrix_error:.8g}")
    print(
        "目标旋转矩阵正交性误差："
        f"{target_rotation_orthogonality_error:.8g}"
    )

    if success:
        dofbot.plot(joint_solution, block=True)
    else:
        print(
            "可能原因：目标位姿超出可达工作空间；5 自由度无法满足该位置与"
            "姿态组合；初值导致 LM 停在局部解；或目标旋转矩阵经过小数舍入。"
        )

    return {
        "joint_solution": joint_solution,
        "success": success,
        "within_physical_limits": within_physical_limits,
        "reached_transform": reached_transform,
        "position_error_mm": position_error_mm,
        "rotation_matrix_error": rotation_matrix_error,
        "matrix_error": matrix_error,
        "target_rotation_orthogonality_error": target_rotation_orthogonality_error,
        "exception": None,
    }


# --------------------- Demo ---------------------
T_des_demo = np.array([
    [-1, 0, 0, 0.1],
    [0, 1, 0, 0],
    [0, 0, -1, -0.1],
    [0, 0, 0, 1],
])
if not (task2_only or task3_only):
    ik_demo = show_ik_case("Demo", T_des_demo, q0)

# --------------------- 第 1 题 ---------------------
T_des_1 = np.array([
    [1, 0, 0, 0.1],
    [0, 1, 0, 0],
    [0, 0, 1, 0.1],
    [0, 0, 0, 1],
])
if not task3_only:
    ik_1 = show_ik_case("第 1 题", T_des_1, q0)

# --------------------- 第 2 题 ---------------------
T_des_2 = np.array([
    [0, 0, 1, 0.2415952047],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.2176752047],
    [0, 0, 0, 1],
])
if not task3_only:
    ik_2 = show_ik_case("第 2 题", T_des_2, q0)

# --------------------- 第 3 题 ---------------------
T_des_3 = np.array([
    [-0.866, -0.25, -0.433, -0.03704],
    [0.5, -0.433, -0.75, -0.06415],
    [0, -0.866, 0.5, 0.3073],
    [0, 0, 0, 1],
])
if not task3_only:
    ik_3 = show_ik_case("第 3 题", T_des_3, q0)

if __name__ == "__main__" and task2_only:
    raise SystemExit(0)

# ===================== 任务 3：工作空间（至少 500 点） =====================
# 在物理关节范围内独立均匀采样，不进行碰撞筛选。
num = 30000
sample_seed = 42
rng = np.random.default_rng(sample_seed)
if num < 500:
    raise ValueError("任务 3 至少需要采样 500 个关节构型。")

joint_samples = rng.uniform(
    PHYSICAL_LOWER,
    PHYSICAL_UPPER,
    size=(num, len(PHYSICAL_LOWER)),
)
points = np.empty((num, 3), dtype=float)
for index, joint_angles in enumerate(joint_samples):
    points[index] = dofbot.fkine(joint_angles).A[:3, 3]
    if (index + 1) % 5000 == 0:
        print(f"工作空间 FK 计算进度：{index + 1}/{num}")

if not np.all(joint_samples >= PHYSICAL_LOWER) or not np.all(
    joint_samples <= PHYSICAL_UPPER
):
    raise RuntimeError("采样结果超出物理关节限位。")

print("\n========== 任务 3：工作空间采样 ==========")
print(f"采样方法：各关节独立均匀采样；随机种子：{sample_seed}")
print(f"样本数：{num}；关节角单位：rad；位置单位：m")
print("末端位置数组形状：", points.shape)
for index, (lower, upper) in enumerate(zip(PHYSICAL_LOWER, PHYSICAL_UPPER), 1):
    print(
        f"J{index} 限位：[{lower:.6f}, {upper:.6f}] rad "
        f"([{np.degrees(lower):.2f}, {np.degrees(upper):.2f}] deg)"
    )
for axis, column in zip("XYZ", range(3)):
    print(
        f"{axis} 范围：[{points[:, column].min():.6f}, "
        f"{points[:, column].max():.6f}] m"
    )

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="blue", alpha=0.45)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"Dofbot Workspace ({num} samples, seed={sample_seed})")
ax.set_box_aspect(np.ptp(points, axis=0))
fig.tight_layout()

workspace_plot_path = Path("results") / "task3_workspace.png"
workspace_plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(workspace_plot_path, dpi=200, bbox_inches="tight")
print("工作空间散点图已保存：", workspace_plot_path.resolve())
plt.show()
