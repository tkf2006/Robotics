"""Dofbot 模型学习：训练与评估。"""

# --------------------- 导入库 ---------------------
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

# 此脚本中的训练曲线和误差图写入文件，不需要 Tk 窗口；Agg 也能避免
# 在无图形桌面或远程终端中导入/初始化 Tk 后端时意外退出。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from dofbot import DATASET_LOWER, DATASET_UPPER, DofbotEnv
from utils_learn.utils_collect_visual import (
    Normalizer, POSE_COLUMNS, collect_dofbot_dataset,
    normalize_dofbot_dataset, visualize_workspace,
    encode_joint_features,
)

# --------------------- 参数设置 ---------------------
raw_path = Path("dataset/60000/dofbot_fk_60000_raw.csv")
hidden_layers = (128, 128, 64)
epochs = 200
batch_size = 128
test_count = 1000
test_seed = 2025
position_threshold_mm = 10.0
orientation_threshold_deg = 5.0

# --------------------- 任务：编写评估函数 ---------------------
def evaluate_pose(predicted_pose, target_pose):
    """输入两个 [N, 12] 数组；返回指标字典和误差分布图 (metrics, figure)。

    前三列为 XYZ（m），后九列为旋转矩阵按列展开：
    [nx, ny, nz, ox, oy, oz, ax, ay, az]。
    姿态比较前将旋转矩阵投影到 SO(3)，以抵消预测数值误差。
    FK 比较预测位姿与真值；IK 比较实际到达位姿与目标。
    """
    predicted_pose = np.asarray(predicted_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if predicted_pose.ndim != 2 or predicted_pose.shape[1] != 12:
        raise ValueError("predicted_pose 必须为 [N, 12]")
    if target_pose.shape != predicted_pose.shape:
        raise ValueError("target_pose 必须与 predicted_pose 同为 [N, 12]")
    if len(predicted_pose) == 0 or not (
        np.all(np.isfinite(predicted_pose)) and np.all(np.isfinite(target_pose))
    ):
        raise ValueError("位姿数组不能为空且必须全部为有限数")

    # 姿态九维按列 [n,o,a] 展开；SVD 投影可处理网络输出的微小非正交误差。
    def rotation_matrices(poses):
        matrices = poses[:, 3:].reshape(-1, 3, 3).transpose(0, 2, 1)
        result = np.empty_like(matrices)
        for index, matrix in enumerate(matrices):
            left, _, right = np.linalg.svd(matrix)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(left @ right)
            result[index] = left @ correction @ right
        return result

    position_error_mm = np.linalg.norm(
        predicted_pose[:, :3] - target_pose[:, :3], axis=1
    ) * 1000.0
    predicted_rotation = rotation_matrices(predicted_pose)
    target_rotation = rotation_matrices(target_pose)
    relative = np.swapaxes(predicted_rotation, 1, 2) @ target_rotation
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    orientation_error_deg = np.degrees(np.arccos(cosine))

    def summarize(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    position_ok = position_error_mm <= position_threshold_mm
    orientation_ok = orientation_error_deg <= orientation_threshold_deg
    metrics = {
        "sample_count": int(len(predicted_pose)),
        "position_error_mm": summarize(position_error_mm),
        "orientation_error_deg": summarize(orientation_error_deg),
        "position_threshold_mm": float(position_threshold_mm),
        "orientation_threshold_deg": float(orientation_threshold_deg),
        "position_success_rate": float(np.mean(position_ok)),
        "orientation_success_rate": float(np.mean(orientation_ok)),
        "joint_success_rate": float(np.mean(position_ok & orientation_ok)),
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(position_error_mm, bins=40, color="royalblue", alpha=0.82)
    axes[0].axvline(position_threshold_mm, color="crimson", linestyle="--", label="10 mm threshold")
    axes[0].set(xlabel="Position error (mm)", ylabel="Sample count", title="Position error")
    axes[0].legend()
    axes[1].hist(orientation_error_deg, bins=40, color="darkcyan", alpha=0.82)
    axes[1].axvline(orientation_threshold_deg, color="crimson", linestyle="--", label="5° threshold")
    axes[1].set(xlabel="Orientation error (deg)", ylabel="Sample count", title="Orientation error")
    axes[1].legend()
    figure.suptitle(f"Pose evaluation ({len(predicted_pose)} samples)")
    figure.tight_layout()
    return metrics, figure


# --------------------- 已提供：生成预测值和仿真真值 ---------------------
# 只准备数据，评价指标在 evaluate_pose() 中计算。
def generate_evaluation_data(fk_model, ik_model, stats_path):
    # PyTorch 只在训练/模型评估时加载。任务 4 的数据预处理不需要它，
    # 延迟导入可避免 Windows 上与 PyBullet/数值库 OpenMP runtime 冲突。
    import torch

    from utils_learn.utils_collect_visual import (
        decode_joint_features_tensor, project_joint_features_tensor,
    )
    from utils_learn.utils_model_train import pose_loss_per_sample

    normalizer = Normalizer(stats_path)
    device = next(fk_model.parameters()).device
    fk_model.eval()
    ik_model.eval()
    rng = np.random.default_rng(test_seed)
    joints = rng.uniform(DATASET_LOWER, DATASET_UPPER, size=(test_count, 5))
    truth = np.empty((test_count, 12), dtype=np.float32)
    reached = np.empty_like(truth)
    env = DofbotEnv(connection_mode=p.DIRECT, enable_debug_ui=False, kinematic_only=True)
    try:
        # 独立生成可达目标，不使用训练集或训练过程中的验证集。
        for i, q in enumerate(joints):
            env.set_kinematic_state(q, 0.0)
            position, quaternion, _ = env.tcp_pose()
            rotation = Rotation.from_quat(quaternion).as_matrix()
            truth[i] = np.concatenate((position, rotation.reshape(-1, order="F")))

        with torch.no_grad():
            joint_tensor = torch.as_tensor(encode_joint_features(joints), device=device)
            prediction_tensor = fk_model(joint_tensor)
            prediction_tensor = normalizer.denormalize_cols_tensor(
                prediction_tensor, POSE_COLUMNS
            )

            # 用 PyTorch 的 SVD 投影到 SO(3)，避免训练进程触发 NumPy
            # MKL 与 PyTorch OpenMP runtime 的冲突。
            matrices = prediction_tensor[:, 3:].reshape(-1, 3, 3).transpose(1, 2)
            left, _, right_t = torch.linalg.svd(matrices)
            handedness = torch.linalg.det(left @ right_t)
            correction = torch.eye(3, dtype=matrices.dtype, device=device).expand_as(matrices).clone()
            correction[:, 2, 2] = handedness
            projected = left @ correction @ right_t
            prediction_tensor[:, 3:] = projected.transpose(1, 2).reshape(-1, 9)
            prediction = prediction_tensor.cpu().numpy()

            # 保留 2026 双候选 IK，由学习 FK 选解，不用仿真真值挑选候选。
            target_tensor = torch.as_tensor(
                normalizer.normalize_cols(truth, POSE_COLUMNS), device=device
            )
            candidates = project_joint_features_tensor(
                ik_model(target_tensor).reshape(-1, 2, 7)
            )
            candidate_pose = fk_model(candidates.reshape(-1, 7))
            repeated_target = target_tensor[:, None, :].expand(-1, 2, -1).reshape(-1, 12)
            half_span = torch.as_tensor(
                normalizer.position_half_span(), dtype=torch.float32, device=device
            )
            scores, _ = pose_loss_per_sample(candidate_pose, repeated_target, half_span)
            selected = scores.reshape(-1, 2).argmin(dim=1).cpu().numpy()
            candidate_joints = decode_joint_features_tensor(candidates).cpu().numpy()
            ik_joints = candidate_joints[np.arange(test_count), selected]

        # IK 必须用 PyBullet 计算所选关节解的实际运动学位姿。
        for i, q in enumerate(ik_joints):
            env.set_kinematic_state(q, 0.0)
            position, quaternion, _ = env.tcp_pose()
            rotation = Rotation.from_quat(quaternion).as_matrix()
            reached[i] = np.concatenate((position, rotation.reshape(-1, order="F")))
    finally:
        env.close()
    return {
        "fk_prediction": prediction, "fk_truth": truth,
        "ik_target": truth.copy(), "ik_reached": reached,
        "test_joints": joints, "ik_joints": ik_joints,
        "selected_candidate": selected,
    }


# --------------------- 任务 4–6 的命令行流程 ---------------------
def prepare_dataset(raw_csv):
    raw_csv = Path(raw_csv)
    if not raw_csv.is_file():
        raise FileNotFoundError(f"找不到原始数据集：{raw_csv}")
    normalized_csv, stats_json = normalize_dofbot_dataset(raw_csv)
    figure, _ = visualize_workspace(raw_csv, show=False)
    plt.close(figure)
    rows = 0
    joint_min = np.full(5, np.inf)
    joint_max = np.full(5, -np.inf)
    with raw_csv.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            joints = np.array([float(row[f"q{i}"]) for i in range(1, 6)])
            joint_min = np.minimum(joint_min, joints)
            joint_max = np.maximum(joint_max, joints)
            rows += 1
    print(f"原始数据：{raw_csv}；样本数：{rows}")
    print("数据采样：各关节在 DATASET_LOWER～DATASET_UPPER 内独立均匀采样；PyBullet DIRECT、纯运动学、无碰撞筛选")
    print("关节范围（数据观测值，rad）：")
    for index in range(5):
        print(f"  J{index + 1}: [{joint_min[index]:.6f}, {joint_max[index]:.6f}]")
    print(f"归一化数据：{normalized_csv}")
    print(f"归一化统计：{stats_json}")
    print("工作空间图：results/workspace_figs/" + raw_csv.stem.replace("_raw", "") + "_workspace.png")
    return normalized_csv, stats_json


def evaluate_saved_data(evaluation_path):
    evaluation_path = Path(evaluation_path)
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"找不到评估数据：{evaluation_path}")
    with np.load(evaluation_path, allow_pickle=False) as data:
        fk_prediction = data["fk_prediction"].copy()
        fk_truth = data["fk_truth"].copy()
        ik_reached = data["ik_reached"].copy()
        ik_target = data["ik_target"].copy()

    fk_metrics, fk_figure = evaluate_pose(fk_prediction, fk_truth)
    ik_metrics, ik_figure = evaluate_pose(ik_reached, ik_target)
    result_dir = evaluation_path.parent / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    fk_figure.savefig(result_dir / "fk_error.png", dpi=200, bbox_inches="tight")
    ik_figure.savefig(result_dir / "ik_error.png", dpi=200, bbox_inches="tight")
    plt.close(fk_figure)
    plt.close(ik_figure)
    report = {"FK": fk_metrics, "IK": ik_metrics}
    (result_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("评估结果目录：", result_dir)


def plot_training_curves(run_info):
    """在 PyTorch 子进程结束后绘制曲线，隔离 Windows OpenMP runtime。"""
    for mode, output_dir in (("fk", run_info["fk_dir"]), ("ik", run_info["ik_dir"])):
        output_dir = Path(output_dir)
        with (output_dir / "training_history.csv").open(
            "r", newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        epochs_seen = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train"]) for row in rows]
        validation_loss = [float(row["validation"]) for row in rows]
        figure, axes = plt.subplots(figsize=(6, 4))
        axes.plot(epochs_seen, train_loss, label="train scheduled objective" if mode == "ik" else "train")
        axes.plot(epochs_seen, validation_loss, label="validation pose" if mode == "ik" else "validation")
        axes.set(xlabel="epoch", ylabel="loss", title=f"{mode.upper()} training history")
        axes.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "training_curve.png", dpi=200)
        plt.close(figure)
        print("训练曲线：", output_dir / "training_curve.png")


def train_worker(raw_csv):
    """独立子进程执行 PyTorch 训练与测试位姿生成。"""
    from utils_learn.utils_model_train import train_dofbot_model

    raw_csv = Path(raw_csv)
    norm_path = raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_norm.csv"))
    stats_path = raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_norm_stats.json"))
    if not norm_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("训练子进程需要先运行数据归一化")
    fk_model, fk_dir, fk_path = train_dofbot_model(
        norm_path, stats_path, mode="fk", hidden_layers=hidden_layers,
        epochs=epochs, batch_size=batch_size, patience=20, checkpoint_interval=20,
    )
    ik_model, ik_dir, ik_path = train_dofbot_model(
        norm_path, stats_path, mode="ik", hidden_layers=hidden_layers,
        epochs=epochs, batch_size=batch_size, patience=20, checkpoint_interval=20,
        fk_path=fk_path,
    )
    evaluation_path = Path(fk_dir) / "evaluation_data.npz"
    evaluation_data = generate_evaluation_data(fk_model, ik_model, stats_path)
    np.savez_compressed(evaluation_path, **evaluation_data)
    def recorded_path(path):
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(Path(__file__).resolve().parent).as_posix()
        except ValueError:
            return resolved.name

    config = {
        "raw_path": recorded_path(raw_csv), "stats_path": recorded_path(stats_path),
        "fk_model": recorded_path(fk_path), "ik_model": recorded_path(ik_path),
        "hidden_layers": list(hidden_layers), "epochs": epochs,
        "batch_size": batch_size, "test_count": test_count, "test_seed": test_seed,
        "training_seed": 42, "learning_rate": 0.001, "patience": 20,
        "checkpoint_interval": 20, "fk_rotation": "PyTorch SVD projection to SO(3)",
        "ik_selection": "learned FK pose loss, best of two",
    }
    evaluation_path.with_name("experiment.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_info = {"fk_dir": str(Path(fk_dir).resolve()),
                "ik_dir": str(Path(ik_dir).resolve()),
                "evaluation_path": str(evaluation_path.resolve())}
    manifest = Path("results/learn_model/latest_run.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("FK 结果目录：", fk_dir)
    print("IK 结果目录：", ik_dir)
    print("独立测试数据已保存：", evaluation_path)


def main():
    parser = argparse.ArgumentParser(description="Dofbot 学习运动学任务 4–6")
    parser.add_argument("--raw-path", type=Path, default=raw_path, help="原始 FK CSV 路径")
    parser.add_argument("--prepare-only", action="store_true", help="只执行任务 4：归一化和工作空间可视化")
    parser.add_argument("--evaluation-only", type=Path, help="只执行任务 6：输入已保存的 evaluation_data.npz")
    parser.add_argument("--collect", action="store_true", help="任务 4 自行采集 120000 个 FK 样本后再处理")
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.evaluation_only:
        evaluate_saved_data(args.evaluation_only)
        return
    if args.train_worker:
        train_worker(args.raw_path)
        return
    selected_raw = args.raw_path
    if args.collect:
        selected_raw, _, _ = collect_dofbot_dataset(num_envs=4, num_samples=120000)
    norm_path, stats_path = prepare_dataset(selected_raw)
    if args.prepare_only:
        return

    # Matplotlib/NumPy 线性代数和 PyTorch 的 OpenMP DLL 在 Windows 上冲突，
    # 因此训练与 PyBullet 测试位姿生成放到新进程，父进程负责绘图和评估。
    worker_command = [
        sys.executable, str(Path(__file__).resolve()), "--train-worker",
        "--raw-path", str(Path(selected_raw).resolve()),
    ]
    subprocess.run(worker_command, check=True)
    run_info = json.loads(Path("results/learn_model/latest_run.json").read_text(encoding="utf-8"))
    plot_training_curves(run_info)
    evaluate_saved_data(run_info["evaluation_path"])


if __name__ == "__main__":
    main()

# 报告：实验配置、训练曲线、实际训练轮数、评估公式与理由、误差图及结果分析。
# 注意 IK 初期训练损失含辅助关节项，验证损失仅含位姿项，不能直接用两者差值判断过拟合。
# IK 关节解可能不唯一；本任务评价位姿，不以关节标签差异直接判错。
