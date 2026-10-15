"""数据集采集、角度编码、归一化与工作空间绘图工具。"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from dofbot import DATASET_LOWER, DATASET_UPPER, DofbotEnv


RAW_COLUMNS = [
    "q1", "q2", "q3", "q4", "q5",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "roll", "pitch", "yaw",
    "nx", "ny", "nz", "ox", "oy", "oz", "ax", "ay", "az",
]
JOINT_FEATURE_COLUMNS = [
    "q1_sin",
    "q1_cos",
    "q2_scaled",
    "q3_scaled",
    "q4_scaled",
    "q5_sin",
    "q5_cos",
]
JOINT_FEATURE_DIM = len(JOINT_FEATURE_COLUMNS)
EULER_ENCODED_COLUMNS = [
    item
    for angle in ("roll", "pitch", "yaw")
    for item in (f"{angle}_sin", f"{angle}_cos")
]
POSITION_COLUMNS = ["x", "y", "z"]
ROTATION_COLUMNS = ["nx", "ny", "nz", "ox", "oy", "oz", "ax", "ay", "az"]
POSE_COLUMNS = POSITION_COLUMNS + ROTATION_COLUMNS
POSE_DIM = len(POSE_COLUMNS)
NORM_COLUMNS = (
    JOINT_FEATURE_COLUMNS
    + POSITION_COLUMNS
    + ["qx", "qy", "qz", "qw"]
    + EULER_ENCODED_COLUMNS
    + ROTATION_COLUMNS
)


def encode_angles(angles):
    """将形状为 [..., N] 的角度编码成交错排列的 [..., 2N] 正余弦对。"""
    values = np.asarray(angles)
    pairs = np.stack((np.sin(values), np.cos(values)), axis=-1)
    return pairs.reshape(*values.shape[:-1], values.shape[-1] * 2)


def normalize_angle_pairs_tensor(encoded):
    import torch

    shape = encoded.shape
    pairs = encoded.reshape(*shape[:-1], shape[-1] // 2, 2)
    norm = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True)
    epsilon = torch.finfo(encoded.dtype).eps
    normalized = pairs / norm.clamp_min(epsilon)
    fallback = torch.zeros_like(pairs)
    fallback[..., 1] = 1.0
    pairs = torch.where(norm > epsilon, normalized, fallback)
    return pairs.reshape(shape)


def _bounded_joint_scale(joints):
    values = np.asarray(joints, dtype=np.float32)
    lower = DATASET_LOWER[1:4]
    span = DATASET_UPPER[1:4] - lower
    return 2.0 * (values[..., 1:4] - lower) / span - 1.0


def encode_joint_features(joints):
    """将 q1/q5 编码为周期正余弦对，将有界的 q2～q4 编码为标量。"""
    values = np.asarray(joints, dtype=np.float32)
    if values.shape[-1] != 5:
        raise ValueError("joints must have a final dimension of 5")
    q1 = np.stack((np.sin(values[..., 0]), np.cos(values[..., 0])), axis=-1)
    q5 = np.stack((np.sin(values[..., 4]), np.cos(values[..., 4])), axis=-1)
    return np.concatenate(
        (q1, _bounded_joint_scale(values), q5), axis=-1
    ).astype(np.float32, copy=False)


def project_joint_features_tensor(raw_features):
    import torch

    """将任意网络输出投影到有效的七维关节特征空间。"""
    if raw_features.shape[-1] != JOINT_FEATURE_DIM:
        raise ValueError(f"joint features must have dimension {JOINT_FEATURE_DIM}")
    q1 = normalize_angle_pairs_tensor(raw_features[..., 0:2])
    bounded = torch.tanh(raw_features[..., 2:5])
    q5 = normalize_angle_pairs_tensor(raw_features[..., 5:7])
    return torch.cat((q1, bounded, q5), dim=-1)


def decode_joint_features_tensor(features):
    import torch

    """将投影后的特征解码为物理关节角，不额外进行裁剪。"""
    if features.shape[-1] != JOINT_FEATURE_DIM:
        raise ValueError(f"joint features must have dimension {JOINT_FEATURE_DIM}")
    q1 = torch.atan2(features[..., 0], features[..., 1]).unsqueeze(-1)
    lower = torch.as_tensor(
        DATASET_LOWER[1:4], dtype=features.dtype, device=features.device
    )
    span = torch.as_tensor(
        DATASET_UPPER[1:4] - DATASET_LOWER[1:4],
        dtype=features.dtype,
        device=features.device,
    )
    bounded = lower + 0.5 * (features[..., 2:5] + 1.0) * span
    q5 = torch.atan2(features[..., 5], features[..., 6]).unsqueeze(-1)
    return torch.cat((q1, bounded, q5), dim=-1)


def _pose_row(position, quaternion, euler):
    rotation = Rotation.from_quat(quaternion).as_matrix()
    return list(position) + list(quaternion) + list(euler) + list(rotation.reshape(-1, order="F"))


def _worker(rank, sample_count, run_dir, flush_every, seed, progress_queue):
    chunk_path = Path(run_dir) / f"chunk_{rank:03d}.csv"
    rng = np.random.default_rng(seed + rank)
    client = p.connect(p.DIRECT)
    env = DofbotEnv(
        physics_client_id=client,
        enable_debug_ui=False,
        kinematic_only=True,
    )
    written = 0
    buffer = []
    try:
        with chunk_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(RAW_COLUMNS)
            while written < sample_count:
                joints = rng.uniform(DATASET_LOWER, DATASET_UPPER)
                env.set_kinematic_state(joints, 0.0)
                position, quaternion, euler = env.tcp_pose()
                buffer.append(list(joints) + _pose_row(position, quaternion, euler))
                written += 1
                if len(buffer) >= flush_every:
                    writer.writerows(buffer)
                    stream.flush()
                    progress_queue.put(len(buffer))
                    buffer.clear()
            if buffer:
                writer.writerows(buffer)
                progress_queue.put(len(buffer))
        return str(chunk_path), written
    finally:
        env.close()
        if p.isConnected(client):
            p.disconnect(client)


def _scan_stats(csv_paths):
    minimum = np.full(len(RAW_COLUMNS), np.inf)
    maximum = np.full(len(RAW_COLUMNS), -np.inf)
    rows = 0
    for csv_path in csv_paths:
        with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                values = np.asarray(row, dtype=np.float64)
                if values.size != len(RAW_COLUMNS):
                    raise ValueError(f"invalid row width in {csv_path}")
                minimum = np.minimum(minimum, values)
                maximum = np.maximum(maximum, values)
                rows += 1
    return minimum, maximum, rows


def _stats_payload(minimum, maximum):
    return {
        "version": 3,
        "normalization": {
            "joints": "q1_q5_sin_cos_and_q2_q4_physical_minmax",
            "position": "training_minmax_to_-1_1",
            "quaternion": "unit_xyzw",
            "rotation_matrix": "identity",
        },
        "columns": {
            name: {"min": float(minimum[index]), "max": float(maximum[index])}
            for index, name in enumerate(RAW_COLUMNS)
        },
    }


def _normalize_position(position, stats):
    minimum = np.array([stats[key]["min"] for key in POSITION_COLUMNS])
    maximum = np.array([stats[key]["max"] for key in POSITION_COLUMNS])
    span = np.where(maximum > minimum, maximum - minimum, 1.0)
    return 2.0 * (position - minimum) / span - 1.0


def _normalized_row(raw, stats):
    joints = raw[:5]
    position = _normalize_position(raw[5:8], stats)
    quaternion = raw[8:12]
    quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-8)
    euler = raw[12:15]
    rotation = raw[15:24]  # Rotation entries are already in [-1, 1].
    return np.concatenate(
        (
            encode_joint_features(joints),
            position,
            quaternion,
            encode_angles(euler),
            rotation,
        )
    )


def normalize_dofbot_dataset(raw_csv, norm_csv=None, stats_json=None):
    """根据原始数据集重新生成归一化 CSV 和配套统计文件。"""
    raw_csv = Path(raw_csv)
    norm_csv = Path(norm_csv) if norm_csv else raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_norm.csv"))
    stats_json = Path(stats_json) if stats_json else raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_norm_stats.json"))
    minimum, maximum, _ = _scan_stats([raw_csv])
    payload = _stats_payload(minimum, maximum)
    stats_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stats = payload["columns"]

    with raw_csv.open("r", newline="", encoding="utf-8") as source, norm_csv.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.reader(source)
        next(reader)
        writer = csv.writer(destination)
        writer.writerow(NORM_COLUMNS)
        for row in reader:
            writer.writerow(_normalized_row(np.asarray(row, dtype=np.float64), stats))
    return norm_csv, stats_json


def collect_dofbot_dataset(
    num_envs=2,
    num_samples=4000,
    flush_every=1000,
    seed=42,
    work_dir="dataset",
):
    """在只含机械臂且关闭碰撞的环境中采集 FK 样本。"""
    if num_envs < 1 or num_samples < 1:
        raise ValueError("num_envs and num_samples must be positive")
    run_dir = Path(work_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    counts = [num_samples // num_envs + (rank < num_samples % num_envs) for rank in range(num_envs)]
    context = mp.get_context("spawn")
    started = time.perf_counter()
    with context.Manager() as manager:
        progress_queue = manager.Queue()
        args = [
            (rank, count, str(run_dir), flush_every, seed, progress_queue)
            for rank, count in enumerate(counts)
            if count
        ]
        with context.Pool(len(args)) as pool:
            pending = pool.starmap_async(_worker, args)
            completed = 0
            while not pending.ready():
                try:
                    completed += progress_queue.get(timeout=0.5)
                except Empty:
                    continue
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = completed / elapsed
                eta = (num_samples - completed) / rate if rate else float("inf")
                print(
                    f"[COLLECT] {completed:>{len(str(num_samples))}}/{num_samples} "
                    f"({100.0 * completed / num_samples:6.2f}%) | "
                    f"{rate:8.1f} samples/s | ETA {eta:7.1f}s",
                    flush=True,
                )
            results = pending.get()
        completed = sum(count for _, count in results)
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            f"[COLLECT] {completed}/{num_samples} (100.00%) | "
            f"{completed / elapsed:.1f} samples/s | completed in {elapsed:.1f}s",
            flush=True,
        )

    print("[COLLECT] Merging worker files...", flush=True)
    chunk_paths = [Path(path) for path, _ in results]
    raw_csv = run_dir / f"dofbot_fk_{num_samples}_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(RAW_COLUMNS)
        for chunk_path in chunk_paths:
            with chunk_path.open("r", newline="", encoding="utf-8") as source:
                next(source)
                shutil.copyfileobj(source, destination)

    print("[COLLECT] Building normalized dataset and statistics...", flush=True)
    norm_csv, stats_json = normalize_dofbot_dataset(raw_csv)
    for chunk_path in chunk_paths:
        chunk_path.unlink()
    print(f"[COLLECT] Raw dataset: {raw_csv}", flush=True)
    print(f"[COLLECT] Normalized dataset: {norm_csv}", flush=True)
    print(f"[COLLECT] Statistics: {stats_json}", flush=True)
    return raw_csv, norm_csv, stats_json


def visualize_workspace(
    raw_csv,
    save_dir="results/workspace_figs",
    show=True,
    point_size=1.0,
    alpha=0.3,
):
    xyz = np.loadtxt(raw_csv, delimiter=",", skiprows=1, usecols=(5, 6, 7))
    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=point_size, alpha=alpha)
    axes.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Dofbot reachable workspace")
    axes.set_box_aspect((1, 1, 1))
    axes.zaxis.labelpad = 14
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / f"{Path(raw_csv).stem.replace('_raw', '')}_workspace.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.3)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes


class Normalizer:
    """数据集生成与模型验证共用的归一化工具。"""

    def __init__(self, stats_path):
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        if payload.get("version") != 3:
            raise ValueError(
                "legacy normalization is not supported; rebuild it from *_raw.csv "
                "with normalize_dofbot_dataset()"
            )
        self.stats = payload["columns"]

    def _parameters(self, columns):
        minimum = np.array([self.stats[key]["min"] for key in columns], dtype=np.float32)
        maximum = np.array([self.stats[key]["max"] for key in columns], dtype=np.float32)
        span = np.where(maximum > minimum, maximum - minimum, 1.0)
        return minimum, span

    def position_half_span(self):
        """返回归一化 XYZ 中一个单位对应的实际米数。"""
        _, span = self._parameters(POSITION_COLUMNS)
        return span / 2.0

    def normalize_cols(self, data, columns):
        result = np.asarray(data, dtype=np.float32).copy()
        for index, name in enumerate(columns):
            if name in POSITION_COLUMNS:
                minimum, span = self._parameters([name])
                result[..., index] = 2.0 * (result[..., index] - minimum[0]) / span[0] - 1.0
            elif name not in ROTATION_COLUMNS:
                raise ValueError(f"no model normalization is defined for column {name}")
        return result

    def denormalize_cols(self, data, columns):
        result = np.asarray(data, dtype=np.float32).copy()
        for index, name in enumerate(columns):
            if name in POSITION_COLUMNS:
                minimum, span = self._parameters([name])
                result[..., index] = (result[..., index] + 1.0) * span[0] / 2.0 + minimum[0]
            elif name not in ROTATION_COLUMNS:
                raise ValueError(f"no model denormalization is defined for column {name}")
        return result

    def normalize_cols_tensor(self, data, columns):
        import torch

        result = data.clone()
        for index, name in enumerate(columns):
            if name in POSITION_COLUMNS:
                minimum, span = self._parameters([name])
                low = torch.as_tensor(minimum[0], dtype=data.dtype, device=data.device)
                width = torch.as_tensor(span[0], dtype=data.dtype, device=data.device)
                result[..., index] = 2.0 * (result[..., index] - low) / width - 1.0
            elif name not in ROTATION_COLUMNS:
                raise ValueError(f"no model normalization is defined for column {name}")
        return result

    def denormalize_cols_tensor(self, data, columns):
        import torch

        result = data.clone()
        for index, name in enumerate(columns):
            if name in POSITION_COLUMNS:
                minimum, span = self._parameters([name])
                low = torch.as_tensor(minimum[0], dtype=data.dtype, device=data.device)
                width = torch.as_tensor(span[0], dtype=data.dtype, device=data.device)
                result[..., index] = (result[..., index] + 1.0) * width / 2.0 + low
            elif name not in ROTATION_COLUMNS:
                raise ValueError(f"no model denormalization is defined for column {name}")
        return result
