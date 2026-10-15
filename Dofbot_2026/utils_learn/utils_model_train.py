"""Dofbot 数据驱动正逆运动学的小批量训练工具。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils_learn.flexible_networks import FlexibleMLP
from utils_learn.utils_collect_visual import (
    JOINT_FEATURE_COLUMNS,
    JOINT_FEATURE_DIM,
    POSE_COLUMNS,
    Normalizer,
    project_joint_features_tensor,
)


IK_CANDIDATES = 2
DEFAULT_EARLY_STOPPING_PATIENCE = 20
MILLIMETRES_PER_METRE = 1000.0


@dataclass(frozen=True)
class PoseLossConfig:
    """物理位姿容差和旋转矩阵正则项配置。"""

    position_tolerance_mm: float = 10.0
    orientation_tolerance_deg: float = 5.0
    rotation_validity_weight: float = 0.01

    def __post_init__(self):
        if self.position_tolerance_mm <= 0.0:
            raise ValueError("position tolerance must be positive")
        if self.orientation_tolerance_deg <= 0.0:
            raise ValueError("orientation tolerance must be positive")
        if self.rotation_validity_weight < 0.0:
            raise ValueError("rotation-validity weight must be non-negative")

    @property
    def position_tolerance_m(self):
        return self.position_tolerance_mm / MILLIMETRES_PER_METRE

    @property
    def orientation_chordal_scale(self):
        tolerance_radians = math.radians(self.orientation_tolerance_deg)
        return 4.0 * (1.0 - math.cos(tolerance_radians))


@dataclass(frozen=True)
class TrainingConfig:
    """FK 与 IK 训练共用的具名优化器保护配置。"""

    max_gradient_norm: float = 100.0

    def __post_init__(self):
        if self.max_gradient_norm <= 0.0:
            raise ValueError("maximum gradient norm must be positive")


@dataclass(frozen=True)
class JointSupervisionSchedule:
    """初期短暂使用关节标签，随后仅通过位姿一致性训练 IK。"""

    initial_weight: float = 0.01
    decay_fraction: float = 0.20

    def __post_init__(self):
        if self.initial_weight < 0.0:
            raise ValueError("initial joint-supervision weight must be non-negative")
        if not 0.0 < self.decay_fraction < 1.0:
            raise ValueError("joint-supervision decay fraction must be in (0, 1)")

    def weight_at(self, epoch_index, total_epochs):
        if not 0 <= epoch_index < total_epochs:
            raise ValueError("epoch index must be inside the training range")
        decay_epochs = max(1, math.ceil(total_epochs * self.decay_fraction))
        if total_epochs > 1:
            decay_epochs = min(decay_epochs, total_epochs - 1)
        if epoch_index >= decay_epochs:
            return 0.0
        remaining_fraction = 1.0 - epoch_index / decay_epochs
        return self.initial_weight * remaining_fraction


DEFAULT_POSE_LOSS_CONFIG = PoseLossConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_JOINT_SUPERVISION_SCHEDULE = JointSupervisionSchedule()
POSE_ONLY_JOINT_WEIGHT = 0.0


def _rotation_matrix(flattened):
    # 数据集按 [n, o, a] 排列，即依次保存旋转矩阵的三列。
    return flattened.reshape(-1, 3, 3).transpose(1, 2)


def pose_loss_per_sample(
    prediction,
    target,
    position_half_span,
    config=DEFAULT_POSE_LOSS_CONFIG,
):
    """返回每个样本按物理容差归一化后的位姿损失。"""
    position_error_m = (
        prediction[:, :3] - target[:, :3]
    ) * position_half_span
    position_loss = position_error_m.square().sum(dim=-1) / (
        config.position_tolerance_m**2
    )
    predicted_rotation = _rotation_matrix(prediction[:, 3:])
    target_rotation = _rotation_matrix(target[:, 3:])
    orientation_loss = (
        (predicted_rotation - target_rotation).square().sum(dim=(-2, -1))
        / config.orientation_chordal_scale
    )
    identity = torch.eye(3, dtype=prediction.dtype, device=prediction.device).expand_as(predicted_rotation)
    gram = predicted_rotation.transpose(1, 2) @ predicted_rotation
    orthogonal_loss = (gram - identity).square().mean(dim=(-2, -1))
    determinant_loss = (torch.linalg.det(predicted_rotation) - 1.0).square()
    rotation_validity_loss = orthogonal_loss + determinant_loss
    pose_terms = torch.stack((position_loss, orientation_loss))
    pose_loss = pose_terms.mean(dim=0)
    component_scale = 1.0 / pose_terms.shape[0]
    position_term = component_scale * position_loss
    orientation_term = component_scale * orientation_loss
    rotation_validity_term = (
        config.rotation_validity_weight * rotation_validity_loss
    )
    total = pose_loss + rotation_validity_term
    return total, {
        "position": position_term,
        "orientation": orientation_term,
        "rotation_validity": rotation_validity_term,
    }


def compute_pose_loss(
    prediction,
    target,
    position_half_span,
    config=DEFAULT_POSE_LOSS_CONFIG,
):
    total, components = pose_loss_per_sample(
        prediction, target, position_half_span, config
    )
    return total.mean(), {name: value.mean() for name, value in components.items()}


def compute_ik_loss(
    raw_joint_candidates,
    true_joint_features,
    target_pose,
    frozen_fk,
    position_half_span,
    config=DEFAULT_POSE_LOSS_CONFIG,
    joint_supervision_weight=POSE_ONLY_JOINT_WEIGHT,
):
    if joint_supervision_weight < 0.0:
        raise ValueError("joint-supervision weight must be non-negative")
    batch_size = raw_joint_candidates.shape[0]
    expected_output_dim = IK_CANDIDATES * JOINT_FEATURE_DIM
    if raw_joint_candidates.shape[-1] != expected_output_dim:
        raise ValueError(f"IK output dimension must be {expected_output_dim}")
    candidates = raw_joint_candidates.reshape(
        batch_size, IK_CANDIDATES, JOINT_FEATURE_DIM
    )
    projected_features = project_joint_features_tensor(candidates)
    fk_inputs = projected_features.reshape(
        batch_size * IK_CANDIDATES, JOINT_FEATURE_DIM
    )
    reconstructed_pose = frozen_fk(fk_inputs)
    expanded_target = target_pose[:, None, :].expand(
        -1, IK_CANDIDATES, -1
    ).reshape(batch_size * IK_CANDIDATES, -1)
    pose_loss, pose_components = pose_loss_per_sample(
        reconstructed_pose, expanded_target, position_half_span, config
    )
    pose_loss = pose_loss.reshape(batch_size, IK_CANDIDATES)

    target_features = true_joint_features[:, None, :]
    joint_loss = (projected_features - target_features).square().mean(dim=-1)
    candidate_loss = pose_loss + joint_supervision_weight * joint_loss
    best_candidate_loss, best_candidate_index = candidate_loss.min(dim=1)

    gather_index = best_candidate_index[:, None]
    selected_components = {
        name: values.reshape(batch_size, IK_CANDIDATES)
        .gather(1, gather_index)
        .mean()
        for name, values in pose_components.items()
    }
    selected_components["joint"] = joint_loss.gather(1, gather_index).mean()
    return best_candidate_loss.mean(), selected_components


def _load_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _plot_history(history, output_path, mode):
    figure, axes = plt.subplots(figsize=(6, 4))
    training_label = "train scheduled objective" if mode == "ik" else "train"
    validation_label = "validation pose" if mode == "ik" else "validation"
    axes.plot(history["train"], label=training_label)
    axes.plot(history["validation"], label=validation_label)
    axes.set(xlabel="epoch", ylabel="loss", title="Training history")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def train_dofbot_model(
    data_path,
    stats_path,
    mode="fk",
    epochs=500,
    learning_rate=1e-3,
    min_learning_rate=1e-5,
    hidden_layers=(128, 128, 64),
    batch_size=4096,
    patience=DEFAULT_EARLY_STOPPING_PATIENCE,
    checkpoint_interval=50,
    fk_path=None,
    seed=42,
    pose_loss_config=DEFAULT_POSE_LOSS_CONFIG,
    training_config=DEFAULT_TRAINING_CONFIG,
    joint_supervision_schedule=DEFAULT_JOINT_SUPERVISION_SCHEDULE,
):
    if mode not in {"fk", "ik"}:
        raise ValueError("mode must be 'fk' or 'ik'")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = Normalizer(stats_path)
    position_half_span = torch.as_tensor(
        normalizer.position_half_span(), dtype=torch.float32, device=device
    )

    if mode == "fk":
        in_cols = JOINT_FEATURE_COLUMNS
        out_cols = POSE_COLUMNS
    else:
        in_cols = POSE_COLUMNS
        out_cols = JOINT_FEATURE_COLUMNS
        if not fk_path:
            raise ValueError("IK training requires fk_path")

    frame = pd.read_csv(data_path, usecols=list(dict.fromkeys(in_cols + out_cols)))
    inputs = torch.as_tensor(frame[in_cols].to_numpy(np.float32))
    targets = torch.as_tensor(frame[out_cols].to_numpy(np.float32))
    dataset = TensorDataset(inputs, targets)
    validation_size = max(1, int(0.2 * len(dataset)))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("dataset must contain at least two rows")
    generator = torch.Generator().manual_seed(seed)
    training_set, validation_set = random_split(
        dataset, (training_size, validation_size), generator=generator
    )
    loader_options = dict(batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    training_loader = DataLoader(training_set, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)

    output_dim = len(out_cols) * (IK_CANDIDATES if mode == "ik" else 1)
    model = FlexibleMLP(len(in_cols), output_dim, hidden_layers=hidden_layers).to(device)
    frozen_fk = None
    if mode == "ik":
        frozen_fk = FlexibleMLP(
            JOINT_FEATURE_DIM,
            len(in_cols),
            hidden_layers=hidden_layers,
        ).to(device)
        frozen_fk.load_state_dict(_load_weights(fk_path, device))
        frozen_fk.eval()
        frozen_fk.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_learning_rate
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/learn_model") / f"mlp_{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_validation = np.inf
    stale_epochs = 0
    history = {"train": [], "validation": []}
    label = mode.upper()
    started = time.perf_counter()
    print(
        f"[{label}] device={device}, samples={len(dataset)}, "
        f"train={training_size}, validation={validation_size}, "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"checkpoint_every={checkpoint_interval or 'disabled'}, "
        f"candidates={IK_CANDIDATES if mode == 'ik' else 1}",
        flush=True,
    )
    print(f"[{label}] pose_loss_config={pose_loss_config}", flush=True)
    print(f"[{label}] training_config={training_config}", flush=True)
    if mode == "ik":
        print(
            f"[{label}] joint_supervision_schedule={joint_supervision_schedule}",
            flush=True,
        )

    for epoch in range(epochs):
        joint_supervision_weight = (
            joint_supervision_schedule.weight_at(epoch, epochs)
            if mode == "ik"
            else POSE_ONLY_JOINT_WEIGHT
        )
        model.train()
        training_total = 0.0
        training_components = {}
        for batch_inputs, batch_targets in training_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs)
            if mode == "fk":
                loss, components = compute_pose_loss(
                    prediction,
                    batch_targets,
                    position_half_span,
                    pose_loss_config,
                )
            else:
                loss, components = compute_ik_loss(
                    prediction,
                    batch_targets,
                    batch_inputs,
                    frozen_fk,
                    position_half_span,
                    config=pose_loss_config,
                    joint_supervision_weight=joint_supervision_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config.max_gradient_norm,
            )
            optimizer.step()
            training_total += float(loss.detach()) * len(batch_inputs)
            for name, value in components.items():
                training_components[name] = training_components.get(name, 0.0) + (
                    float(value.detach()) * len(batch_inputs)
                )

        model.eval()
        validation_total = 0.0
        validation_components = {}
        with torch.no_grad():
            for batch_inputs, batch_targets in validation_loader:
                batch_inputs = batch_inputs.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                prediction = model(batch_inputs)
                if mode == "fk":
                    loss, components = compute_pose_loss(
                        prediction,
                        batch_targets,
                        position_half_span,
                        pose_loss_config,
                    )
                else:
                    loss, components = compute_ik_loss(
                        prediction,
                        batch_targets,
                        batch_inputs,
                        frozen_fk,
                        position_half_span,
                        config=pose_loss_config,
                        joint_supervision_weight=POSE_ONLY_JOINT_WEIGHT,
                    )
                validation_total += float(loss) * len(batch_inputs)
                for name, value in components.items():
                    validation_components[name] = validation_components.get(
                        name, 0.0
                    ) + float(value) * len(batch_inputs)

        training_loss = training_total / training_size
        validation_loss = validation_total / validation_size
        training_components = {
            name: value / training_size
            for name, value in training_components.items()
        }
        validation_components = {
            name: value / validation_size
            for name, value in validation_components.items()
        }
        history["train"].append(training_loss)
        history["validation"].append(validation_loss)
        scheduler.step()

        improved = validation_loss < best_validation
        joint_supervision_active = (
            mode == "ik"
            and joint_supervision_weight > POSE_ONLY_JOINT_WEIGHT
        )
        if improved:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        elif joint_supervision_active:
            stale_epochs = 0
        else:
            stale_epochs += 1

        elapsed = time.perf_counter() - started
        learning_rate_now = optimizer.param_groups[0]["lr"]
        if improved:
            status = "saved"
        elif joint_supervision_active:
            status = "joint warm-up"
        else:
            status = f"stale {stale_epochs}/{patience}"
        component_summary = ", ".join(
            f"{name}={training_components[name]:.6f}/{value:.6f}"
            for name, value in validation_components.items()
        )
        validation_label = "val_pose" if mode == "ik" else "val"
        joint_weight_summary = (
            f"joint_w={joint_supervision_weight:.6f} | "
            if mode == "ik"
            else ""
        )
        print(
            f"[{label}] epoch {epoch + 1:>{len(str(epochs))}}/{epochs} | "
            f"train={training_loss:.8f} | "
            f"{validation_label}={validation_loss:.8f} | "
            f"best={best_validation:.8f} | lr={learning_rate_now:.2e} | "
            f"{joint_weight_summary}"
            f"{status} | parts train/val({component_summary}) | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        completed_epoch = epoch + 1
        if checkpoint_interval and completed_epoch % checkpoint_interval == 0:
            checkpoint_path = output_dir / f"model_epoch_{completed_epoch:04d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[{label}] Checkpoint: {checkpoint_path}", flush=True)
        if stale_epochs >= patience:
            print(
                f"[{label}] Early stopping after {epoch + 1} epochs: "
                f"validation loss did not improve for {patience} epochs.",
                flush=True,
            )
            break

    model.load_state_dict(_load_weights(best_path, device))
    pd.DataFrame({"epoch": np.arange(1, len(history["train"]) + 1), **history}).to_csv(
        output_dir / "training_history.csv", index=False
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "epochs_completed": len(history["train"]),
                "epochs_requested": int(epochs),
                "early_stopped": len(history["train"]) < int(epochs),
                "best_validation_loss": float(best_validation),
                "best_epoch": int(np.argmin(history["validation"]) + 1),
                "training_samples": int(training_size),
                "validation_samples": int(validation_size),
                "batch_size": int(batch_size),
                "hidden_layers": list(hidden_layers),
                "patience": int(patience),
                "seed": int(seed),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{label}] Best validation loss: {best_validation:.8f}", flush=True)
    print(f"[{label}] Best model: {best_path}", flush=True)
    print(f"[{label}] Training history: {output_dir / 'training_history.csv'}", flush=True)
    return model, output_dir, str(best_path)
