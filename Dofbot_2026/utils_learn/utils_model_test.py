"""使用 PyBullet 可达样本验证学习得到的 FK/IK 模型。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import torch

from dofbot import DofbotEnv
from utils_learn.flexible_networks import FlexibleMLP
from utils_learn.utils_collect_visual import (
    JOINT_FEATURE_DIM,
    POSE_COLUMNS,
    POSE_DIM,
    Normalizer,
    decode_joint_features_tensor,
    encode_joint_features,
    project_joint_features_tensor,
)
from utils_learn.utils_model_train import (
    DEFAULT_POSE_LOSS_CONFIG,
    IK_CANDIDATES,
    PoseLossConfig,
    pose_loss_per_sample,
)


@dataclass(frozen=True)
class CandidateDiagnosticConfig:
    """用于识别有效且不同 IK 分支的具名容差配置。"""

    pose_loss: PoseLossConfig = DEFAULT_POSE_LOSS_CONFIG
    joint_cluster_tolerance_deg: float = 10.0

    def __post_init__(self):
        if self.joint_cluster_tolerance_deg <= 0.0:
            raise ValueError("joint-cluster tolerance must be positive")


DEFAULT_CANDIDATE_DIAGNOSTICS = CandidateDiagnosticConfig()
PERIODIC_JOINT_INDICES = (0, 4)


def _load_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _flat_to_rotation(flattened):
    return np.asarray(flattened).reshape(-1, 3, 3).transpose(0, 2, 1)


def _rotation_to_flat(rotation):
    return np.asarray(rotation).transpose(0, 2, 1).reshape(-1, 9)


def _project_to_rotation(flattened):
    matrices = _flat_to_rotation(flattened)
    projected = []
    for matrix in matrices:
        left, _, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        projected.append(rotation)
    return np.asarray(projected)


def _orientation_error_degrees(predicted, target):
    relative = predicted.transpose(0, 2, 1) @ target
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _metrics(predicted_pose, target_pose):
    position_error = (
        np.linalg.norm(predicted_pose[:, :3] - target_pose[:, :3], axis=1)
        * 1000.0
    )
    predicted_rotation = _flat_to_rotation(predicted_pose[:, 3:])
    target_rotation = _flat_to_rotation(target_pose[:, 3:])
    orientation_error = _orientation_error_degrees(
        predicted_rotation,
        target_rotation,
    )
    return {
        "position_error_mm": position_error,
        "orientation_error_deg": orientation_error,
        "mean_position_error_mm": float(position_error.mean()),
        "max_position_error_mm": float(position_error.max()),
        "mean_orientation_error_deg": float(orientation_error.mean()),
        "max_orientation_error_deg": float(orientation_error.max()),
    }


def _joint_distance_degrees(left, right):
    difference = np.asarray(left) - np.asarray(right)
    difference[..., list(PERIODIC_JOINT_INDICES)] = np.arctan2(
        np.sin(difference[..., list(PERIODIC_JOINT_INDICES)]),
        np.cos(difference[..., list(PERIODIC_JOINT_INDICES)]),
    )
    return np.degrees(np.max(np.abs(difference), axis=-1))


def save_candidate_diagnostics(result, output_path):
    """为每个目标及其两个 IK 候选解分别写入一行 CSV 诊断数据。"""
    diagnostics = result["candidate_diagnostics"]
    candidate_joints = result["candidate_joints"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joint_columns = [
        f"q{index + 1}_rad" for index in range(candidate_joints.shape[-1])
    ]
    fieldnames = [
        "target_index",
        "candidate_index",
        "selected_by_learned_fk",
        "oracle_by_pybullet",
        "valid",
        "branch_id",
        "predicted_valid_branch_count",
        "learned_fk_score",
        "pybullet_pose_score",
        "position_error_mm",
        "orientation_error_deg",
        *joint_columns,
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for target_index in range(candidate_joints.shape[0]):
            for candidate_index in range(candidate_joints.shape[1]):
                row = {
                    "target_index": target_index,
                    "candidate_index": candidate_index,
                    "selected_by_learned_fk": int(
                        result["selected_candidate"][target_index]
                        == candidate_index
                    ),
                    "oracle_by_pybullet": int(
                        diagnostics["oracle_candidate"][target_index]
                        == candidate_index
                    ),
                    "valid": int(
                        diagnostics["valid_mask"][target_index, candidate_index]
                    ),
                    "branch_id": diagnostics["branch_labels"][
                        target_index, candidate_index
                    ],
                    "predicted_valid_branch_count": diagnostics[
                        "unique_valid_branch_count"
                    ][target_index],
                    "learned_fk_score": diagnostics["learned_fk_score"][
                        target_index, candidate_index
                    ],
                    "pybullet_pose_score": diagnostics["pybullet_pose_score"][
                        target_index, candidate_index
                    ],
                    "position_error_mm": diagnostics["position_error_mm"][
                        target_index, candidate_index
                    ],
                    "orientation_error_deg": diagnostics[
                        "orientation_error_deg"
                    ][target_index, candidate_index],
                }
                row.update(
                    zip(
                        joint_columns,
                        candidate_joints[target_index, candidate_index],
                    )
                )
                writer.writerow(row)
    return output


class ModelValidator:
    def __init__(
        self,
        fk_model_path,
        ik_model_path,
        stats_path,
        hidden_layers_fk=(128, 128, 64),
        hidden_layers_ik=(128, 128, 64),
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.normalizer = Normalizer(stats_path)
        self.position_half_span = torch.as_tensor(
            self.normalizer.position_half_span(),
            dtype=torch.float32,
            device=self.device,
        )
        self.fk_model = FlexibleMLP(
            JOINT_FEATURE_DIM,
            POSE_DIM,
            hidden_layers=hidden_layers_fk,
        ).to(self.device)
        self.fk_model.load_state_dict(_load_weights(fk_model_path, self.device))
        self.fk_model.eval()
        self.ik_model = FlexibleMLP(
            POSE_DIM,
            IK_CANDIDATES * JOINT_FEATURE_DIM,
            hidden_layers=hidden_layers_ik,
        ).to(self.device)
        self.ik_model.load_state_dict(_load_weights(ik_model_path, self.device))
        self.ik_model.eval()
        self.env = DofbotEnv(
            connection_mode=p.DIRECT,
            enable_debug_ui=False,
            kinematic_only=True,
        )
        self.env.reset()

    def validate_fk(self, joint_positions):
        joints = np.asarray(joint_positions, dtype=np.float32)
        if joints.ndim == 1:
            joints = joints[None, :]
        if joints.shape[1] != 5:
            raise ValueError("joint_positions must have shape [N, 5]")
        encoded = encode_joint_features(joints).astype(np.float32)
        with torch.no_grad():
            predicted_normalized = self.fk_model(
                torch.as_tensor(encoded, device=self.device)
            ).cpu().numpy()
        predicted_pose = self.normalizer.denormalize_cols(
            predicted_normalized,
            POSE_COLUMNS,
        )
        predicted_rotation = _project_to_rotation(predicted_pose[:, 3:])
        predicted_pose[:, 3:] = _rotation_to_flat(predicted_rotation)

        bullet_pose = np.empty_like(predicted_pose)
        for index, joint_sample in enumerate(joints):
            self.env.set_kinematic_state(joint_sample, 0.0)
            position, quaternion, _ = self.env.tcp_pose()
            rotation = Rotation.from_quat(quaternion).as_matrix()
            bullet_pose[index] = np.concatenate(
                (position, rotation.reshape(-1, order="F"))
            )
        return {
            "metrics": _metrics(predicted_pose, bullet_pose),
            "joint_positions": joints,
            "predicted_pose": predicted_pose,
            "bullet_pose": bullet_pose,
        }

    def validate_ik(
        self,
        target_pose,
        diagnostic_config=DEFAULT_CANDIDATE_DIAGNOSTICS,
    ):
        """使用学习 FK 选择候选，并用 PyBullet 诊断两个候选的误差。"""
        targets = np.asarray(target_pose, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[None, :]
        if targets.shape[1] != POSE_DIM:
            raise ValueError(f"target_pose must have shape [N, {POSE_DIM}]")

        normalized_targets = self.normalizer.normalize_cols(
            targets, POSE_COLUMNS
        )
        with torch.no_grad():
            target_tensor = torch.as_tensor(
                normalized_targets, device=self.device
            )
            raw_candidates = self.ik_model(target_tensor).reshape(
                len(targets), IK_CANDIDATES, JOINT_FEATURE_DIM
            )
            candidate_features = project_joint_features_tensor(raw_candidates)
            candidate_pose = self.fk_model(
                candidate_features.reshape(-1, JOINT_FEATURE_DIM)
            )
            expanded_targets = target_tensor[:, None, :].expand(
                -1, IK_CANDIDATES, -1
            ).reshape(-1, POSE_DIM)
            candidate_scores, _ = pose_loss_per_sample(
                candidate_pose,
                expanded_targets,
                self.position_half_span,
                diagnostic_config.pose_loss,
            )
            candidate_scores = candidate_scores.reshape(
                len(targets), IK_CANDIDATES
            )
            selected_candidate = candidate_scores.argmin(dim=1)
            candidate_joints = decode_joint_features_tensor(candidate_features)
        candidate_joints = candidate_joints.cpu().numpy()
        selected_candidate = selected_candidate.cpu().numpy()
        learned_fk_score = candidate_scores.cpu().numpy()

        candidate_reached_pose = np.empty(
            (len(targets), IK_CANDIDATES, POSE_DIM), dtype=np.float32
        )
        for target_index, target_candidates in enumerate(candidate_joints):
            for candidate_index, joint_sample in enumerate(target_candidates):
                self.env.set_kinematic_state(joint_sample, 0.0)
                position, quaternion, _ = self.env.tcp_pose()
                rotation = Rotation.from_quat(quaternion).as_matrix()
                candidate_reached_pose[
                    target_index, candidate_index
                ] = np.concatenate(
                    (position, rotation.reshape(-1, order="F"))
                )

        flat_reached_pose = candidate_reached_pose.reshape(-1, POSE_DIM)
        flat_target_pose = np.repeat(targets, IK_CANDIDATES, axis=0)
        candidate_metrics = _metrics(flat_reached_pose, flat_target_pose)
        position_error = candidate_metrics["position_error_mm"].reshape(
            len(targets), IK_CANDIDATES
        )
        orientation_error = candidate_metrics["orientation_error_deg"].reshape(
            len(targets), IK_CANDIDATES
        )
        normalized_reached_pose = self.normalizer.normalize_cols(
            flat_reached_pose, POSE_COLUMNS
        )
        normalized_target_pose = self.normalizer.normalize_cols(
            flat_target_pose, POSE_COLUMNS
        )
        with torch.no_grad():
            bullet_pose_score, _ = pose_loss_per_sample(
                torch.as_tensor(normalized_reached_pose, device=self.device),
                torch.as_tensor(normalized_target_pose, device=self.device),
                self.position_half_span,
                diagnostic_config.pose_loss,
            )
        bullet_pose_score = bullet_pose_score.cpu().numpy().reshape(
            len(targets), IK_CANDIDATES
        )
        oracle_candidate = bullet_pose_score.argmin(axis=1)
        valid_mask = (
            position_error
            <= diagnostic_config.pose_loss.position_tolerance_mm
        ) & (
            orientation_error
            <= diagnostic_config.pose_loss.orientation_tolerance_deg
        )

        sample_indices = np.arange(len(targets))
        predicted_joints = candidate_joints[
            sample_indices, selected_candidate
        ]
        reached_pose = candidate_reached_pose[
            sample_indices, selected_candidate
        ]
        pair_distance = _joint_distance_degrees(
            candidate_joints[:, 0], candidate_joints[:, 1]
        )
        same_configuration = (
            pair_distance <= diagnostic_config.joint_cluster_tolerance_deg
        )
        both_valid = valid_mask[:, 0] & valid_mask[:, 1]
        distinct_valid_pair = both_valid & ~same_configuration
        branch_count = valid_mask.any(axis=1).astype(int) + distinct_valid_pair
        branch_labels = np.full_like(valid_mask, -1, dtype=int)
        branch_labels[valid_mask[:, 0], 0] = 0
        branch_labels[valid_mask[:, 1], 1] = 0
        branch_labels[distinct_valid_pair, 1] = 1
        diagnostics = {
            "config": diagnostic_config,
            "position_error_mm": position_error,
            "orientation_error_deg": orientation_error,
            "learned_fk_score": learned_fk_score,
            "pybullet_pose_score": bullet_pose_score,
            "valid_mask": valid_mask,
            "branch_labels": branch_labels,
            "unique_valid_branch_count": branch_count,
            "branch_count_histogram": np.bincount(
                branch_count, minlength=IK_CANDIDATES + 1
            ),
            "candidate_valid_rate": valid_mask.mean(axis=0),
            "candidate_mean_position_error_mm": position_error.mean(axis=0),
            "candidate_mean_orientation_error_deg": orientation_error.mean(
                axis=0
            ),
            "oracle_candidate": oracle_candidate,
            "oracle_usage": np.bincount(
                oracle_candidate, minlength=IK_CANDIDATES
            ),
            "selector_oracle_match_rate": float(
                np.mean(selected_candidate == oracle_candidate)
            ),
            "targets_without_valid_candidate": int(
                np.sum(branch_count == 0)
            ),
            "maximum_unique_valid_branches": int(branch_count.max()),
            "mean_unique_valid_branches": float(branch_count.mean()),
            "duplicate_pair_fraction": float(np.mean(same_configuration)),
            "mean_selected_position_error_mm": float(
                position_error[sample_indices, selected_candidate].mean()
            ),
            "mean_oracle_position_error_mm": float(
                position_error[sample_indices, oracle_candidate].mean()
            ),
            "mean_selected_orientation_error_deg": float(
                orientation_error[sample_indices, selected_candidate].mean()
            ),
            "mean_oracle_orientation_error_deg": float(
                orientation_error[sample_indices, oracle_candidate].mean()
            ),
        }
        return {
            "metrics": _metrics(reached_pose, targets),
            "predicted_joints": predicted_joints,
            "candidate_joints": candidate_joints,
            "selected_candidate": selected_candidate,
            "candidate_usage": np.bincount(
                selected_candidate, minlength=IK_CANDIDATES
            ),
            "target_pose": targets,
            "reached_pose": reached_pose,
            "candidate_reached_pose": candidate_reached_pose,
            "candidate_diagnostics": diagnostics,
        }

    @staticmethod
    def plot(metrics, save_path=None):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].boxplot(metrics["position_error_mm"])
        axes[0].set(title="Position error", ylabel="mm", xticks=[])
        axes[1].boxplot(metrics["orientation_error_deg"])
        axes[1].set(title="Orientation error", ylabel="degrees", xticks=[])
        figure.tight_layout()
        if save_path:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=200)
            plt.close(figure)
        return figure

    def close(self):
        self.env.close()
