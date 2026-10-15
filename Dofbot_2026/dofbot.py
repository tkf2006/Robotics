"""Dofbot 仿真任务共用的 PyBullet 环境。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pybullet as p


ARM_JOINTS = (0, 1, 2, 3, 4)
GRIPPER_JOINTS = (6, 7, 8, 9, 10, 11)

# IK 和动力学控制使用的物理/URDF 关节范围。连续关节统一表示为一个标准周期，
# 使学习数据与轨迹插值具有唯一且明确的数值范围。
PHYSICAL_LOWER = np.array([-np.pi, 0.0, -1.05, -0.16, -np.pi])
PHYSICAL_UPPER = np.array([np.pi, np.pi, 4.19, 3.30, np.pi])

# 学习、验证和 GUI 采样使用与物理控制及 IK 相同的标准关节范围。
DATASET_LOWER = PHYSICAL_LOWER.copy()
DATASET_UPPER = PHYSICAL_UPPER.copy()

REST_POSE = np.array([np.pi / 2.0] * 5)
GRIPPER_LIMITS = (-0.785, 0.785)
BASE_DIR = Path(__file__).resolve().parent


def _as_quaternion(quaternion):
    if quaternion is None:
        return None
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,):
        raise ValueError("orientation must be a quaternion in [x, y, z, w] order")
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("orientation quaternion must be non-zero")
    return (value / norm).tolist()


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _as_arm_positions(joint_positions):
    q = np.asarray(joint_positions, dtype=float).copy()
    if q.shape != (5,):
        raise ValueError("Dofbot requires exactly five arm joint positions")
    if not np.all(np.isfinite(q)):
        raise ValueError("arm joint positions must be finite")
    return q


def clip_arm_positions(joint_positions, lower=PHYSICAL_LOWER, upper=PHYSICAL_UPPER):
    q = _as_arm_positions(joint_positions)
    q[[0, 4]] = wrap_to_pi(q[[0, 4]])
    return np.clip(q, lower, upper)


@dataclass(frozen=True)
class Observation:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    @property
    def euler_xyz(self):
        return np.asarray(p.getEulerFromQuaternion(self.quaternion_xyzw.tolist()))


class Dofbot:
    def __init__(self, urdf_path, physics_client_id, collision_enabled=True):
        self.cid = physics_client_id
        self.lower_limits = PHYSICAL_LOWER.tolist()
        self.upper_limits = PHYSICAL_UPPER.tolist()
        self.joint_ranges = (PHYSICAL_UPPER - PHYSICAL_LOWER).tolist()
        self.rest_pose = REST_POSE.tolist()
        self.start_joint_positions = REST_POSE.copy()
        self.gripper_angle = 0.0

        self.robot_id = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=0,
            physicsClientId=self.cid,
        )
        if not collision_enabled:
            for link_index in range(
                -1, p.getNumJoints(self.robot_id, physicsClientId=self.cid)
            ):
                p.setCollisionFilterGroupMask(
                    self.robot_id,
                    link_index,
                    collisionFilterGroup=0,
                    collisionFilterMask=0,
                    physicsClientId=self.cid,
                )
        self.tcp_link_index = self._find_link_index("tool_center")
        self._tcp_position = np.zeros(3)
        self._tcp_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self._tcp_euler = np.zeros(3)
        self.reset()

    def _find_link_index(self, link_name):
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.cid)
            if info[12].decode("utf-8") == link_name:
                return joint_index
        raise RuntimeError(f"URDF link not found: {link_name}")

    def reset(self):
        for joint_index, value in zip(ARM_JOINTS, self.start_joint_positions):
            p.resetJointState(
                self.robot_id, joint_index, float(value), physicsClientId=self.cid
            )
        self.set_gripper_state(0.0)
        self.pose()

    def set_joint_state(self, joint_positions):
        q = clip_arm_positions(joint_positions)
        for joint_index, value in zip(ARM_JOINTS, q):
            p.resetJointState(
                self.robot_id, joint_index, float(value), physicsClientId=self.cid
            )
        return q

    def command_joints(self, joint_positions):
        # 动力学控制指令不进行软件裁剪，旋转关节限位仍由 URDF 和物理引擎执行。
        q = _as_arm_positions(joint_positions)
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=list(ARM_JOINTS),
            controlMode=p.POSITION_CONTROL,
            targetPositions=q.tolist(),
            forces=[200.0] * len(ARM_JOINTS),
            positionGains=[0.3] * len(ARM_JOINTS),
            velocityGains=[1.0] * len(ARM_JOINTS),
            physicsClientId=self.cid,
        )
        return q

    def set_gripper_state(self, angle):
        value = float(np.clip(angle, *GRIPPER_LIMITS))
        for joint_index in GRIPPER_JOINTS:
            p.resetJointState(
                self.robot_id, joint_index, value, physicsClientId=self.cid
            )
        self.gripper_angle = value
        return value

    def command_gripper(self, angle):
        value = float(np.clip(angle, *GRIPPER_LIMITS))
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=list(GRIPPER_JOINTS),
            controlMode=p.POSITION_CONTROL,
            targetPositions=[value] * len(GRIPPER_JOINTS),
            forces=[2.5] * len(GRIPPER_JOINTS),
            physicsClientId=self.cid,
        )
        self.gripper_angle = value
        return value

    def solve_ik(self, position, orientation=None):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,):
            raise ValueError("target position must contain x, y, z")

        kwargs = dict(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tcp_link_index,
            targetPosition=position.tolist(),
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_pose,
            maxNumIterations=150,
            residualThreshold=1e-6,
            physicsClientId=self.cid,
        )
        quaternion = _as_quaternion(orientation)
        if quaternion is not None:
            kwargs["targetOrientation"] = quaternion
        solution = p.calculateInverseKinematics(**kwargs)
        # 原样返回求解结果，使调用方能够发现关节越界，而不是静默计算被裁剪后的关节角。
        return _as_arm_positions(solution[:5])

    def joint_positions(self):
        q = [
            p.getJointState(self.robot_id, index, physicsClientId=self.cid)[0]
            for index in ARM_JOINTS
        ]
        return q, self.gripper_angle

    def pose(self):
        state = p.getLinkState(
            self.robot_id,
            self.tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.cid,
        )
        self._tcp_position = np.asarray(state[4], dtype=float)
        self._tcp_quaternion = np.asarray(state[5], dtype=float)
        self._tcp_euler = np.asarray(
            p.getEulerFromQuaternion(self._tcp_quaternion.tolist()), dtype=float
        )
        return self._tcp_position.copy(), self._tcp_quaternion.copy(), self._tcp_euler.copy()

    def observation(self):
        position, quaternion, _ = self.pose()
        return Observation(position, quaternion)


class Object:
    def __init__(
        self,
        urdf_path,
        physics_client_id,
        initial_position,
        initial_orientation=(0.0, 0.0, 0.0, 1.0),
        fixed=False,
        collision_enabled=True,
    ):
        self.cid = physics_client_id
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.initial_orientation = _as_quaternion(initial_orientation)
        self.id = p.loadURDF(
            str(urdf_path),
            useFixedBase=fixed,
            physicsClientId=self.cid,
        )
        self.half_extents = self._measure_half_extents()
        self.size = self.half_extents * 2.0
        self.half_height = float(self.half_extents[2])
        if not collision_enabled:
            p.setCollisionFilterGroupMask(
                self.id, -1, 0, 0, physicsClientId=self.cid
            )
        self.reset()

    def _measure_half_extents(self):
        lower, upper = p.getAABB(self.id, -1, physicsClientId=self.cid)
        return (np.asarray(upper) - np.asarray(lower)) / 2.0

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.id,
            self.initial_position.tolist(),
            self.initial_orientation,
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self.id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.cid,
        )

    def observation(self):
        position, quaternion, _ = self.pose()
        return Observation(np.asarray(position), np.asarray(quaternion))

    def pose(self):
        position, quaternion = p.getBasePositionAndOrientation(
            self.id, physicsClientId=self.cid
        )
        euler = p.getEulerFromQuaternion(quaternion)
        return np.asarray(position), np.asarray(quaternion), np.asarray(euler)


def _adjacent_link_pairs(robot_uid, physics_client_id):
    pairs = set()
    for child in range(p.getNumJoints(robot_uid, physicsClientId=physics_client_id)):
        parent = p.getJointInfo(robot_uid, child, physicsClientId=physics_client_id)[16]
        pairs.add(frozenset((parent, child)))
    return pairs


def any_self_collision(robot_uid, safety_margin=0.0, physics_client_id=0):
    adjacent = _adjacent_link_pairs(robot_uid, physics_client_id)
    points = p.getClosestPoints(
        robot_uid,
        robot_uid,
        distance=max(0.0, float(safety_margin)),
        physicsClientId=physics_client_id,
    )
    for point in points:
        link_a, link_b = point[3], point[4]
        if link_a == link_b or frozenset((link_a, link_b)) in adjacent:
            continue
        if point[8] <= safety_margin:
            return True
    return False


def check_pairwise_collisions(bodies, physics_client_id=0, margin=0.0):
    return any(
        p.getClosestPoints(a, b, margin, physicsClientId=physics_client_id)
        for a, b in combinations(bodies, 2)
    )


class DofbotEnv:
    def __init__(
        self,
        physics_client_id=None,
        connection_mode=p.GUI,
        enable_debug_ui=None,
        load_task_objects=True,
        kinematic_only=False,
    ):
        self._time_step = 0.001
        self._owns_connection = physics_client_id is None
        self.physics_client_id = (
            p.connect(connection_mode) if physics_client_id is None else physics_client_id
        )
        if enable_debug_ui is None:
            enable_debug_ui = self._owns_connection and connection_mode == p.GUI
        self.enable_debug_ui = bool(enable_debug_ui)
        self.kinematic_only = bool(kinematic_only)
        self.load_task_objects = bool(load_task_objects and not self.kinematic_only)

        p.setAdditionalSearchPath(str(BASE_DIR), physicsClientId=self.physics_client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=150, physicsClientId=self.physics_client_id
        )
        p.setTimeStep(self._time_step, physicsClientId=self.physics_client_id)
        gravity_z = 0.0 if self.kinematic_only else -9.8
        p.setGravity(0, 0, gravity_z, physicsClientId=self.physics_client_id)
        if self.enable_debug_ui:
            p.resetDebugVisualizerCamera(
                1.0, 90, -40, [0, 0, 0], physicsClientId=self.physics_client_id
            )

        self.floor_id = None
        self.table_id = None
        if not self.kinematic_only:
            self.floor_id = p.loadURDF(
                "models/floor.urdf",
                [0, 0, -0.625],
                useFixedBase=True,
                physicsClientId=self.physics_client_id,
            )
            self.table_id = p.loadURDF(
                "models/table_collision/table.urdf",
                [0.5, 0, -0.625],
                useFixedBase=True,
                physicsClientId=self.physics_client_id,
            )
        self._robot = Dofbot(
            # 传给 loadURDF 的路径只使用 ASCII 字符。Windows 下 PyBullet 原生加载器
            # 可能错误处理含中文的绝对路径；上面设置的搜索路径仍可保证加载过程不依赖
            # 当前工作目录。
            "models/dofbot_urdf_with_gripper/dofbot_with_gripper.urdf",
            self.physics_client_id,
            collision_enabled=not self.kinematic_only,
        )

        self._object1 = None
        self._object2 = None
        if self.load_task_objects:
            self._object1 = Object(
                "models/box_green.urdf",
                self.physics_client_id,
                [0.20, 0.10, 0.015],
                p.getQuaternionFromEuler([0, 0, np.pi / 6]),
            )
            self._object2 = Object(
                "models/box_purple.urdf",
                self.physics_client_id,
                [0.20, -0.10, 0.0005],
                fixed=True,
                collision_enabled=False,
            )
            marker_pos, _, _ = self._object2.pose()
            self._target_position = np.array(
                [marker_pos[0], marker_pos[1], self._object1.half_height]
            )
        else:
            self._target_position = None

        self._success_frames = 0
        self._debug_line_ids = {}
        self._debug_text_ids = {}
        if self.enable_debug_ui:
            self._create_debug_ui()

    def _create_debug_ui(self):
        self.control_mode_slider = p.addUserDebugParameter(
            "Control mode (0=joint, 1=TCP)", 0, 1, 0,
            physicsClientId=self.physics_client_id,
        )
        self.sliders = [
            p.addUserDebugParameter(
                f"Joint {index + 1}",
                float(DATASET_LOWER[index]),
                float(DATASET_UPPER[index]),
                float(REST_POSE[index]),
                physicsClientId=self.physics_client_id,
            )
            for index in range(5)
        ]
        self.gripper_slider = p.addUserDebugParameter(
            "Gripper",
            GRIPPER_LIMITS[0],
            GRIPPER_LIMITS[1],
            0.0,
            physicsClientId=self.physics_client_id,
        )
        position, _, euler = self.tcp_pose()
        xyz_ranges = [(-0.30, 0.45), (-0.35, 0.35), (-0.10, 0.50)]
        self.tcp_position_sliders = [
            p.addUserDebugParameter(
                name, low, high, float(value), physicsClientId=self.physics_client_id
            )
            for name, (low, high), value in zip(("TCP x", "TCP y", "TCP z"), xyz_ranges, position)
        ]
        self.tcp_euler_sliders = [
            p.addUserDebugParameter(
                name, -np.pi, np.pi, float(value), physicsClientId=self.physics_client_id
            )
            for name, value in zip(("TCP roll", "TCP pitch", "TCP yaw"), euler)
        ]

    def reset(self):
        if self._object1 is not None:
            self._object1.reset()
            self._object2.reset()
        self._robot.reset()
        self._success_frames = 0
        self.step_simulation()

    def close(self):
        if self._owns_connection and p.isConnected(self.physics_client_id):
            p.disconnect(self.physics_client_id)

    def step_simulation(self, steps=1, real_time=None):
        if real_time is None:
            real_time = self.enable_debug_ui
        for _ in range(int(steps)):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            if real_time:
                time.sleep(self._time_step)

    def has_collision(self, safety_margin=0.0):
        if self.kinematic_only:
            return False
        p.performCollisionDetection(physicsClientId=self.physics_client_id)
        if any_self_collision(
            self._robot.robot_id,
            safety_margin=safety_margin,
            physics_client_id=self.physics_client_id,
        ):
            return True
        return any(
            p.getClosestPoints(
                self._robot.robot_id,
                body,
                max(0.0, safety_margin),
                physicsClientId=self.physics_client_id,
            )
            for body in (self.floor_id, self.table_id)
            if body is not None
        )

    def command(self, joint_positions, gripper_angle):
        self._robot.command_joints(joint_positions)
        self._robot.command_gripper(gripper_angle)
        self.step_simulation()

    def set_kinematic_state(self, joint_positions, gripper_angle=0.0):
        """立即设置一组运动学状态，用于 FK 计算和数据生成。"""
        self._robot.set_joint_state(joint_positions)
        self._robot.set_gripper_state(gripper_angle)
        if not self.kinematic_only:
            self.step_simulation(real_time=False)
        return self.tcp_pose()

    def evaluate_fk(self, joint_positions):
        """在不推进仿真的情况下计算一组机械臂关节角对应的位姿。"""
        current_joints, _ = self.joint_state()
        try:
            self._robot.set_joint_state(joint_positions)
            return self.tcp_pose()
        finally:
            self._robot.set_joint_state(current_joints)

    def solve_ik(self, position, orientation=None):
        return self._robot.solve_ik(position, orientation)

    def joint_state(self):
        return self._robot.joint_positions()

    def tcp_pose(self):
        return self._robot.pose()

    def block_pose(self):
        if self._object1 is None:
            raise RuntimeError("task objects are disabled in this environment")
        return self._object1.pose()

    def block_size(self):
        """返回被抓物体以米为单位的完整 xyz 尺寸。"""
        if self._object1 is None:
            raise RuntimeError("task objects are disabled in this environment")
        return self._object1.size.copy()

    def target_position(self):
        """返回物体中心期望到达的最终位置。"""
        if self._target_position is None:
            raise RuntimeError("task objects are disabled in this environment")
        return self._target_position.copy()

    def set_target_position(self, target_position):
        if self._object1 is None:
            raise RuntimeError("task objects are disabled in this environment")
        target = np.asarray(target_position, dtype=float)
        self._target_position = np.array(
            [target[0], target[1], self._object1.half_height]
        )
        marker_position, marker_orientation, _ = self._object2.pose()
        marker_position[:2] = target[:2]
        p.resetBasePositionAndOrientation(
            self._object2.id,
            marker_position.tolist(),
            marker_orientation.tolist(),
            physicsClientId=self.physics_client_id,
        )
        self._success_frames = 0

    def task_succeeded(self, required_stable_frames=100):
        if self._object1 is None:
            return False
        position, _, _ = self._object1.pose()
        linear, angular = p.getBaseVelocity(
            self._object1.id, physicsClientId=self.physics_client_id
        )
        position_ok = (
            np.linalg.norm(position[:2] - self._target_position[:2]) < 0.01
            and abs(position[2] - self._target_position[2]) < 0.01
        )
        stable = np.linalg.norm(linear) < 0.02 and np.linalg.norm(angular) < 0.2
        self._success_frames = self._success_frames + 1 if position_ok and stable else 0
        return self._success_frames >= required_stable_frames

    def _replace_line(self, key, start, end, color):
        old_id = self._debug_line_ids.get(key, -1)
        self._debug_line_ids[key] = p.addUserDebugLine(
            start,
            end,
            color,
            lineWidth=3,
            lifeTime=0,
            replaceItemUniqueId=old_id,
            physicsClientId=self.physics_client_id,
        )

    def _update_frame(self, prefix, position, quaternion):
        for axis, local, color in (
            ("x", [0.05, 0, 0], [1, 0, 0]),
            ("y", [0, 0.05, 0], [0, 1, 0]),
            ("z", [0, 0, 0.05], [0, 0, 1]),
        ):
            direction = p.multiplyTransforms(
                [0, 0, 0], quaternion, local, [0, 0, 0, 1]
            )[0]
            end = (np.asarray(position) + np.asarray(direction)).tolist()
            self._replace_line(f"{prefix}_{axis}", position, end, color)

    def _replace_text(self, key, text, position, color):
        old_id = self._debug_text_ids.get(key, -1)
        self._debug_text_ids[key] = p.addUserDebugText(
            text,
            position,
            textColorRGB=color,
            textSize=1.1,
            lifeTime=0,
            replaceItemUniqueId=old_id,
            physicsClientId=self.physics_client_id,
        )

    def _update_debug_display(self):
        position, quaternion, euler = self.tcp_pose()
        self._update_frame("tcp", position.tolist(), quaternion.tolist())
        if self._object1 is not None:
            object_position, object_quaternion, _ = self.block_pose()
            self._update_frame(
                "object", object_position.tolist(), object_quaternion.tolist()
            )
        rows = (
            ("position", f"TCP position: {np.round(position, 3)}", [1, 0, 0]),
            ("quaternion", f"TCP quaternion xyzw: {np.round(quaternion, 3)}", [0, 1, 0]),
            ("euler", f"TCP euler xyz: {np.round(euler, 3)}", [0, 0, 1]),
        )
        for row, (key, text, color) in enumerate(rows):
            self._replace_text(key, text, [-0.50, 0.30, 0.70 - 0.05 * row], color)

    def step_with_sliders(self):
        if not self.enable_debug_ui:
            raise RuntimeError("debug sliders require a GUI environment")
        mode = p.readUserDebugParameter(
            self.control_mode_slider, physicsClientId=self.physics_client_id
        )
        gripper = p.readUserDebugParameter(
            self.gripper_slider, physicsClientId=self.physics_client_id
        )
        if mode < 0.5:
            joints = [
                p.readUserDebugParameter(slider, physicsClientId=self.physics_client_id)
                for slider in self.sliders
            ]
        else:
            position = [
                p.readUserDebugParameter(slider, physicsClientId=self.physics_client_id)
                for slider in self.tcp_position_sliders
            ]
            euler = [
                p.readUserDebugParameter(slider, physicsClientId=self.physics_client_id)
                for slider in self.tcp_euler_sliders
            ]
            joints = self.solve_ik(
                position, p.getQuaternionFromEuler(euler)
            )
        self.set_kinematic_state(joints, gripper)
        self._update_debug_display()
