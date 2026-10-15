"""任务 7：在 PyBullet 中完成物块抓取与放置。"""

# --------------------- 导入常用库 ---------------------
import argparse

import numpy as np
import pybullet as p

from dofbot import (
    DofbotEnv, PHYSICAL_LOWER, PHYSICAL_UPPER, REST_POSE, wrap_to_pi,
)

# --------------------- 抓放参数 ---------------------
open_gripper = np.deg2rad(20)
closed_gripper = np.deg2rad(-20)

# TCP 目标 = 物块中心或目标中心 + 世界坐标系偏移（m）。
# 参考补偿在此 URDF 中会使夹爪只碰到物块的一侧；下列数值由接触测量校正。
offset_pre_grasp = np.array([-0.005, 0.005, 0.05])
offset_grasp = np.array([-0.005, 0.005, 0.01])
offset_pre_place = np.array([0, 0, 0.05])
offset_place = np.array([0, 0, 0])
position_tolerance = 0.015  # 单次移动的 TCP 位置容差（m）。


# --------------------- 已提供：平滑关节运动 ---------------------
# 每次 env.command() 发送五个关节角、一个夹爪指令，并推进一个仿真步。
def move_joints(env, target, gripper, steps=600, settle_steps=0):
    start, _ = env.joint_state()
    start = np.asarray(start)
    target = np.asarray(target)
    delta = target - start
    delta[[0, 4]] = wrap_to_pi(delta[[0, 4]])  # J1、J5 沿较短角度方向移动。
    for t in np.linspace(0, 1, steps):
        s = t**3 * (10 - 15 * t + 6 * t**2)  # 五次平滑插值。
        env.command(start + s * delta, gripper)
    for _ in range(settle_steps):
        env.command(target, gripper)


def move_tcp_position(env, target_position, gripper, label):
    """对 TCP 位置做 IK，检查关节限位和 FK，再运动并检查实际误差。

    五自由度 Dofbot 无法独立指定完整六维姿态，因此这里使用位置型 IK；
    它保留机械臂自然的俯视夹爪方向，避免把不可达的完整姿态交给 IK。
    """
    target_position = np.asarray(target_position, dtype=float)
    q_target = env.solve_ik(target_position)  # 不约束完整姿态
    q_check = np.asarray(q_target, dtype=float).copy()
    q_check[[0, 4]] = wrap_to_pi(q_check[[0, 4]])
    if np.any(q_check < PHYSICAL_LOWER - 1e-5) or np.any(q_check > PHYSICAL_UPPER + 1e-5):
        raise RuntimeError(f"{label}：IK 解超出物理关节限位：{np.round(q_target, 4)}")
    fk_position, _, fk_euler = env.evaluate_fk(q_check)
    ik_error = float(np.linalg.norm(fk_position - target_position))
    print(f"{label}：IK/FK 位置误差 {ik_error * 1000:.2f} mm；TCP 欧拉角 {np.round(fk_euler, 3)} rad")
    if not np.isfinite(ik_error) or ik_error > position_tolerance:
        raise RuntimeError(f"{label}：目标位置不可达，FK 复算误差 {ik_error:.4f} m")

    move_joints(env, q_check, gripper)
    reached, _, _ = env.tcp_pose()
    actual_error = float(np.linalg.norm(reached - target_position))
    print(f"{label}：实际 TCP 位置误差 {actual_error * 1000:.2f} mm")
    if not np.isfinite(actual_error) or actual_error > position_tolerance:
        raise RuntimeError(f"{label}：运动后 TCP 未到达目标位置")


def check_block_lifted(env, label):
    """物块应离开桌面；否则停止后续搬运，避免空夹爪移动。"""
    position, _, _ = env.block_pose()
    print(f"{label}：物块中心 {np.round(position, 5)} m")
    if position[2] <= 0.030:
        raise RuntimeError(f"{label}：物块未被稳定提起，请检查夹爪接触与补偿")


# --------------------- 初始化环境 ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务 7：PyBullet 物块抓取与放置")
    parser.add_argument("--headless", action="store_true", help="使用 DIRECT 模式运行并打印结果，不打开 GUI")
    args = parser.parse_args()
    env = DofbotEnv(
        connection_mode=p.DIRECT if args.headless else p.GUI,
        enable_debug_ui=not args.headless,
    )
    try:
        env.reset()

        # --------------------- 读取物块和目标位置 ---------------------
        # block_pose() 返回位置、四元数、欧拉角；位置单位 m，角度单位 rad。
        block_position, _, block_euler = env.block_pose()
        target_position = env.target_position()
        print("物块中心：", block_position)
        print("目标中心：", target_position)

        # --------------------- 计算路点和姿态 ---------------------
        pre_grasp = block_position + offset_pre_grasp
        grasp = block_position + offset_grasp
        pre_place = target_position + offset_pre_place
        place = target_position + offset_place

        # 固定 TCP 俯视抓取，抓取偏航角对齐物块，放置偏航角为 0。
        # 物理模型为五自由度，任意固定 roll/pitch/yaw 常会使 IK 超限；
        # 运动使用位置型 IK，由机械臂自然姿态保持夹爪朝下。以下目标姿态
        # 仅用于记录任务意图，不传给 IK。四元数顺序为 [x,y,z,w]。
        grasp_orientation = p.getQuaternionFromEuler([np.pi, 0.0, float(block_euler[2])])
        place_orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        print("目标 TCP 路点（m）：", {"pre_grasp": pre_grasp, "grasp": grasp,
                                      "pre_place": pre_place, "place": place})
        print("意图姿态四元数（仅记录；位置型 IK 不施加完整姿态约束）：",
              {"grasp": grasp_orientation, "place": place_orientation})

        # --------------------- Demo：移动到物块上方 ---------------------
        print("Demo：移动到物块上方")
        move_tcp_position(env, pre_grasp, open_gripper, "物块上方")

        # --------------------- 步骤 1：下降到抓取位置 ---------------------
        move_tcp_position(env, grasp, open_gripper, "下降抓取")

        # --------------------- 步骤 2：闭合夹爪 ---------------------
        print("闭合夹爪")
        joints, _ = env.joint_state()
        for _ in range(400):
            env.command(joints, closed_gripper)

        # --------------------- 步骤 3：提起物块 ---------------------
        move_tcp_position(env, pre_grasp, closed_gripper, "提起物块")
        check_block_lifted(env, "提起后")

        # --------------------- 步骤 4：移动到放置位置上方 ---------------------
        move_tcp_position(env, pre_place, closed_gripper, "移动到目标上方")
        check_block_lifted(env, "搬运后")

        # --------------------- 步骤 5：下降到放置位置 ---------------------
        move_tcp_position(env, place, closed_gripper, "下降放置")

        # --------------------- 步骤 6：释放物块 ---------------------
        print("平滑打开夹爪释放物块")
        joints, _ = env.joint_state()
        for progress in np.linspace(0.0, 1.0, 400):
            amount = progress**3 * (10 - 15 * progress + 6 * progress**2)
            env.command(joints, closed_gripper + amount * (open_gripper - closed_gripper))

        # --------------------- 步骤 7：抬起并复位 ---------------------
        move_tcp_position(env, pre_place, open_gripper, "放置后抬起")
        print("机械臂返回复位角")
        move_joints(env, REST_POSE, open_gripper)

        # --------------------- 已提供：判断稳定放置 ---------------------
        # 连续推进仿真并检查，最多等待 1500 步；不能只凭动作执行完就判成功。
        for _ in range(1500):
            joints, _ = env.joint_state()
            env.command(joints, open_gripper)
            if env.task_succeeded():
                final_position, _, _ = env.block_pose()
                error_mm = 1000 * np.linalg.norm(final_position - target_position)
                print(f"最终物块中心：{np.round(final_position, 5)} m；目标中心误差：{error_mm:.2f} mm")
                print("抓取放置成功")
                break
        else:
            raise RuntimeError("物块未在规定时间内稳定放置到目标位置")
    finally:
        env.close()

# 上方路点用于抬起与撤离，关节插值不保证 TCP 沿严格直线运动。
# 报告：描述路点、姿态和夹爪动作，分析位置误差，提供最终结果及视频。
