import yaml
import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
from pynput import keyboard
from cmd_keyboard import get_cmd

# ------------------ 工作环境初始化 ------------------ #
# 读取配置文件
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
# 自动检测并使用 GPU 加速（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从配置中提取参数
paths = cfg["paths"]                  # 模型和策略网络的文件路径
joint_names = cfg["joint_names"]      # 关节名称列表
wheel_ids = cfg["wheel_ids"]          # 轮式电机的 ID 索引（针对轮腿机器人的轮子）

# 初始化目标关节位置和下蹲准备位置，并将其转换为张量放在指定设备上
default_dof_pos = torch.tensor(cfg["default_dof_pos"] * 4, dtype=torch.float32, device=device)
crouch_dof_pos  = torch.tensor(cfg["crouch_dof_pos"] * 4, dtype=torch.float32, device=device)

# ------------------ 控制器参数 ------------------ #
# PD 控制器的比例(P)和微分(D)增益
p_gains = torch.tensor(cfg["p_gains"] * 4, dtype=torch.float32, device=device)
d_gains = torch.tensor(cfg["d_gains"] * 4, dtype=torch.float32, device=device)
actions_scale = cfg["actions_scale"]  # 动作缩放系数
vel_scale = cfg["vel_scale"]          # 速度缩放系数（主要用于轮子）
yaw_kp = cfg["yaw_kp"]                # 航向角(Yaw)的 P 控制系数

scale_factors = cfg["scale_factors"]  # 观测值缩放因子（线速度、角速度、关节位置等）

# 加载 MuJoCo 物理场景和数据
m = mujoco.MjModel.from_xml_path(paths["scene_xml"])
d = mujoco.MjData(m)

# ------------------ 辅助函数 ------------------ #

def get_sensor_data(name):
    """根据传感器名称获取传感器数据"""
    id_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if id_ == -1:
        raise ValueError(f"Sensor {name} not found") # 找不到传感器时报错
    adr, dim = m.sensor_adr[id_], m.sensor_dim[id_]
    # 返回指定维度的数据，并转换为 PyTorch 张量
    return torch.tensor(d.sensordata[adr:adr+dim], device=device, dtype=torch.float32)

def world2self(quat, v):
    """将世界坐标系下的向量转换到机器人机体坐标系（四元数旋转）"""
    q_w, q_vec = quat[0], quat[1:]
    v_vec = torch.tensor(v, device=device, dtype=torch.float32)
    # 利用四元数公式计算旋转后的向量（常用于计算投影重力）
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    return a - b + c

def get_obs(actions, default_dof_pos, commands):
    """构建 RL 策略网络所需的观测空间 (Observation)"""
    sf = scale_factors
    commands_scale = torch.tensor([sf["scale_lin_vel"], sf["scale_lin_vel"], sf["scale_ang_vel"]], device=device)
    
    # 获取机体四元数，并计算机体坐标系下的重力投影
    base_quat = get_sensor_data("imu_quat")
    projected_gravity = world2self(base_quat, torch.tensor([0., 0., -1.], device=device))
    
    # 获取角速度
    imu_gyro = get_sensor_data("imu_gyro")

    # 获取所有关节的位置 (dof pos)
    dof_pos = torch.zeros(16, device=device)
    for i, n in enumerate(joint_names):
        dof_pos[i] = get_sensor_data(n + "_pos")[0]
    # 注意：对于轮腿机器人，轮子的转角绝对位置通常没有意义，因此在观测中将其强制置零
    dof_pos[wheel_ids] = 0.0

    # 获取所有关节的速度 (dof vel)
    dof_vel = torch.zeros(16, device=device)
    for i, n in enumerate(joint_names):
        dof_vel[i] = get_sensor_data(n + "_vel")[0]

    cmds = torch.tensor(commands, device=device)
    
    # 拼接所有的观测值：角速度、投影重力、指令、关节位置误差、关节速度、上一步的动作
    return torch.cat([
        imu_gyro * sf["scale_ang_vel"],
        projected_gravity,
        cmds * commands_scale,
        (dof_pos - default_dof_pos) * sf["scale_dof_pos"],
        dof_vel * sf["scale_dof_vel"],
        actions
    ], dim=-1)

def main():
    global control_mode
    control_mode = 0 # 初始模式：0 -> 阻尼模式，1 -> PD 站立模式，2 -> RL 策略控制模式

    # 尝试加载训练好的 RL 策略网络（通常是用 PPO 等算法导出的 jit 模型）
    try:
        policy = torch.jit.load(paths["policy_path"])
        policy.eval().to(device) # 设置为评估模式并移动到计算设备
        print("Success to load policy network")
    except Exception as e:
        policy = None
        print("Fail to load policy network", e)

    # 初始化机器人姿态为下蹲准备状态
    for i, name in enumerate(joint_names):
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jnt_id]] = crouch_dof_pos[i].item()
    
    # 预热仿真，让机器人落到地面并稳定
    for _ in range(200):
        mujoco.mj_step(m, d)
        
    print("Damping mode......")
    print("1 -> PD   2 -> RL")

    # 初始化动作和观测缓冲区（用于存储历史帧，由于网络需要输入历史观测序列）
    actions = torch.zeros(16, device=device)
    obs_buffer = torch.zeros((6, 57), device=device) # 假设需要 6 帧历史，每帧 57 维

    # ------------------ 键盘监听器 ------------------ #
    def on_press(key):
        global control_mode
        try:
            # 按 '1' 切换到 PD 控制（站立）
            if key.char == '1' and control_mode == 0:
                control_mode = 1
                print(" PD mode ......")
            # 按 '2' 切换到 RL 控制（行走/运动）
            elif key.char == '2' and control_mode == 1 and policy is not None:
                control_mode = 2
                print(" RL mode ......")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # ------------------ 仿真主循环 ------------------ #
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # 获取当前所有关节的位置、速度，并计算位置误差
            dof_pos = torch.cat([get_sensor_data(n+"_pos") for n in joint_names]).to(device)
            dof_vel = torch.cat([get_sensor_data(n+"_vel") for n in joint_names]).to(device)
            dof_err = default_dof_pos - dof_pos

            # --- RL 控制模式 (Mode 2) 准备指令 ---
            if control_mode == 2:
                kb_cmd = get_cmd() # 从外部获取键盘输入指令 (vx, vy, yaw_rate)
                commands = kb_cmd
                
                # 获取四元数以计算当前的偏航角 (Yaw)
                base_quat = get_sensor_data("imu_quat")
                q_w, q_x, q_y, q_z = base_quat
                yaw_now = torch.atan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z))
                
                # 计算目标偏航角与当前偏航角的误差
                yaw_err = torch.atan2(torch.sin(commands[2] - yaw_now), torch.cos(commands[2] - yaw_now))
                # 将航向角误差通过 P 控制转换为角速度指令
                commands[2] = yaw_kp * yaw_err
                
                # 动态打印当前指令和状态
                print(f"\rRL cmd: vx={commands[0]:+4.1f}  "f"vy={commands[1]:+4.1f}  wz={commands[2]:+4.1f}  "
                f"yaw_now={yaw_now:+4.2f}", end='')
            else:
                commands = [0., 0., 0.]

            # --- PD 控制模式 (Mode 1) ---
            if control_mode == 1:
                act = torch.zeros(16, device=device)
                for i in range(16):
                    if i in wheel_ids:
                        # 轮子只做阻尼控制，不控制位置
                        act[i] = -d_gains[i]*dof_vel[i]
                    else:
                        # 腿部关节执行 PD 控制以维持默认站立姿态
                        act[i] = (1.2 * 1.25 * p_gains[i]*dof_err[i] - d_gains[i]*dof_vel[i])
                # 限制输出扭矩在 [-100, 100] 之间，并下发给电机
                d.ctrl[:] = torch.clip(act, -100, 100).cpu().numpy()

            # --- RL 策略控制执行 (Mode 2) ---
            elif control_mode == 2 and policy is not None:
                # 获取当前观测值，裁剪防止异常跳变
                obs_now = get_obs(actions, default_dof_pos, commands)
                obs_now = torch.clip(obs_now, -100, 100)
                
                # 更新观测缓冲区（移除最旧的一帧，加入最新的一帧）
                obs_buffer = torch.cat([obs_now.unsqueeze(0), obs_buffer[:-1]], dim=0)
                obs_seq = obs_buffer.flatten() # 展平为一维送入网络
                
                # 推理获取动作
                actions = policy(obs_seq)
                actions_scaled = actions * actions_scale
                
                # 将网络输出转换为目标速度（主要针对轮子）
                vel_ref = torch.zeros_like(actions_scaled)
                vel_ref[wheel_ids] = actions[wheel_ids] * vel_scale
                
                # 最终将 RL 输出的动作作为目标位置/速度，通过底层 PD 转化为扭矩
                act = p_gains * (actions_scaled + dof_err) + d_gains * (vel_ref - dof_vel)
                d.ctrl[:] = torch.clip(act, -100, 100).detach().cpu().numpy()
            
            # --- 阻尼模式 (Mode 0) ---
            else:
                d.ctrl[:] = 0.0 # 不输出力矩，靠重力和摩擦力自然下落

            # 记录当前时间，执行仿真步进
            step_start = time.time()
            for _ in range(cfg["sim_steps_per_loop"]):
                mujoco.mj_step(m, d)
            
            # 摄像机跟随机器人 (base_link)
            viewer.cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')]
            viewer.sync()
            
            # 保持控制频率，避免仿真跑得比真实时间快太多
            time_until_next_step = m.opt.timestep*4 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()