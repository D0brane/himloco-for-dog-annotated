# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymtorch, gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    # Prepare wheel monitoring: collect wheel DOF indices and names so we can
    # print their velocities in the main loop for real-time inspection.
    wheel_indices = []
    wheel_names = []
    if hasattr(env, "wheel_indices") and getattr(env, "dof_names", None) is not None:
        try:
            # wheel_indices is a tensor on device; move to CPU and convert
            wheel_indices = env.wheel_indices.to('cpu').numpy().astype(int).tolist()
            wheel_names = [env.dof_names[i] for i in wheel_indices]
        except Exception:
            wheel_indices = []
            wheel_names = []
    if wheel_names:
        print("Monitoring wheel joints:", wheel_names)

    # --- Diagnostic A: dump DOF props and state for wheel indices (once) ---
    try:
        if wheel_indices:
            env0 = env.envs[0]
            actor0 = env.actor_handles[0]
            dp = env.gym.get_actor_dof_properties(env0, actor0)
            print("=== DOF props for wheel indices ===")
            for wi, name in zip(wheel_indices, wheel_names):
                try:
                    dm = int(dp['driveMode'][wi])
                except Exception:
                    dm = None
                try:
                    effort = float(dp['effort'][wi])
                except Exception:
                    effort = None
                try:
                    stiff = float(dp['stiffness'][wi])
                    damp = float(dp['damping'][wi])
                except Exception:
                    stiff = None
                    damp = None
                try:
                    vel_lim = float(dp['velocity'][wi])
                except Exception:
                    vel_lim = None
                print(f"{name} (idx {wi}): driveMode={dm}, stiffness={stiff}, damping={damp}, effort_limit={effort}, vel_limit={vel_lim}")

            print("=== DOF state/limits for wheel indices ===")
            for wi, name in zip(wheel_indices, wheel_names):
                try:
                    pos = env.dof_pos[0, wi].item()
                    vel = env.dof_vel[0, wi].item()
                    pos_lo = env.dof_pos_limits[wi, 0].item()
                    pos_hi = env.dof_pos_limits[wi, 1].item()
                    vel_lim = env.dof_vel_limits[wi].item()
                    print(f"{name}: pos={pos:.6f} rad, vel={vel:.6f} rad/s, pos_lim=[{pos_lo:.3f},{pos_hi:.3f}], vel_lim={vel_lim:.3f}")
                except Exception as e:
                    print(f"Failed reading DOF state for {name}: {e}")
    except Exception as e:
        print("Diagnostic A failed:", e)

    # --- Diagnostic B: manual torque test on first wheel (apply 10 Nm for a few sim steps) ---
    try:
        if wheel_indices:
            test_idx = wheel_indices[0]
            test_name = wheel_names[0]
            print(f"Running manual torque test on {test_name} (idx {test_idx}) => applying 10 Nm for 10 frames")
            test_torques = torch.zeros_like(env.torques)
            test_torques[0, test_idx] = 10.0
            # send once
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(test_torques))
            for _step in range(10):
                env.gym.simulate(env.sim)
                if env.device == 'cpu':
                    env.gym.fetch_results(env.sim, True)
                env.gym.refresh_dof_state_tensor(env.sim)
                try:
                    v = env.dof_vel[0, test_idx].item()
                    print(f" manual test step {_step}: dof_vel[{test_name}] = {v:.6f} rad/s")
                except Exception as e:
                    print(" manual test read failed:", e)
    except Exception as e:
        print("Diagnostic B failed:", e)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
    
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # Real-time wheel velocity monitoring (robot_index 0)
        if wheel_indices:
            try:
                # env.dof_vel shape: (num_envs, num_dof)
                wheel_vels = env.dof_vel[robot_index, wheel_indices].cpu().numpy()
                vel_str = ", ".join([f"{n}:{v:.3f}rad/s" for n, v in zip(wheel_names, wheel_vels)])
                print(f"Step {i}: {vel_str}")

                # Diagnostic printing every 10 steps to understand why wheels don't move
                if i % 10 == 0:
                    # actions (policy output)
                    try:
                        act = actions[robot_index, :].cpu().numpy()
                    except Exception:
                        act = None
                    # actions_scaled (position/torque part) and vel_ref are computed inside the env;
                    # we attempt to fetch torques, torque_limits and dof_vel and print wheel-related slices.
                    try:
                        # torques shape: (num_envs, num_actions)
                        torques = env.torques[robot_index, :].cpu().numpy()
                    except Exception:
                        torques = None
                    try:
                        torque_limits = env.torque_limits.cpu().numpy()
                    except Exception:
                        torque_limits = None

                    print("  diag: action_sample=", None if act is None else act.tolist())
                    if torques is not None and torque_limits is not None:
                        wheel_torques = [torques[idx] for idx in wheel_indices]
                        wheel_limits = [torque_limits[idx] for idx in wheel_indices]
                        print(f"  diag: wheel_torques={['{:.3f}'.format(t) for t in wheel_torques]}")
                        print(f"  diag: wheel_limits={['{:.3f}'.format(l) for l in wheel_limits]}")
                    else:
                        print("  diag: torques/limits not available")
                    # Extra diagnostics: check for resets and recent actions/contact forces
                    try:
                        rb = int(env.reset_buf[robot_index].item()) if hasattr(env, 'reset_buf') else None
                    except Exception:
                        rb = None
                    try:
                        last_act = env.last_actions[robot_index].cpu().numpy().tolist()
                    except Exception:
                        last_act = None
                    try:
                        foot_cf = env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy().tolist()
                    except Exception:
                        foot_cf = None
                    print(f"  diag: reset_buf[{robot_index}]={rb}, last_actions={last_act}, foot_contact_forces_z={foot_cf}")
            except Exception:
                # don't crash the loop on monitoring errors
                pass

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)
