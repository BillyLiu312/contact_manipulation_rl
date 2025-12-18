
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from .arx_x5_config import ArxX5RoughCfg, ArxX5RoughCfgPPO
from legged_gym.utils.se3_math import *
import time
import imageio
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

class ArxX5Robot(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.gravity * noise_level
        noise_vec[3:3+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[3+self.num_actions:3+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[3+2*self.num_actions:3+3*self.num_actions] = 0. # previous actions
        noise_vec[3+3*self.num_actions:3+3*self.num_actions+3] = noise_scales.contact * noise_level * self.obs_scales.contact # contact forces on left hand

        return noise_vec
        
    def _init_buffers(self):
        super()._init_buffers()
        self.end_effector_link = self.cfg.asset.end_effector_name
        self.left_ee_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.end_effector_link)
        self.num_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.actor_handles[0])
        self.left_cf = self.contact_forces[:, self.left_ee_handle, :3]  # (N, 3)
        self.last_left_cf = torch.zeros_like(self.left_cf)

        self.twist_left = torch.zeros(self.num_envs, 6, device=self.device)
        self.T0_left = torch.zeros(self.num_envs, 4, 4, device=self.device)
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)  # (env, body, state)

        self.episode_time = torch.zeros(self.num_envs, device=self.device)
        self.force_tensor = torch.zeros(self.num_envs * self.num_bodies, 3, dtype=torch.float32, device=self.device)
        self.torque_tensor = torch.zeros(self.num_envs * self.num_bodies, 3, dtype=torch.float32, device=self.device)

        if self.viewer:
            self.experiment_name = H1_2ArmRoughCfgPPO.runner.experiment_name
            log_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name)
            runs = os.listdir(log_path)
            self.video_writer = imageio.get_writer(os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', f'{self.experiment_name}_{runs[-1]}.mp4'), fps=10)
            
    def random_unit_twist(self):
        """Generate a random unit twist in se(3) on the correct device"""
        choice = torch.randint(0, 3, (1,), device=self.device).item()
        if choice == 0:  # pure translation
            v = torch.randn(3, device=self.device)
            v /= v.norm() + 1e-8
            w = torch.zeros(3, device=self.device)
        elif choice == 1:  # pure rotation
            w = torch.randn(3, device=self.device)
            w /= w.norm() + 1e-8
            v = torch.zeros(3, device=self.device)
        else:  # spiral motion
            v = torch.randn(3, device=self.device)
            w = torch.randn(3, device=self.device)
            twist = torch.cat([v, w])
            twist /= twist.norm() + 1e-8
            v, w = twist[:3], twist[3:]
        return torch.cat([v, w])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.episode_time[env_ids] = 0.0
        # Sample random twists
        num_reset = len(env_ids)
        if num_reset > 0:
            twists = torch.stack([self.random_unit_twist() for _ in range(num_reset)])
            self.twist_left[env_ids] = twists

        # Record initial poses
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)
        pos_reset = self.rb_states[env_ids, self.left_ee_handle, :3]
        quat_reset = self.rb_states[env_ids, self.left_ee_handle, 3:7]
        self.T0_left[env_ids] = pose_to_se3_batch(pos_reset, quat_reset)  # (N_reset, 4, 4)
    
    def _apply_twist_forces(self):
        self.force_tensor.zero_()
        self.torque_tensor.zero_()

        # Current end-effector states
        curr_pos = self.rb_states[:, self.left_ee_handle, :3]      # (N, 3)
        curr_quat = self.rb_states[:, self.left_ee_handle, 3:7]    # (N, 4)

        # Current pose as SE(3)
        T_curr = pose_to_se3_batch(curr_pos, curr_quat)            # (N, 4, 4)

        # Error transform: T_err = T_target^{-1} @ T_curr
        T_target_inv = torch.inverse(self.T0_left)                     # (N, 4, 4)
        T_err = torch.bmm(T_target_inv, T_curr)                    # (N, 4, 4)

        # Log map to get 6D error twist
        err_twist = se3_log_map_batch(T_err)                       # (N, 6) = [v_err; w_err]

        # Project error onto subspace orthogonal to allowed twist
        xi = self.twist_left                                      # (N, 6), already unit
        xi_norm = torch.norm(xi, dim=1, keepdim=True)             # (N, 1)
        xi_unit = xi / (xi_norm + 1e-8)                           # (N, 6)

        # Parallel component: (err · xi) * xi
        dot_product = (err_twist * xi_unit).sum(dim=1, keepdim=True)  # (N, 1)
        parallel_component = dot_product * xi_unit                # (N, 6)
        perpendicular_err = err_twist - parallel_component        # (N, 6)

        v_perp = perpendicular_err[:, :3]   # (N, 3)
        w_perp = perpendicular_err[:, 3:]   # (N, 3)

        # Spring forces (in end-effector local frame)
        Kp_lin = 200.0
        Kp_rot = 50.0
        force_local = - Kp_lin * v_perp      # (N, 3)
        torque_local = - Kp_rot * w_perp     # (N, 3)

        # Transform to world frame
        R_ee = quat_to_rot_matrix_batch(curr_quat)  # (N, 3, 3)
        force_world = torch.bmm(R_ee, force_local.unsqueeze(-1)).squeeze(-1)    # (N, 3)
        torque_world = torch.bmm(R_ee, torque_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Write into flat force/torque tensors
        env_ids = torch.arange(self.num_envs, device=self.device)
        flat_indices = env_ids * self.num_bodies + self.left_ee_handle  # (N,)
        self.force_tensor[flat_indices, :] = force_world
        self.torque_tensor[flat_indices, :] = torque_world

        self.last_perp_error = torch.norm(perpendicular_err, dim=1)

        # Apply forces
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),
            gymtorch.unwrap_tensor(self.torque_tensor),
            gymapi.ENV_SPACE
        )

    def step(self, actions):
        """ Apply actions, simulate with projection at every sub-step """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        self.episode_time += self.cfg.control.decimation * self.cfg.sim.dt
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self._apply_twist_forces()
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
                       
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        # After all sub-steps: run standard post-processing
        self.post_physics_step()

        # Clip and return
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _draw_debug_vis(self):
        """Draw twist direction as arrows at the left end-effector (GPU-optimized)."""
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)

        # 只处理前 N 个环境（避免杂乱）
        num_envs_to_draw = self.num_envs
        if num_envs_to_draw == 0:
            return

        # 所有计算在 GPU 上进行
        curr_pos = self.rb_states[:num_envs_to_draw, self.left_ee_handle, :3]  # (N, 3)
        v = self.twist_left[:num_envs_to_draw, :3]   # (N, 3)
        w = self.twist_left[:num_envs_to_draw, 3:]   # (N, 3)

        scale_v = 0.3
        scale_w = 0.3

        # GPU 上计算箭头终点
        end_v = curr_pos + v * scale_v  # (N, 3)
        end_w = curr_pos + w * scale_w  # (N, 3)

        # 一次性拷贝到 CPU（只拷贝需要绘制的部分）
        pos_np = curr_pos.cpu().numpy()
        end_v_np = end_v.cpu().numpy()
        end_w_np = end_w.cpu().numpy()

        color_v = [0.0, 1.0, 0.0]  # green
        color_w = [1.0, 0.0, 0.0]  # red

        for i in range(num_envs_to_draw):
            start = pos_np[i]
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                1,
                start.tolist() + end_v_np[i].tolist(),
                color_v
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                1,
                start.tolist() + end_w_np[i].tolist(),
                color_w
            )

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        super().post_physics_step()
        self.last_left_cf[:] = self.left_cf[:]
        if self.cfg.env.debug_vis and self.viewer:
            self._draw_debug_vis()
            img_path = f"temp_frame.png"
            self.gym.write_viewer_image_to_file(self.viewer, img_path)
            frame = imageio.imread(img_path)
            self.video_writer.append_data(frame)
            os.remove(img_path)

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  
                                    self.projected_gravity, # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 7
                                    self.dof_vel * self.obs_scales.dof_vel, # 7
                                    self.actions, # 7
                                    self.left_cf # 3
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.left_cf
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    #------------ reward functions----------------
    def _reward_contact_force_mag(self):
        # 鼓励非零接触力（但不过大）
        force_mag = torch.norm(self.left_cf, dim=1)  # (N,)
        # 使用 softplus 或 clamp 避免爆炸
        return torch.tanh(force_mag / 20.0)  # 20N 为典型阈值
    
    def _reward_contact_force_variation(self):
        # 鼓励接触力变化
        dF = torch.norm(self.left_cf - self.last_left_cf, dim=1)
        reward = torch.tanh(dF / 10.0)  # 10 N/s 变化率
        return reward

    def _reward_no_excessive_force(self):
        force_mag = torch.norm(self.left_cf, dim=1)
        # 超过 40N 开始惩罚
        excess = torch.clamp(force_mag - 40.0, min=0.0)
        return -excess * 0.1

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_task_movement(self):
        """
        Reward velocity projected onto the allowed twist direction.
        Encourages the robot to explore the 'free' DOF.
        """
        # Get end-effector linear and angular velocity
        # self.rb_states: (N, num_bodies, 13) -> 13: [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        lin_vel = self.rb_states[:, self.left_ee_handle, 7:10]
        ang_vel = self.rb_states[:, self.left_ee_handle, 10:13]
        
        # Current twist in world frame (N, 6)
        current_twist = torch.cat([lin_vel, ang_vel], dim=1)
        
        # Target unit twist (N, 6)
        xi = self.twist_left
        
        # Project current velocity onto allowed axis
        # Dot product: (N, 6) * (N, 6) -> sum -> (N,)
        parallel_vel = (current_twist * xi).sum(dim=1)
        
        # Reward absolute speed along the axis (move back or forth)
        return torch.abs(parallel_vel)

    def _reward_task_compliance(self):
        """
        Penalize generating forces against the constraint.
        If the robot moves perfectly, force is 0.
        """
        force_mag = torch.norm(self.left_cf, dim=1)
        # Use tanh to cap the penalty
        return torch.tanh(force_mag / 20.0)

    def _reward_constraint_deviation(self):
        """
        Penalize the geometric error (how far we are from the allowed twist).
        This is cleaner than penalizing contact forces.
        """
        # Returns 1.0 when error is 0, drops to 0.0 as error increases.
        return torch.exp(-self.last_perp_error / 0.1)