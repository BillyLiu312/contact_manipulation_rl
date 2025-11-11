
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from .h1_2_arm_config import H1_2ArmRoughCfg
from legged_gym.utils.se3_math import *

class H1_2ArmRobot(LeggedRobot):

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
        self.left_ee_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "left_ee_op")
        self.num_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.actor_handles[0])
        self.left_cf = self.contact_forces[:, self.left_ee_handle, :3]  # (N, 3)
        self.last_left_cf = torch.zeros_like(self.left_cf)

        self.twist_left = torch.zeros(self.num_envs, 6, device=self.device)
        self.T0_left = torch.zeros(self.num_envs, 4, 4, device=self.device)
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)  # (env, body, state)

        self.episode_time = torch.zeros(self.num_envs, device=self.device)
        self.force_tensor = torch.zeros(self.num_envs * self.num_bodies, 3, dtype=torch.float32, device=self.device)
        self.torque_tensor = torch.zeros(self.num_envs * self.num_bodies, 3, dtype=torch.float32, device=self.device)

    def random_unit_twist(self):
        """Generate a random unit twist in se(3)"""
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:  # pure translation
            v = torch.randn(3)
            v /= v.norm() + 1e-8
            w = torch.zeros(3)
        elif choice == 1:  # pure rotation
            w = torch.randn(3)
            w /= w.norm() + 1e-8
            v = torch.zeros(3)
        else:  # spiral motion
            v = torch.randn(3)
            w = torch.randn(3)
            twist = torch.cat([v, w])
            twist /= twist.norm() + 1e-8
            v, w = twist[:3], twist[3:]
        return torch.cat([v, w])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self.episode_time[env_ids] = 0.0
        # Sample random twists
        for i in env_ids:
            self.twist_left[i] = self.random_unit_twist()
            # self.twist_right[i] = self.random_unit_twist()

        # Record initial poses
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)
        for i in env_ids:
            pos = self.rb_states[i, self.left_ee_handle, :3]
            quat = self.rb_states[i, self.left_ee_handle, 3:7]
            self.T0_left[i] = pose_to_se3(pos, quat)
    
    def _apply_twist_forces(self):
        self.force_tensor.zero_()
        self.torque_tensor.zero_()

        speed_scale = 0.5

        curr_pos_all = self.rb_states[:, self.left_ee_handle, :3]
        curr_quat_all = self.rb_states[:, self.left_ee_handle, 3:7]
        lin_vel_all = self.rb_states[:, self.left_ee_handle, 7:10]
        ang_vel_all = self.rb_states[:, self.left_ee_handle, 10:13]

        for i in range(self.num_envs):
            s = self.episode_time[i] * speed_scale
            T_offset = se3_exp_map(self.twist_left[i], s)
            T_target = self.T0_left[i] @ T_offset
            target_pos, target_quat = se3_to_pose(T_target)

            pos_err = target_pos - curr_pos_all[i]
            quat_err = quat_mul(target_quat, quat_conjugate(curr_quat_all[i]))
            ang_err = 2.0 * quat_err[:3]

            force = 150.0 * pos_err - 15.0 * lin_vel_all[i]
            torque = 50.0 * ang_err - 5.0 * ang_vel_all[i]

            flat_idx = i * self.num_bodies + self.left_ee_handle

            self.force_tensor[flat_idx, :] = force
            self.torque_tensor[flat_idx, :] = torque

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.force_tensor),
            gymtorch.unwrap_tensor(self.torque_tensor),
            gymapi.ENV_SPACE  # or WORLD_SPACE
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

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        super().post_physics_step()
        self.last_left_cf[:] = self.left_cf[:]

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
    
    
    