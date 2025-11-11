
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
        self.left_driver_rb_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "left_ee_op")
        self.num_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.actor_handles[0])
        # Buffers for twist constraint
        self.twist_left = torch.zeros(self.num_envs, 6, device=self.device)
        # self.twist_right = torch.zeros(self.num_envs, 6, device=self.device)
        self.T0_left = torch.zeros(self.num_envs, 4, 4, device=self.device)
        # self.T0_right = torch.zeros(self.num_envs, 4, 4, device=self.device)

        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)  # (env, body, state)

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

        # Sample random twists
        for i in env_ids:
            self.twist_left[i] = self.random_unit_twist()
            # self.twist_right[i] = self.random_unit_twist()

        # Record initial poses
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)
        for i in env_ids:
            # Left
            pos_l = self.rb_states[i, self.left_driver_rb_index, :3]
            quat_l = self.rb_states[i, self.left_driver_rb_index, 3:7]
            self.T0_left[i] = pose_to_se3(pos_l, quat_l)
            # Right
            # pos_r = self.rb_states[self.right_driver_rb_handles[i], :3].clone()
            # quat_r = self.rb_states[self.right_driver_rb_handles[i], 3:7].clone()
            # self.T0_right[i] = pose_to_se3(pos_r, quat_r)

    def _project_hand_poses(self):
        """Project driver poses onto 1D SE(3) manifold defined by twist"""
        for i in range(self.num_envs):
            # --- Left hand ---
            pos_curr = self.rb_states[i, self.left_driver_rb_index, :3]
            quat_curr = self.rb_states[i, self.left_driver_rb_index, 3:7]
            T_curr = pose_to_se3(pos_curr, quat_curr)
            T_rel = torch.inverse(self.T0_left[i]) @ T_curr
            delta_twist = se3_log_map(T_rel)
            s = torch.dot(delta_twist, self.twist_left[i])
            T_proj = self.T0_left[i] @ se3_exp_map(self.twist_left[i], s)
            pos_proj, quat_proj = se3_to_pose(T_proj)
            
            self.gym.set_rigid_transform(
                self.envs[i],
                self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "left_ee_op"
                ),
                gymapi.Transform(
                    gymapi.Vec3(*pos_proj),
                    gymapi.Quat(*quat_proj[[3, 0, 1, 2]])  # wxyz → Quat(w,x,y,z)
                )
            )
    
    def step(self, actions):
        """ Apply actions, simulate with projection at every sub-step """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)

            # === CRITICAL: Project hand drivers AFTER each physics sub-step ===
            self._project_hand_poses()
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
                       
            self.gym.refresh_dof_state_tensor(self.sim)         # Refresh necessary tensors for next sub-step (e.g., for PD control)
            self.gym.refresh_rigid_body_state_tensor(self.sim)  # ← needed for _project_hand_poses

        # After all sub-steps: run standard post-processing
        self.post_physics_step()

        # Clip and return
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        """ Computes observations
        """
        # Get contact forces on drivers
        left_cf = self.contact_forces[:, self.left_driver_rb_index, :3]  # (N, 3)

        self.obs_buf = torch.cat((  
                                    self.projected_gravity, # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 7
                                    self.dof_vel * self.obs_scales.dof_vel, # 7
                                    self.actions, # 7
                                    left_cf # 3
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    left_cf
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    #------------ reward functions----------------
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,2,6,8]]), dim=1)
    