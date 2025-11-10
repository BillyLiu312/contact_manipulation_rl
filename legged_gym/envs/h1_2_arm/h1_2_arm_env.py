
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from .h1_2_arm_config import H1_2ArmRoughCfg
from legged_gym.utils.se3_math import *

class H1_2ArmRobot(LeggedRobot):
    def __init__(self, cfg: H1_2ArmRoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._create_hand_drivers()                                                          # create hand drivers after sim creation

    def _create_hand_drivers(self):
        """Create driver rigid bodies for left/right hands"""
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        self.driver_asset = self.gym.create_sphere(self.sim, 0.01, asset_options)

        self.left_driver_handles = []
        # self.right_driver_handles = []
        self.left_driver_rb_handles = []
        # self.right_driver_rb_handles = []

        for i in range(self.num_envs):
            # Left driver
            l_handle = self.gym.create_actor(
                self.envs[i], self.driver_asset, gymapi.Transform(), "left_driver", i, 0
            )
            l_rb_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], l_handle, 0)
            self.left_driver_handles.append(l_handle)
            self.left_driver_rb_handles.append(l_rb_handle)

            # Right driver
            # r_handle = self.gym.create_actor(
            #     self.envs[i], self.driver_asset, gymapi.Transform(), "right_driver", i, 0
            # )
            # r_rb_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], r_handle, 0)
            # self.right_driver_handles.append(r_handle)
            # self.right_driver_rb_handles.append(r_rb_handle)

        # Create fixed joints between ee_op and drivers
        for i in range(self.num_envs):
            # Left
            left_ee_handle = self.gym.find_actor_rigid_body_handle(
                self.envs[i], self.actor_handles[i], "left_ee_op"
            )
            self.gym.create_fixed_joint(
                self.envs[i], left_ee_handle, self.left_driver_rb_handles[i], gymapi.Transform()
            )
            # Right
            # right_ee_handle = self.gym.find_actor_rigid_body_handle(
            #     self.envs[i], self.actor_handles[i], "right_ee_op"
            # )
            # self.gym.create_fixed_joint(
            #     self.envs[i], right_ee_handle, self.right_driver_rb_handles[i], gymapi.Transform()
            # )

        # Store indices for contact force access
        self.left_driver_indices = torch.tensor(self.left_driver_rb_handles, dtype=torch.long, device=self.device)
        self.right_driver_indices = torch.tensor(self.right_driver_rb_handles, dtype=torch.long, device=self.device)

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

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

        # Buffers for twist constraint
        self.twist_left = torch.zeros(self.num_envs, 6, device=self.device)
        # self.twist_right = torch.zeros(self.num_envs, 6, device=self.device)
        self.T0_left = torch.zeros(self.num_envs, 4, 4, device=self.device)
        # self.T0_right = torch.zeros(self.num_envs, 4, 4, device=self.device)

        # Contact force buffer
        self.contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))

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
        rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        for i in env_ids:
            # Left
            pos_l = rb_states[self.left_driver_rb_handles[i], :3].clone()
            quat_l = rb_states[self.left_driver_rb_handles[i], 3:7].clone()
            self.T0_left[i] = pose_to_se3(pos_l, quat_l)
            # Right
            # pos_r = rb_states[self.right_driver_rb_handles[i], :3].clone()
            # quat_r = rb_states[self.right_driver_rb_handles[i], 3:7].clone()
            # self.T0_right[i] = pose_to_se3(pos_r, quat_r)

    def _project_hand_poses(self):
        """Project driver poses onto 1D SE(3) manifold defined by twist"""
        rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        for i in range(self.num_envs):
            # --- Left hand ---
            pos_curr = rb_states[self.left_driver_rb_handles[i], :3]
            quat_curr = rb_states[self.left_driver_rb_handles[i], 3:7]
            T_curr = pose_to_se3(pos_curr, quat_curr)
            T_rel = torch.inverse(self.T0_left[i]) @ T_curr
            delta_twist = se3_log_map(T_rel)
            s = torch.dot(delta_twist, self.twist_left[i])
            T_proj = self.T0_left[i] @ se3_exp_map(self.twist_left[i], s)
            pos_proj, quat_proj = se3_to_pose(T_proj)
            self.gym.set_rigid_transform(
                self.envs[i], self.left_driver_rb_handles[i],
                gymapi.Transform(gymapi.Vec3(*pos_proj), gymapi.Quat(*quat_proj[[3,0,1,2]]))
            )

            # --- Right hand ---
            # pos_curr = rb_states[self.right_driver_rb_handles[i], :3]
            # quat_curr = rb_states[self.right_driver_rb_handles[i], 3:7]
            # T_curr = pose_to_se3(pos_curr, quat_curr)
            # T_rel = torch.inverse(self.T0_right[i]) @ T_curr
            # delta_twist = se3_log_map(T_rel)
            # s = torch.dot(delta_twist, self.twist_right[i])
            # T_proj = self.T0_right[i] @ se3_exp_map(self.twist_right[i], s)
            # pos_proj, quat_proj = se3_to_pose(T_proj)
            # self.gym.set_rigid_transform(
            #     self.envs[i], self.right_driver_rb_handles[i],
            #     gymapi.Transform(gymapi.Vec3(*pos_proj), gymapi.Quat(*quat_proj[[3,0,1,2]]))
            # )

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
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
        left_cf = self.contact_forces[self.left_driver_indices, :3]  # (N, 3)
        right_cf = self.contact_forces[self.right_driver_indices, :3]  # (N, 3)

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
    