from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.commands import (
    MotionCommand,
    MotionCommandCfg,
)


# =============================================================================
# User CSV default pose
# trajectory (7).csv, row 0
# =============================================================================

CSV_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

CSV_JOINT_POS = [
    -0.645241,
     0.151306,
     0.001266,
     0.938340,
    -0.269836,
    -0.097535,
    -0.414963,
    -0.253771,
    -0.208547,
     1.025081,
    -0.572281,
     0.075960,
     0.000000,
     0.000000,
     0.000000,
    -0.728043,
     0.108223,
    -0.009257,
     0.087854,
    -0.125702,
    -0.066006,
    -0.024761,
    -0.755417,
    -0.223752,
     0.240973,
    -0.088188,
     0.125702,
    -0.066006,
     0.030802,
]

# CSV:
# root position = [0, 0, 0.7098747459723271]
# quaternion xyzw = [0, 0, 0, 1]
# IsaacLab uses wxyz.
CSV_ROOT_POS = [0.0, 0.0, 0.7098747459723271]
CSV_ROOT_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]


# =============================================================================
# Motion command:
# V4 final-frame hold + CSV fixed reset
# =============================================================================


class G9CsvV4MotionCommand(MotionCommand):
    """V4 motion behavior with the user's CSV pose as the fixed reset state.

    Behavior:
        reset
          -> exact CSV pose, zero physical velocities
          -> NPZ reference starts from frame 0
          -> play complete G9
          -> hold final NPZ frame for hold_last_s
             (desired velocities are zero)
          -> reset back to exact CSV pose

    Adaptive mid-motion reset is intentionally disabled.
    """

    cfg: "G9CsvV4MotionCommandCfg"

    def __init__(self, cfg: "G9CsvV4MotionCommandCfg", env):
        super().__init__(cfg, env)

        control_dt = float(
            env.cfg.sim.dt * env.cfg.decimation
        )

        self.hold_max_steps = max(
            int(round(float(cfg.hold_last_s) / control_dt)),
            1,
        )

        self.hold_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # Recovery-push state used by g9_recovery_push_force.
        self.g9_recovery_push_decided = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.g9_recovery_push_remaining = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.g9_recovery_push_force_w = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float32,
            device=self.device,
        )

        # Resolve CSV values to IsaacLab joints BY NAME.
        # This avoids relying on any assumed internal joint ordering.
        csv_joint_ids = self.robot.find_joints(
            CSV_JOINT_NAMES,
            preserve_order=True,
        )[0]

        if len(csv_joint_ids) != len(CSV_JOINT_NAMES):
            raise RuntimeError(
                "Could not resolve all CSV G1 joints: "
                f"resolved={len(csv_joint_ids)}, "
                f"expected={len(CSV_JOINT_NAMES)}"
            )

        self.csv_joint_ids = torch.as_tensor(
            csv_joint_ids,
            dtype=torch.long,
            device=self.device,
        )

        self.csv_joint_pos = torch.tensor(
            CSV_JOINT_POS,
            dtype=torch.float32,
            device=self.device,
        )

        self.csv_root_pos = torch.tensor(
            CSV_ROOT_POS,
            dtype=torch.float32,
            device=self.device,
        )

        self.csv_root_quat = torch.tensor(
            CSV_ROOT_QUAT_WXYZ,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_holding(self) -> torch.Tensor:
        return self.hold_steps > 0

    @property
    def hold_progress(self) -> torch.Tensor:
        return torch.clamp(
            self.hold_steps.float()
            / float(max(self.hold_max_steps, 1)),
            0.0,
            1.0,
        )

    # -------------------------------------------------------------------------
    # V4 final hold:
    # pose remains on last NPZ frame, desired velocities become zero.
    # -------------------------------------------------------------------------

    @property
    def joint_vel(self) -> torch.Tensor:
        value = self.motion.joint_vel[
            self.time_steps
        ].clone()
        value[self.is_holding] = 0.0
        return value

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        value = self.motion.body_lin_vel_w[
            self.time_steps
        ].clone()
        value[self.is_holding] = 0.0
        return value

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        value = self.motion.body_ang_vel_w[
            self.time_steps
        ].clone()
        value[self.is_holding] = 0.0
        return value

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        value = self.motion.body_lin_vel_w[
            self.time_steps,
            self.motion_anchor_body_index,
        ].clone()
        value[self.is_holding] = 0.0
        return value

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        value = self.motion.body_ang_vel_w[
            self.time_steps,
            self.motion_anchor_body_index,
        ].clone()
        value[self.is_holding] = 0.0
        return value

    def _reset_cycle_state(self, env_ids):
        if env_ids.numel() == 0:
            return

        self.hold_steps[env_ids] = 0
        self.g9_recovery_push_decided[env_ids] = False
        self.g9_recovery_push_remaining[env_ids] = 0
        self.g9_recovery_push_force_w[env_ids] = 0.0

    def _resample_command(self, env_ids):
        """Reset physical robot to CSV pose and reference to NPZ frame 0."""

        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )

        # Reference always starts at frame 0.
        self.time_steps[env_ids] = 0
        self._reset_cycle_state(env_ids)

        # -------------------------------------------------------------
        # Exact CSV joint pose.
        # Start with defaults for safety, then overwrite named joints.
        # -------------------------------------------------------------
        joint_pos = self.robot.data.default_joint_pos[
            env_ids
        ].clone()

        joint_vel = torch.zeros_like(
            joint_pos
        )

        joint_pos[
            :,
            self.csv_joint_ids,
        ] = self.csv_joint_pos.unsqueeze(0)

        soft_limits = self.robot.data.soft_joint_pos_limits[
            env_ids
        ]

        joint_pos = torch.clamp(
            joint_pos,
            min=soft_limits[:, :, 0],
            max=soft_limits[:, :, 1],
        )

        # -------------------------------------------------------------
        # Exact CSV root pose, with environment origin offset.
        # -------------------------------------------------------------
        root_pos = self.csv_root_pos.unsqueeze(0).repeat(
            len(env_ids),
            1,
        )

        root_pos = (
            root_pos
            + self._env.scene.env_origins[env_ids]
        )

        root_quat = self.csv_root_quat.unsqueeze(0).repeat(
            len(env_ids),
            1,
        )

        root_lin_vel = torch.zeros(
            (len(env_ids), 3),
            dtype=torch.float32,
            device=self.device,
        )

        root_ang_vel = torch.zeros_like(
            root_lin_vel
        )

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

        root_state = torch.cat(
            [
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
            ],
            dim=-1,
        )

        self.robot.write_root_state_to_sim(
            root_state,
            env_ids=env_ids,
        )

        # We deliberately do NOT call _adaptive_sampling().
        # Keep sampling metrics neutral.
        self._current_bin_failed.zero_()
        self.metrics["sampling_entropy"][:] = 0.0
        self.metrics["sampling_top1_prob"][:] = 0.0
        self.metrics["sampling_top1_bin"][:] = 0.0

    def _update_command(self):
        """Advance G9, hold final frame, then reset to CSV stance."""

        last_frame = self.motion.time_step_total - 1

        moving = self.time_steps < last_frame
        self.time_steps[moving] += 1

        holding = self.time_steps >= last_frame
        self.time_steps[holding] = last_frame
        self.hold_steps[holding] += 1

        reset_env_ids = torch.where(
            holding
            & (self.hold_steps > self.hold_max_steps)
        )[0]

        self._resample_command(
            reset_env_ids
        )

        # Preserve original BeyondMimic relative reference update.
        anchor_pos_w_repeat = self.anchor_pos_w[
            :, None, :
        ].repeat(
            1,
            len(self.cfg.body_names),
            1,
        )

        anchor_quat_w_repeat = self.anchor_quat_w[
            :, None, :
        ].repeat(
            1,
            len(self.cfg.body_names),
            1,
        )

        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[
            :, None, :
        ].repeat(
            1,
            len(self.cfg.body_names),
            1,
        )

        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[
            :, None, :
        ].repeat(
            1,
            len(self.cfg.body_names),
            1,
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]

        delta_ori_w = yaw_quat(
            quat_mul(
                robot_anchor_quat_w_repeat,
                quat_inv(anchor_quat_w_repeat),
            )
        )

        self.body_quat_relative_w = quat_mul(
            delta_ori_w,
            self.body_quat_w,
        )

        self.body_pos_relative_w = (
            delta_pos_w
            + quat_apply(
                delta_ori_w,
                self.body_pos_w - anchor_pos_w_repeat,
            )
        )

        # Adaptive phase sampling is disabled, but keep original buffers valid.
        self.bin_failed_count = (
            self.cfg.adaptive_alpha
            * self._current_bin_failed
            + (1.0 - self.cfg.adaptive_alpha)
            * self.bin_failed_count
        )

        self._current_bin_failed.zero_()


@configclass
class G9CsvV4MotionCommandCfg(MotionCommandCfg):
    class_type: type = G9CsvV4MotionCommand

    # Same as V4.
    hold_last_s: float = 2.0


# =============================================================================
# Helper
# =============================================================================


def _motion_phase(command: MotionCommand) -> torch.Tensor:
    total_steps = max(
        int(command.motion.time_step_total) - 1,
        1,
    )

    return (
        command.time_steps.float()
        / float(total_steps)
    )


# =============================================================================
# V4 Reward 1: support-foot slip
# =============================================================================


def g9_support_slip_l2(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.30,
    end_phase: float = 0.60,
    contact_threshold: float = 15.0,
    free_slip_speed: float = 0.10,
) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(
        command_name
    )
    robot: Articulation = env.scene[
        asset_cfg.name
    ]
    sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]

    phase = _motion_phase(command)

    phase_gate = (
        (phase >= start_phase)
        & (phase <= end_phase)
    ).float()

    contact_force = sensor.data.net_forces_w[
        :,
        sensor_cfg.body_ids,
    ]

    contact_force_norm = torch.linalg.vector_norm(
        contact_force,
        dim=-1,
    )

    left_contact = (
        contact_force_norm[..., 0]
        > contact_threshold
    ).float()

    foot_vel_xy = robot.data.body_lin_vel_w[
        :,
        asset_cfg.body_ids,
        :2,
    ]

    left_vel_xy = foot_vel_xy[..., 0, :]

    slip_speed = torch.linalg.vector_norm(
        left_vel_xy,
        dim=-1,
    )

    excess_slip = torch.relu(
        slip_speed - free_slip_speed
    )

    return (
        phase_gate
        * left_contact
        * torch.square(excess_slip)
    )


# =============================================================================
# V4 Reward 2: mild core tracking
# =============================================================================


def g9_core_tracking_exp(
    env,
    command_name: str,
    start_phase: float = 0.32,
    end_phase: float = 0.64,
    ori_std: float = 0.35,
    ang_vel_std: float = 2.5,
) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(
        command_name
    )

    phase = _motion_phase(command)

    phase_gate = (
        (phase >= start_phase)
        & (phase <= end_phase)
    ).float()

    ori_error = quat_error_magnitude(
        command.anchor_quat_w,
        command.robot_anchor_quat_w,
    )

    ori_reward = torch.exp(
        -torch.square(ori_error)
        / (ori_std**2)
    )

    ang_vel_error = torch.linalg.vector_norm(
        command.anchor_ang_vel_w
        - command.robot_anchor_ang_vel_w,
        dim=-1,
    )

    ang_vel_reward = torch.exp(
        -torch.square(ang_vel_error)
        / (ang_vel_std**2)
    )

    return phase_gate * (
        0.70 * ori_reward
        + 0.30 * ang_vel_reward
    )


# =============================================================================
# V4 Reward 3: recovery response
# =============================================================================


def g9_recovery_response_exp(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    start_phase: float = 0.64,
    contact_threshold: float = 12.0,
    ori_std: float = 0.40,
    ang_vel_std: float = 2.0,
    lin_vel_std: float = 0.35,
) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(
        command_name
    )

    robot: Articulation = env.scene[
        asset_cfg.name
    ]

    sensor: ContactSensor = env.scene.sensors[
        sensor_cfg.name
    ]

    phase = _motion_phase(command)
    phase_gate = (
        phase >= start_phase
    ).float()

    ori_error = quat_error_magnitude(
        command.anchor_quat_w,
        command.robot_anchor_quat_w,
    )

    ori_reward = torch.exp(
        -torch.square(ori_error)
        / (ori_std**2)
    )

    torso_ang_speed = torch.linalg.vector_norm(
        command.robot_anchor_ang_vel_w,
        dim=-1,
    )

    ang_reward = torch.exp(
        -torch.square(torso_ang_speed)
        / (ang_vel_std**2)
    )

    root_lin_xy = robot.data.root_lin_vel_b[
        :,
        :2,
    ]

    lin_speed = torch.linalg.vector_norm(
        root_lin_xy,
        dim=-1,
    )

    lin_reward = torch.exp(
        -torch.square(lin_speed)
        / (lin_vel_std**2)
    )

    contact_force = sensor.data.net_forces_w[
        :,
        sensor_cfg.body_ids,
    ]

    contact_force_norm = torch.linalg.vector_norm(
        contact_force,
        dim=-1,
    )

    contacts = (
        contact_force_norm
        > contact_threshold
    ).float()

    contact_reward = contacts.mean(
        dim=-1
    )

    return phase_gate * (
        0.35 * ori_reward
        + 0.25 * ang_reward
        + 0.20 * lin_reward
        + 0.20 * contact_reward
    )


# =============================================================================
# V4 Event 1: kick target reaction
# =============================================================================


def g9_kick_reaction_force(
    env,
    env_ids,
    command_name: str,
    impact_phase: float,
    phase_tolerance: float,
    force_gain: float,
    force_min: float,
    force_max: float,
    min_foot_speed: float,
    asset_cfg: SceneEntityCfg,
):

    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    command: MotionCommand = env.command_manager.get_term(
        command_name
    )

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        env_ids = torch.as_tensor(
            env_ids,
            device=asset.device,
            dtype=torch.long,
        )

    if env_ids.numel() == 0:
        return

    body_ids = asset_cfg.body_ids

    if (
        isinstance(body_ids, slice)
        or len(body_ids) != 1
    ):
        raise ValueError(
            "g9_kick_reaction_force requires "
            "right_ankle_roll_link only."
        )

    foot_body_id = body_ids[0]

    total_steps = max(
        int(command.motion.time_step_total) - 1,
        1,
    )

    phase = (
        command.time_steps[env_ids].float()
        / float(total_steps)
    )

    in_impact = (
        torch.abs(
            phase - impact_phase
        )
        <= phase_tolerance
    )

    foot_vel_w = asset.data.body_lin_vel_w[
        env_ids,
        foot_body_id,
        :,
    ]

    foot_vel_xy = foot_vel_w[:, :2]

    speed = torch.linalg.vector_norm(
        foot_vel_xy,
        dim=-1,
    )

    active = (
        in_impact
        & (speed >= min_foot_speed)
    )

    safe_speed = torch.clamp(
        speed,
        min=1.0e-6,
    )

    direction_xy = (
        foot_vel_xy
        / safe_speed.unsqueeze(-1)
    )

    force_mag = torch.clamp(
        force_gain * speed,
        min=force_min,
        max=force_max,
    )

    force_mag = (
        force_mag
        * active.float()
    )

    forces_w = torch.zeros(
        (env_ids.numel(), 1, 3),
        device=asset.device,
        dtype=torch.float32,
    )

    forces_w[:, 0, :2] = (
        -direction_xy
        * force_mag.unsqueeze(-1)
    )

    torques_w = torch.zeros_like(
        forces_w
    )

    asset.set_external_force_and_torque(
        forces=forces_w,
        torques=torques_w,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )


# =============================================================================
# V4 Event 2: recovery torso push
# =============================================================================


def g9_recovery_push_force(
    env,
    env_ids,
    command_name: str,
    probability: float,
    start_hold_s: float,
    duration_s: float,
    force_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):

    asset: Articulation = env.scene[
        asset_cfg.name
    ]

    command = env.command_manager.get_term(
        command_name
    )

    required_attrs = (
        "hold_steps",
        "is_holding",
        "g9_recovery_push_decided",
        "g9_recovery_push_remaining",
        "g9_recovery_push_force_w",
    )

    if not all(
        hasattr(command, name)
        for name in required_attrs
    ):
        raise RuntimeError(
            "g9_recovery_push_force requires "
            "G9CsvV4MotionCommand."
        )

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        env_ids = torch.as_tensor(
            env_ids,
            device=asset.device,
            dtype=torch.long,
        )

    if env_ids.numel() == 0:
        return

    body_ids = asset_cfg.body_ids

    if (
        isinstance(body_ids, slice)
        or len(body_ids) != 1
    ):
        raise ValueError(
            "g9_recovery_push_force requires "
            "torso_link only."
        )

    control_dt = float(
        env.cfg.sim.dt
        * env.cfg.decimation
    )

    start_step = max(
        int(round(
            start_hold_s / control_dt
        )),
        1,
    )

    duration_steps = max(
        int(round(
            duration_s / control_dt
        )),
        1,
    )

    candidate_mask = (
        command.is_holding[env_ids]
        & (
            command.hold_steps[env_ids]
            >= start_step
        )
        & (
            ~command.g9_recovery_push_decided[
                env_ids
            ]
        )
    )

    candidate_env_ids = env_ids[
        candidate_mask
    ]

    if candidate_env_ids.numel() > 0:
        command.g9_recovery_push_decided[
            candidate_env_ids
        ] = True

        selected = (
            torch.rand(
                candidate_env_ids.numel(),
                device=asset.device,
            )
            < probability
        )

        selected_env_ids = candidate_env_ids[
            selected
        ]

        if selected_env_ids.numel() > 0:
            angle = (
                2.0
                * torch.pi
                * torch.rand(
                    selected_env_ids.numel(),
                    device=asset.device,
                )
            )

            magnitude = (
                force_range[0]
                + (
                    force_range[1]
                    - force_range[0]
                )
                * torch.rand(
                    selected_env_ids.numel(),
                    device=asset.device,
                )
            )

            direction_xy = torch.stack(
                (
                    torch.cos(angle),
                    torch.sin(angle),
                ),
                dim=-1,
            )

            force_xy = (
                direction_xy
                * magnitude.unsqueeze(-1)
            )

            command.g9_recovery_push_force_w[
                selected_env_ids,
                :2,
            ] = force_xy

            command.g9_recovery_push_force_w[
                selected_env_ids,
                2,
            ] = 0.0

            command.g9_recovery_push_remaining[
                selected_env_ids
            ] = duration_steps

    active_mask = (
        command.g9_recovery_push_remaining[
            env_ids
        ]
        > 0
    )

    forces_w = torch.zeros(
        (env_ids.numel(), 1, 3),
        device=asset.device,
        dtype=torch.float32,
    )

    if torch.any(active_mask):
        active_local_ids = torch.where(
            active_mask
        )[0]

        active_env_ids = env_ids[
            active_local_ids
        ]

        forces_w[
            active_local_ids,
            0,
            :,
        ] = command.g9_recovery_push_force_w[
            active_env_ids
        ]

        command.g9_recovery_push_remaining[
            active_env_ids
        ] -= 1

    torques_w = torch.zeros_like(
        forces_w
    )

    asset.set_external_force_and_torque(
        forces=forces_w,
        torques=torques_w,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )
