from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import (
    G1_ACTION_SCALE,
    G1_CYLINDER_CFG,
)
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
    LOW_FREQ_SCALE,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    CommandsCfg,
    EventCfg,
    RewardsCfg,
    TrackingEnvCfg,
)

import whole_body_tracking.tasks.tracking.mdp as mdp


# =============================================================================
# Standard G1 flat configs
# =============================================================================


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.actions.joint_pos.scale = G1_ACTION_SCALE

        self.commands.motion.anchor_body_name = "torso_link"

        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(
    G1FlatEnvCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


# =============================================================================
# FINAL command = V4 behavior + CSV fixed reset
# =============================================================================


@configclass
class G9CsvV4CommandsCfg(CommandsCfg):
    motion = mdp.G9CsvV4MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,

        # Custom reset ignores random phase/pose perturbation.
        # Keep these explicitly zero for clarity.
        pose_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },

        velocity_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },

        joint_position_range=(0.0, 0.0),

        # Same final-frame recovery hold as V4.
        hold_last_s=2.0,
    )


# =============================================================================
# V4 rewards -- unchanged
# =============================================================================


@configclass
class G9StableRewardsCfg(RewardsCfg):

    g9_support_slip = RewTerm(
        func=mdp.g9_support_slip_l2,
        weight=-0.10,
        params={
            "command_name": "motion",

            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                ],
            ),

            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_ankle_roll_link",
                ],
            ),

            "start_phase": 0.30,
            "end_phase": 0.60,
            "contact_threshold": 15.0,
            "free_slip_speed": 0.10,
        },
    )

    g9_core_tracking = RewTerm(
        func=mdp.g9_core_tracking_exp,
        weight=0.40,
        params={
            "command_name": "motion",
            "start_phase": 0.32,
            "end_phase": 0.64,
            "ori_std": 0.35,
            "ang_vel_std": 2.5,
        },
    )

    g9_recovery_response = RewTerm(
        func=mdp.g9_recovery_response_exp,
        weight=0.85,
        params={
            "command_name": "motion",

            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),

            "asset_cfg": SceneEntityCfg(
                "robot"
            ),

            "start_phase": 0.64,
            "contact_threshold": 12.0,
            "ori_std": 0.40,
            "ang_vel_std": 2.0,
            "lin_vel_std": 0.35,
        },
    )


# =============================================================================
# V4 events -- unchanged
# =============================================================================


@configclass
class G9StableEventCfg(EventCfg):

    # Disable original random root-velocity teleport push.
    push_robot = None

    # Kick target reaction: V4 15~60 N.
    kick_reaction = EventTerm(
        func=mdp.g9_kick_reaction_force,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "command_name": "motion",
            "impact_phase": 0.495,
            "phase_tolerance": 0.006,
            "force_gain": 7.5,
            "force_min": 15.0,
            "force_max": 60.0,
            "min_foot_speed": 2.0,

            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "right_ankle_roll_link",
                ],
            ),
        },
    )

    # Recovery training push: V4 35~60 N.
    recovery_push = EventTerm(
        func=mdp.g9_recovery_push_force,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "command_name": "motion",
            "probability": 0.60,
            "start_hold_s": 0.35,
            "duration_s": 0.08,
            "force_range": (35.0, 60.0),

            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "torso_link",
                ],
            ),
        },
    )


# =============================================================================
# FINAL G9 environment
# =============================================================================


@configclass
class G1FlatWoStateEstimationG9StableEnvCfg(
    G1FlatWoStateEstimationEnvCfg
):
    commands: G9CsvV4CommandsCfg = G9CsvV4CommandsCfg()
    rewards: G9StableRewardsCfg = G9StableRewardsCfg()
    events: G9StableEventCfg = G9StableEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # V4 episode / recovery hold
        # ---------------------------------------------------------------------

        self.episode_length_s = 14.0
        self.commands.motion.hold_last_s = 2.0

        # ---------------------------------------------------------------------
        # V4 existing reward tuning
        # ---------------------------------------------------------------------

        self.rewards.action_rate_l2.weight = -0.05

        # ---------------------------------------------------------------------
        # V4 physics/domain randomization
        # ---------------------------------------------------------------------

        self.events.physics_material.params[
            "static_friction_range"
        ] = (0.75, 1.15)

        self.events.physics_material.params[
            "dynamic_friction_range"
        ] = (0.65, 1.00)

        self.events.physics_material.params[
            "restitution_range"
        ] = (0.00, 0.08)

        self.events.base_com.params[
            "com_range"
        ] = {
            "x": (-0.012, 0.012),
            "y": (-0.015, 0.015),
            "z": (-0.012, 0.012),
        }

        self.events.add_joint_default_pos.params[
            "pos_distribution_params"
        ] = (-0.0075, 0.0075)

        # ---------------------------------------------------------------------
        # Termination
        # ---------------------------------------------------------------------

        self.terminations.anchor_pos.params[
            "threshold"
        ] = 0.40

        self.terminations.anchor_ori.params[
            "threshold"
        ] = 1.15

        # IMPORTANT:
        # Your later training log showed ee_body_pos caused ~99.98% of
        # terminations. Keep the V4 stability setup but carry forward this
        # confirmed fix so recovery is actually allowed to happen.
        self.terminations.ee_body_pos = None


# =============================================================================
# Existing low-frequency config
# =============================================================================


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.decimation = round(
            self.decimation / LOW_FREQ_SCALE
        )

        self.rewards.action_rate_l2.weight *= (
            LOW_FREQ_SCALE
        )
