"""Minimal WBT play script: one large forward target + alternating lateral pushes."""

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Play WBT G1 with one large forward target and lateral pushes."
)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)

# Large forward target.
parser.add_argument("--target_x", type=float, default=0.55)
parser.add_argument("--target_y", type=float, default=0.0)
parser.add_argument("--target_z", type=float, default=1.05)
parser.add_argument("--target_mass", type=float, default=8.0)

# Alternating lateral disturbance.
parser.add_argument("--push_start", type=float, default=2.5)
parser.add_argument("--push_every", type=float, default=2.0)
parser.add_argument("--push_dv", type=float, default=1.5)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401


def get_policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, dict):
        obs = obs["policy"]

    if hasattr(obs, "dim"):
        while obs.dim() > 2:
            squeezed = False
            for dim in range(obs.dim()):
                if obs.shape[dim] == 1:
                    obs = obs.squeeze(dim)
                    squeezed = True
                    break

            if not squeezed:
                obs = obs.reshape(-1, obs.shape[-1])
                break

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

    return obs


def get_policy_actions(actions, env):
    if isinstance(actions, tuple):
        actions = actions[0]

    if isinstance(actions, list):
        actions = actions[0]

    expected_dim = getattr(
        getattr(env.unwrapped, "action_manager", None),
        "total_action_dim",
        None,
    )

    if hasattr(actions, "dim"):
        while actions.dim() > 2:
            squeezed = False
            for dim in range(actions.dim()):
                if actions.shape[dim] == 1:
                    actions = actions.squeeze(dim)
                    squeezed = True
                    break

            if not squeezed:
                actions = actions.reshape(-1, actions.shape[-1])
                break

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        if (
            actions.dim() == 2
            and expected_dim is not None
            and actions.shape[1] != expected_dim
            and actions.shape[0] == expected_dim
        ):
            actions = actions.transpose(0, 1).contiguous()

    return actions


@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    agent_cfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task,
        args_cli,
    )

    env_cfg.scene.num_envs = args_cli.num_envs

    # -------------------------------------------------------------------------
    # Motion file
    # -------------------------------------------------------------------------

    motion_path = pathlib.Path(
        args_cli.motion_file
    ).expanduser().resolve()

    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Motion file does not exist: {motion_path}"
        )

    env_cfg.commands.motion.motion_file = str(motion_path)

    # -------------------------------------------------------------------------
    # Deterministic evaluation:
    # remove training-time random disturbances/randomization.
    # -------------------------------------------------------------------------

    if hasattr(env_cfg, "events"):
        for name in (
            "attack_reaction",
            "push_robot",
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "body_mass",
            "actuator_gains",
        ):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)

    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    if hasattr(env_cfg, "observations") and hasattr(
        env_cfg.observations,
        "policy",
    ):
        env_cfg.observations.policy.enable_corruption = False

    # -------------------------------------------------------------------------
    # One large forward target.
    #
    # It starts stationary, but remains a real dynamic rigid body.
    # Therefore --target_mass changes the physical interaction.
    # -------------------------------------------------------------------------

    env_cfg.scene.punch_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PunchTarget",
        spawn=sim_utils.CuboidCfg(
            # thickness, width, height
            size=(0.12, 0.90, 0.80),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=1.2,
                angular_damping=2.0,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=args_cli.target_mass,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.80, 0.10, 0.10),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                args_cli.target_x,
                args_cli.target_y,
                args_cli.target_z,
            ),
        ),
        collision_group=0,
    )

    print(
        "[TARGET] "
        f"x={args_cli.target_x:.2f}, "
        f"y={args_cli.target_y:.2f}, "
        f"z={args_cli.target_z:.2f}, "
        f"mass={args_cli.target_mass:.2f} kg"
    )

    print(
        "[PUSH] "
        f"start={args_cli.push_start:.2f}s, "
        f"every={args_cli.push_every:.2f}s, "
        f"dv=+/-{args_cli.push_dv:.2f} m/s"
    )

    # -------------------------------------------------------------------------
    # Local checkpoint
    # -------------------------------------------------------------------------

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    resume_path = get_checkpoint_path(
        log_root_path,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
    )

    print(f"[INFO] Motion: {motion_path}")
    print(f"[INFO] Checkpoint: {resume_path}")

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(
        device=env.unwrapped.device
    )

    # -------------------------------------------------------------------------
    # Runtime references
    # -------------------------------------------------------------------------

    scene = env.unwrapped.scene
    robot = scene["robot"]
    punch_target = scene["punch_target"]

    motion_command = (
        env.unwrapped.command_manager.get_term("motion")
    )

    control_dt = env.unwrapped.step_dt

    first_push_step = max(
        1,
        int(args_cli.push_start / control_dt),
    )

    push_interval_steps = max(
        1,
        int(args_cli.push_every / control_dt),
    )

    demo_step = 0
    push_count = 0
    next_push_step = first_push_step

    previous_motion_steps = motion_command.time_steps.clone()

    # -------------------------------------------------------------------------
    # Reset large target in front of current robot heading.
    # -------------------------------------------------------------------------

    def reset_punch_target():
        root_pos_w = robot.data.root_pos_w

        heading_quat = math_utils.yaw_quat(
            robot.data.root_quat_w
        )

        local_offset = torch.zeros(
            (root_pos_w.shape[0], 3),
            device=robot.device,
            dtype=root_pos_w.dtype,
        )

        local_offset[:, 0] = args_cli.target_x
        local_offset[:, 1] = args_cli.target_y

        offset_w = math_utils.quat_apply(
            heading_quat,
            local_offset,
        )

        target_pos_w = root_pos_w + offset_w

        target_pos_w[:, 2] = (
            scene.env_origins[:, 2]
            + args_cli.target_z
        )

        state = punch_target.data.default_root_state.clone()

        state[:, :3] = target_pos_w
        state[:, 3:7] = heading_quat
        state[:, 7:13] = 0.0

        punch_target.write_root_state_to_sim(state)

        print("[TARGET] reset")

    # -------------------------------------------------------------------------
    # Alternating heading-frame lateral delta-v push.
    # -------------------------------------------------------------------------

    def apply_lateral_push(side_sign: float):
        root_state = robot.data.root_state_w.clone()

        heading_quat = math_utils.yaw_quat(
            root_state[:, 3:7]
        )

        local_dv = torch.zeros(
            (root_state.shape[0], 3),
            device=robot.device,
            dtype=root_state.dtype,
        )

        # +Y = robot left, -Y = robot right.
        local_dv[:, 1] = (
            side_sign * args_cli.push_dv
        )

        dv_world = math_utils.quat_apply(
            heading_quat,
            local_dv,
        )

        # Add disturbance to current velocity.
        root_state[:, 7:10] += dv_world

        robot.write_root_state_to_sim(root_state)

        direction = (
            "LEFT"
            if side_sign > 0.0
            else "RIGHT"
        )

        print(
            "\n"
            "========================================\n"
            f"[LATERAL PUSH] {direction}\n"
            f"delta-v = {args_cli.push_dv:.2f} m/s\n"
            "========================================\n"
        )

    # -------------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------------

    reset_punch_target()

    obs = get_policy_obs(
        env.get_observations()
    )

    printed_shape = False

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

            actions = get_policy_actions(
                actions,
                env,
            )

            if not printed_shape:
                print(
                    "[DEBUG] obs shape =",
                    tuple(obs.shape)
                    if hasattr(obs, "shape")
                    else type(obs),
                )

                print(
                    "[DEBUG] action shape =",
                    tuple(actions.shape)
                    if hasattr(actions, "shape")
                    else type(actions),
                )

                printed_shape = True

            # Alternating big lateral push.
            if demo_step >= next_push_step:
                side_sign = (
                    +1.0
                    if push_count % 2 == 0
                    else -1.0
                )

                apply_lateral_push(side_sign)

                push_count += 1
                next_push_step += push_interval_steps

            step_out = env.step(actions)

            if isinstance(step_out, tuple):
                obs = step_out[0]
            else:
                obs = step_out

            obs = get_policy_obs(obs)

            # Put target back when motion wraps/resamples.
            current_motion_steps = (
                motion_command.time_steps.clone()
            )

            if torch.any(
                current_motion_steps
                < previous_motion_steps
            ):
                reset_punch_target()

            previous_motion_steps = current_motion_steps

            demo_step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
