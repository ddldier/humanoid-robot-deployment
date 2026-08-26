#!/usr/bin/env python3
"""Isaac-aligned MuJoCo Sim2Sim runner for WBT G1 Wo-State policy."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path("/root/gpufree-data/whole_body_tracking")

DEFAULT_MJCF = Path(
    "/root/gpufree-data/unitree_rl_mjlab/"
    "resource/g1_description/scene_29dof_official.xml"
)

# IMPORTANT:
# Use model_46500 exported ONNX by default.
DEFAULT_ONNX = (
    ROOT
    / "sim2sim_mujoco"
    / "exported"
    / "policy_wo_state_shortmove12_defaultpose_model_46500.onnx"
)

# IMPORTANT:
# Use the same 159-frame motion used for model_46500.
DEFAULT_MOTION = (
    ROOT
    / "artifacts"
    / "m_shortmove12_frames280_350_defaultpose_v4"
    / "motion.npz"
)

OBS_DIM = 154
ACT_DIM = 29


def isaac_velocity_limits(
    joint_names: list[str],
) -> np.ndarray:
    """Isaac Lab velocity_limit_sim values in ONNX joint-name order."""

    values = []

    for name in joint_names:
        if any(
            key in name
            for key in (
                "hip_yaw",
                "hip_pitch",
            )
        ):
            values.append(32.0)

        elif any(
            key in name
            for key in (
                "hip_roll",
                "knee",
            )
        ):
            values.append(20.0)

        elif (
            "wrist_pitch" in name
            or "wrist_yaw" in name
        ):
            values.append(22.0)

        else:
            values.append(37.0)

    return np.asarray(
        values,
        dtype=np.float64,
    )


def csv_floats(
    meta: dict[str, str],
    key: str,
) -> np.ndarray:

    values = np.fromstring(
        meta[key],
        sep=",",
        dtype=np.float64,
    )

    if values.size != ACT_DIM:
        raise RuntimeError(
            f"{key}: expected {ACT_DIM}, "
            f"got {values.size}"
        )

    return values


def load_metadata(
    session: ort.InferenceSession,
) -> dict:

    meta = (
        session
        .get_modelmeta()
        .custom_metadata_map
    )

    joint_names = meta["joint_names"].split(",")
    body_names = meta["body_names"].split(",")

    expected_obs = [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]

    if (
        joint_names
        and len(joint_names) != ACT_DIM
    ):
        raise RuntimeError(
            f"Expected 29 joints, "
            f"got {len(joint_names)}"
        )

    if (
        meta["observation_names"].split(",")
        != expected_obs
    ):
        raise RuntimeError(
            "Unexpected observation order: "
            f"{meta['observation_names']}"
        )

    anchor = meta["anchor_body_name"]

    if anchor not in body_names:
        raise RuntimeError(
            f"Anchor body {anchor!r} "
            "not in ONNX body_names"
        )

    return {
        "raw": meta,
        "joint_names": joint_names,
        "body_names": body_names,
        "anchor": anchor,
        "q_default": csv_floats(
            meta,
            "default_joint_pos",
        ),
        "action_scale": csv_floats(
            meta,
            "action_scale",
        ),
        "kp": csv_floats(
            meta,
            "joint_stiffness",
        ),
        "kd": csv_floats(
            meta,
            "joint_damping",
        ),
    }


def validate_onnx(
    session: ort.InferenceSession,
) -> None:

    inputs = {
        x.name: x
        for x in session.get_inputs()
    }

    outputs = {
        x.name: x
        for x in session.get_outputs()
    }

    if (
        inputs.get("obs") is None
        or inputs["obs"].shape[-1] != OBS_DIM
    ):
        raise RuntimeError(
            "Unexpected ONNX obs input: "
            f"{inputs.get('obs')}"
        )

    if inputs.get("time_step") is None:
        raise RuntimeError(
            "ONNX input 'time_step' is missing"
        )

    if (
        outputs.get("actions") is None
        or outputs["actions"].shape[-1]
        != ACT_DIM
    ):
        raise RuntimeError(
            "Unexpected ONNX action output: "
            f"{outputs.get('actions')}"
        )

    for name in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
    ):
        if name not in outputs:
            raise RuntimeError(
                f"ONNX output {name!r} "
                "is missing"
            )


def q_inv(
    q: np.ndarray,
) -> np.ndarray:

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    out = q.copy()
    out[1:] *= -1.0

    return out / np.dot(
        q,
        q,
    )


def q_mul(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    w1, x1, y1, z1 = np.asarray(
        a,
        dtype=np.float64,
    )

    w2, x2, y2, z2 = np.asarray(
        b,
        dtype=np.float64,
    )

    return np.array(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dtype=np.float64,
    )


def q_norm(
    q: np.ndarray,
) -> np.ndarray:

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    n = np.linalg.norm(q)

    if n < 1.0e-12:
        raise RuntimeError(
            "Zero quaternion"
        )

    return q / n


def matrix_from_quat(
    q: np.ndarray,
) -> np.ndarray:
    """Exact NumPy equivalent of Isaac Lab matrix_from_quat (wxyz)."""

    r, i, j, k = np.asarray(
        q,
        dtype=np.float64,
    )

    s = 2.0 / np.dot(
        q,
        q,
    )

    return np.array(
        [
            [
                1 - s*(j*j+k*k),
                s*(i*j-k*r),
                s*(i*k+j*r),
            ],
            [
                s*(i*j+k*r),
                1 - s*(i*i+k*k),
                s*(j*k-i*r),
            ],
            [
                s*(i*k-j*r),
                s*(j*k+i*r),
                1 - s*(i*i+j*j),
            ],
        ],
        dtype=np.float64,
    )


def anchor_ori_6d(
    current_wxyz: np.ndarray,
    reference_wxyz: np.ndarray,
) -> np.ndarray:
    """Isaac-style relative anchor orientation in 6D."""

    rel = q_mul(
        q_inv(current_wxyz),
        reference_wxyz,
    )

    return (
        matrix_from_quat(rel)
        [:, :2]
        .reshape(-1)
    )


def quat_error_deg(
    current: np.ndarray,
    target: np.ndarray,
) -> float:

    rel = q_norm(
        q_mul(
            q_inv(current),
            target,
        )
    )

    return math.degrees(
        2.0
        * math.acos(
            float(
                np.clip(
                    abs(rel[0]),
                    0.0,
                    1.0,
                )
            )
        )
    )


def load_motion(
    path: Path,
) -> dict[str, np.ndarray]:

    names = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    with np.load(path) as f:
        motion = {
            name: np.asarray(
                f[name]
            ).copy()
            for name in names
        }

    if (
        motion["joint_pos"].shape[1]
        != ACT_DIM
    ):
        raise RuntimeError(
            "Unexpected joint_pos shape "
            f"{motion['joint_pos'].shape}"
        )

    return motion


def onnx_reference(
    session: ort.InferenceSession,
    frame: int,
) -> dict[str, np.ndarray]:

    names = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
    ]

    values = session.run(
        names,
        {
            "obs": np.zeros(
                (1, OBS_DIM),
                np.float32,
            ),
            "time_step": np.array(
                [[frame]],
                np.float32,
            ),
        },
    )

    return {
        name: value[0]
        for name, value
        in zip(
            names,
            values,
        )
    }


def validate_motion(
    session: ort.InferenceSession,
    motion: dict[str, np.ndarray],
) -> None:

    n = motion["joint_pos"].shape[0]

    frames = sorted(
        {
            0,
            min(
                1,
                n - 1,
            ),
            min(
                100,
                n - 1,
            ),
            n - 1,
        }
    )

    for frame in frames:

        ref = onnx_reference(
            session,
            frame,
        )

        e1 = np.max(
            np.abs(
                ref["joint_pos"]
                - motion["joint_pos"][frame]
            )
        )

        e2 = np.max(
            np.abs(
                ref["joint_vel"]
                - motion["joint_vel"][frame]
            )
        )

        if max(e1, e2) > 1.0e-6:
            raise RuntimeError(
                "ONNX/motion mismatch at "
                f"frame {frame}: "
                f"{e1=}, {e2=}"
            )


def infer_motion_body_map(
    session: ort.InferenceSession,
    motion: dict[str, np.ndarray],
    selected_names: list[str],
) -> dict[str, int]:
    """Map ONNX selected bodies back to full motion arrays."""

    n = motion["joint_pos"].shape[0]

    frames = sorted(
        {
            0,
            min(
                1,
                n - 1,
            ),
            min(
                17,
                n - 1,
            ),
            min(
                100,
                n - 1,
            ),
            n - 1,
        }
    )

    refs = {
        f: onnx_reference(
            session,
            f,
        )
        for f in frames
    }

    scores = np.zeros(
        (
            len(selected_names),
            motion["body_pos_w"].shape[1],
        )
    )

    for i in range(
        scores.shape[0]
    ):
        for j in range(
            scores.shape[1]
        ):

            worst = 0.0

            for f in frames:

                pe = np.max(
                    np.abs(
                        refs[f]["body_pos_w"][i]
                        - motion[
                            "body_pos_w"
                        ][f, j]
                    )
                )

                qa = refs[f][
                    "body_quat_w"
                ][i]

                qb = motion[
                    "body_quat_w"
                ][f, j]

                qe = min(
                    np.max(
                        np.abs(
                            qa - qb
                        )
                    ),
                    np.max(
                        np.abs(
                            qa + qb
                        )
                    ),
                )

                worst = max(
                    worst,
                    float(pe),
                    float(qe),
                )

            scores[i, j] = worst

    best = np.argmin(
        scores,
        axis=1,
    )

    if (
        len(set(best.tolist()))
        != len(selected_names)
    ):
        raise RuntimeError(
            "Motion body mapping "
            "is not unique"
        )

    result = {}

    for i, j in enumerate(best):

        if scores[i, j] > 1.0e-5:
            raise RuntimeError(
                "Failed to map body "
                f"{selected_names[i]}: "
                f"error={scores[i,j]:.3e}"
            )

        result[selected_names[i]] = int(j)

    return result


def build_model_map(
    model: mujoco.MjModel,
    cfg: dict,
) -> dict:

    actuator_joint_names = []

    for aid in range(model.nu):

        jid = int(
            model.actuator_trnid[
                aid,
                0,
            ]
        )

        actuator_joint_names.append(
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )
            if jid >= 0
            else None
        )

    aids = []
    jids = []
    qpos = []
    qvel = []

    for name in cfg["joint_names"]:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if (
            jid < 0
            or name not in actuator_joint_names
        ):
            raise RuntimeError(
                "Missing MuJoCo "
                f"joint/actuator: {name}"
            )

        aid = actuator_joint_names.index(
            name
        )

        if (
            not np.isclose(
                model.actuator_gainprm[
                    aid,
                    0,
                ],
                1.0,
            )
            or not np.allclose(
                model.actuator_biasprm[
                    aid,
                    :3,
                ],
                0.0,
            )
        ):
            raise RuntimeError(
                f"Actuator {aid} for {name} "
                "is not a direct torque motor"
            )

        aids.append(aid)
        jids.append(jid)

        qpos.append(
            int(
                model.jnt_qposadr[
                    jid
                ]
            )
        )

        qvel.append(
            int(
                model.jnt_dofadr[
                    jid
                ]
            )
        )

    gyro = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "imu_gyro",
    )

    pelvis = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    anchor = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        cfg["anchor"],
    )

    if (
        min(
            gyro,
            pelvis,
            anchor,
        )
        < 0
        or int(
            model.sensor_dim[
                gyro
            ]
        )
        != 3
    ):
        raise RuntimeError(
            "Missing pelvis/anchor/"
            "imu_gyro in MuJoCo model"
        )

    return {
        "aids": np.asarray(
            aids,
            np.int32,
        ),
        "jids": np.asarray(
            jids,
            np.int32,
        ),
        "qpos": np.asarray(
            qpos,
            np.int32,
        ),
        "qvel": np.asarray(
            qvel,
            np.int32,
        ),
        "pelvis": pelvis,
        "anchor": anchor,
        "gyro_adr": int(
            model.sensor_adr[
                gyro
            ]
        ),
    }


def configure_model(
    model: mujoco.MjModel,
    m: dict,
    cfg: dict,
    args,
) -> np.ndarray:

    integrators = {
        "xml": None,
        "euler":
            mujoco.mjtIntegrator.mjINT_EULER,
        "implicitfast":
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }

    if (
        integrators[
            args.integrator
        ]
        is not None
    ):
        model.opt.integrator = (
            integrators[
                args.integrator
            ]
        )

    model.opt.timestep = args.sim_dt

    # Isaac:
    # action_scale =
    # 0.25 * effort_limit / stiffness
    isaac_limits = (
        4.0
        * cfg["action_scale"]
        * cfg["kp"]
    )

    if args.dynamics == "isaac":

        limits = isaac_limits

        omega = (
            2.0
            * math.pi
            * 10.0
        )

        armature = (
            cfg["kp"]
            / (omega * omega)
        )

        for i, (
            aid,
            jid,
            dof,
        ) in enumerate(
            zip(
                m["aids"],
                m["jids"],
                m["qvel"],
            )
        ):

            lim = float(
                limits[i]
            )

            model.actuator_ctrllimited[
                aid
            ] = 1

            model.actuator_ctrlrange[
                aid
            ] = [
                -lim,
                lim,
            ]

            model.jnt_actfrclimited[
                jid
            ] = 1

            model.jnt_actfrcrange[
                jid
            ] = [
                -lim,
                lim,
            ]

            model.dof_armature[
                dof
            ] = float(
                armature[i]
            )

            model.dof_damping[
                dof
            ] = 0.0

            model.dof_frictionloss[
                dof
            ] = 0.0

    else:

        limits = np.array(
            [
                min(
                    abs(
                        model.actuator_ctrlrange[
                            aid,
                            0,
                        ]
                    ),
                    abs(
                        model.actuator_ctrlrange[
                            aid,
                            1,
                        ]
                    ),
                )
                for aid
                in m["aids"]
            ],
            dtype=np.float64,
        )

    return limits


def reference_qpos(
    data: mujoco.MjData,
    m: dict,
    motion: dict,
    frame: int,
    pelvis_index: int,
):

    q = data.qpos.copy()

    q[:3] = motion[
        "body_pos_w"
    ][
        frame,
        pelvis_index,
    ]

    q[3:7] = q_norm(
        motion[
            "body_quat_w"
        ][
            frame,
            pelvis_index,
        ]
    )

    q[
        m["qpos"]
    ] = motion[
        "joint_pos"
    ][frame]

    return q


def initialize(
    model,
    data,
    m,
    motion,
    frame,
    pelvis_index,
):

    mujoco.mj_resetData(
        model,
        data,
    )

    q0 = reference_qpos(
        data,
        m,
        motion,
        frame,
        pelvis_index,
    )

    data.qpos[:] = q0
    data.qvel[:] = 0.0

    if (
        frame + 1
        < motion["joint_pos"].shape[0]
    ):

        q1 = data.qpos.copy()

        q1[:3] = motion[
            "body_pos_w"
        ][
            frame + 1,
            pelvis_index,
        ]

        q1[3:7] = q_norm(
            motion[
                "body_quat_w"
            ][
                frame + 1,
                pelvis_index,
            ]
        )

        q1[
            m["qpos"]
        ] = motion[
            "joint_pos"
        ][
            frame + 1
        ]

        try:

            mujoco.mj_differentiatePos(
                model,
                data.qvel,
                1.0
                / float(
                    motion[
                        "fps"
                    ].reshape(-1)[0]
                ),
                q0,
                q1,
            )

        except Exception as exc:

            print(
                "[WARNING] "
                "mj_differentiatePos failed; "
                "root velocity starts at zero: "
                f"{exc}"
            )

    data.qvel[
        m["qvel"]
    ] = motion[
        "joint_vel"
    ][frame]

    mujoco.mj_forward(
        model,
        data,
    )


def make_obs(
    data,
    m,
    cfg,
    ref_q,
    ref_dq,
    ref_anchor_q,
    last_action,
):

    ori6 = anchor_ori_6d(
        data.xquat[
            m["anchor"]
        ],
        ref_anchor_q,
    )

    gyro = data.sensordata[
        m["gyro_adr"]:
        m["gyro_adr"] + 3
    ]

    q = data.qpos[
        m["qpos"]
    ]

    dq = data.qvel[
        m["qvel"]
    ]

    obs = np.concatenate(
        [
            ref_q,
            ref_dq,
            ori6,
            gyro,
            q - cfg["q_default"],
            dq,
            last_action,
        ]
    ).astype(
        np.float32
    )

    if (
        obs.shape != (OBS_DIM,)
        or not np.isfinite(
            obs
        ).all()
    ):
        raise RuntimeError(
            "Invalid observation: "
            f"shape={obs.shape}, "
            "finite="
            f"{np.isfinite(obs).all()}"
        )

    return obs, ori6


def stage_name(
    ref_frame: int,
    raw_frame: int,
    frames: int,
    motion_mode: str,
) -> str:
    """Readable log stage for the current 159-frame punch motion."""

    if (
        motion_mode == "hold"
        and raw_frame >= frames
    ):
        return "hold_last"

    # Current shortmove12 v4:
    # 0-49 stance
    # 50-108 punch
    # 109-158 final stance
    if frames == 159:

        if ref_frame <= 49:
            return "stand_in"

        if ref_frame <= 108:
            return "punch"

        return "stand_out"

    return "motion"


def parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--mjcf",
        type=Path,
        default=DEFAULT_MJCF,
    )

    p.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
    )

    p.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOTION,
    )

    p.add_argument(
        "--mode",
        choices=(
            "policy",
            "reference",
        ),
        default="policy",
    )

    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Maximum policy control steps. "
            "0 = motion end for once mode; "
            "0 = unlimited for loop/hold."
        ),
    )

    p.add_argument(
        "--motion-mode",
        choices=(
            "once",
            "loop",
            "hold",
        ),
        default="hold",
        help=(
            "once: stop at motion end; "
            "loop: repeat 0..N-1 forever; "
            "hold: execute motion once then "
            "hold the final reference forever."
        ),
    )

    p.add_argument(
        "--validate-only",
        action="store_true",
    )

    p.add_argument(
        "--headless",
        action="store_true",
    )

    p.add_argument(
        "--no-realtime",
        action="store_true",
    )

    p.add_argument(
        "--sim-dt",
        type=float,
        default=0.005,
    )

    p.add_argument(
        "--integrator",
        choices=(
            "xml",
            "euler",
            "implicitfast",
        ),
        default="implicitfast",
    )

    p.add_argument(
        "--dynamics",
        choices=(
            "isaac",
            "xml",
        ),
        default="isaac",
    )

    p.add_argument(
        "--fall-height",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--no-stop-on-fall",
        action="store_true",
    )

    p.add_argument(
        "--strict-anchor-error-deg",
        type=float,
        default=15.0,
    )

    p.add_argument(
        "--action-abort",
        type=float,
        default=100.0,
    )

    return p


def main() -> int:

    args = parser().parse_args()

    for path in (
        args.mjcf,
        args.onnx,
        args.motion,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    session = ort.InferenceSession(
        str(args.onnx),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    validate_onnx(
        session
    )

    cfg = load_metadata(
        session
    )

    motion = load_motion(
        args.motion
    )

    validate_motion(
        session,
        motion,
    )

    body_map = infer_motion_body_map(
        session,
        motion,
        cfg["body_names"],
    )

    model = mujoco.MjModel.from_xml_path(
        str(args.mjcf)
    )

    m = build_model_map(
        model,
        cfg,
    )

    limits = configure_model(
        model,
        m,
        cfg,
        args,
    )

    velocity_limits = (
        isaac_velocity_limits(
            cfg["joint_names"]
        )
    )

    data = mujoco.MjData(
        model
    )

    frames = motion[
        "joint_pos"
    ].shape[0]

    fps = float(
        motion[
            "fps"
        ].reshape(-1)[0]
    )

    decimation_f = (
        (1.0 / fps)
        / model.opt.timestep
    )

    decimation = round(
        decimation_f
    )

    if (
        decimation < 1
        or abs(
            decimation
            - decimation_f
        )
        > 1.0e-7
    ):
        raise RuntimeError(
            f"1/{fps:g}s is not "
            "an integer multiple of "
            f"dt={model.opt.timestep}"
        )

    if not (
        0
        <= args.start_frame
        < frames
    ):
        raise ValueError(
            "start frame must be in "
            f"[0, {frames-1}]"
        )

    pelvis_motion = body_map[
        "pelvis"
    ]

    anchor_motion = body_map[
        cfg["anchor"]
    ]

    initialize(
        model,
        data,
        m,
        motion,
        args.start_frame,
        pelvis_motion,
    )

    initial_ref_anchor = motion[
        "body_quat_w"
    ][
        args.start_frame,
        anchor_motion,
    ]

    initial_error = quat_error_deg(
        data.xquat[
            m["anchor"]
        ],
        initial_ref_anchor,
    )

    initial_ori6 = anchor_ori_6d(
        data.xquat[
            m["anchor"]
        ],
        initial_ref_anchor,
    )

    print(
        "=== Validation ==="
    )

    print(
        "Run: "
        f"{cfg['raw'].get('run_path', 'unknown')}"
    )

    print(
        f"ONNX: {OBS_DIM} -> {ACT_DIM}; "
        f"motion frames={frames}, "
        f"fps={fps:g}"
    )

    print(
        f"motion_mode={args.motion_mode}; "
        f"max_frames={args.max_frames}"
    )

    print(
        f"MuJoCo dt="
        f"{model.opt.timestep:g}; "
        f"decimation={decimation}; "
        f"dynamics={args.dynamics}"
    )

    print(
        "Control period="
        f"{model.opt.timestep * decimation:.5f}s; "
        "Isaac target=0.02000s "
        "(dt=0.005, decimation=4)"
    )

    if args.dynamics == "isaac":
        print(
            "Passive joint damping/"
            "frictionloss: cleared to "
            "match Isaac actuator config"
        )

    print(
        f"Motion body map: "
        f"{body_map}"
    )

    print(
        "Torque limits: "
        f"min={limits.min():.2f}, "
        f"max={limits.max():.2f} Nm"
    )

    print(
        "Initial pelvis: "
        f"{data.xpos[m['pelvis']].tolist()}, "
        f"contacts={data.ncon}"
    )

    print(
        "Initial torso orientation error: "
        f"{initial_error:.4f} deg"
    )

    print(
        "Initial motion_anchor_ori_b: "
        f"{np.array2string(initial_ori6, precision=5)}"
    )

    if (
        initial_error
        > args.strict_anchor_error_deg
    ):
        raise RuntimeError(
            "Initial torso frame mismatch "
            "is too large"
        )

    if args.validate_only:
        print(
            "Validation-only: PASS"
        )
        return 0

    viewer = None

    if not args.headless:

        from mujoco import (
            viewer as mj_viewer
        )

        viewer = (
            mj_viewer.launch_passive(
                model,
                data,
            )
        )

        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        # Initial sync is useful so the viewer
        # immediately has the current simulation state.
        viewer.sync()

    # Raw frame counter.
    #
    # once:
    #   0 1 ... 158 -> stop
    #
    # loop:
    #   raw frame keeps increasing forever;
    #   ref_frame = raw_frame % 159
    #
    # hold:
    #   raw frame keeps increasing forever;
    #   ref_frame = min(raw_frame, 158)
    frame = args.start_frame

    executed = 0

    last_action = np.zeros(
        ACT_DIM,
        dtype=np.float64,
    )

    wall0 = time.perf_counter()
    sim0 = float(
        data.time
    )

    next_log = float(
        data.time
    )

    try:

        while True:

            # -----------------------------------------
            # Viewer closed -> stop.
            # -----------------------------------------
            if (
                viewer is not None
                and not viewer.is_running()
            ):
                break

            # -----------------------------------------
            # Optional finite run length.
            #
            # max_frames counts policy/control steps,
            # not reference-frame indices.
            # -----------------------------------------
            if (
                args.max_frames > 0
                and executed
                >= args.max_frames
            ):
                break

            raw_frame = frame

            # -----------------------------------------
            # Select legal reference frame.
            # -----------------------------------------
            if args.motion_mode == "once":

                if raw_frame >= frames:
                    break

                ref_frame = raw_frame

            elif args.motion_mode == "loop":

                ref_frame = (
                    raw_frame
                    % frames
                )

            elif args.motion_mode == "hold":

                ref_frame = min(
                    raw_frame,
                    frames - 1,
                )

            else:
                raise RuntimeError(
                    "Unknown motion mode: "
                    f"{args.motion_mode}"
                )

            # -----------------------------------------
            # Reference data.
            # -----------------------------------------
            ref_q = motion[
                "joint_pos"
            ][
                ref_frame
            ]

            ref_dq = motion[
                "joint_vel"
            ][
                ref_frame
            ]

            ref_anchor = motion[
                "body_quat_w"
            ][
                ref_frame,
                anchor_motion,
            ]

            # -----------------------------------------
            # Build exactly the same 154-D observation.
            # -----------------------------------------
            obs, _ = make_obs(
                data,
                m,
                cfg,
                ref_q,
                ref_dq,
                ref_anchor,
                last_action,
            )

            # -----------------------------------------
            # Policy / reference controller.
            # -----------------------------------------
            if args.mode == "policy":

                action = session.run(
                    [
                        "actions"
                    ],
                    {
                        "obs":
                            obs.reshape(
                                1,
                                OBS_DIM,
                            ),

                        # CRITICAL:
                        # Never pass raw_frame here
                        # in loop/hold mode.
                        #
                        # hold -> always 158
                        # loop -> always 0..158
                        "time_step":
                            np.array(
                                [[ref_frame]],
                                np.float32,
                            ),
                    },
                )[0][0].astype(
                    np.float64
                )

                if (
                    not np.isfinite(
                        action
                    ).all()
                    or np.max(
                        np.abs(action)
                    )
                    > args.action_abort
                ):
                    raise RuntimeError(
                        "Invalid policy action "
                        f"at raw={raw_frame}, "
                        f"ref={ref_frame}: "
                        "max="
                        f"{np.max(np.abs(action))}"
                    )

                target = (
                    cfg["q_default"]
                    + cfg["action_scale"]
                    * action
                )

                # Do NOT reset this at loop boundary.
                last_action = (
                    action.copy()
                )

            else:

                action = np.zeros(
                    ACT_DIM,
                    dtype=np.float64,
                )

                target = ref_q

                last_action.fill(
                    0.0
                )

            max_tau = 0.0
            max_sat = 0

            # -----------------------------------------
            # Physics steps:
            # 4 * 0.005 = 0.020 s
            # policy = 50 Hz.
            # -----------------------------------------
            for _ in range(
                decimation
            ):

                q = data.qpos[
                    m["qpos"]
                ]

                dq = data.qvel[
                    m["qvel"]
                ]

                requested = (
                    cfg["kp"]
                    * (
                        target
                        - q
                    )
                    - cfg["kd"]
                    * dq
                )

                tau = np.clip(
                    requested,
                    -limits,
                    limits,
                )

                max_tau = max(
                    max_tau,
                    float(
                        np.max(
                            np.abs(tau)
                        )
                    ),
                )

                max_sat = max(
                    max_sat,
                    int(
                        np.count_nonzero(
                            np.abs(
                                requested
                            )
                            > (
                                limits
                                + 1.0e-9
                            )
                        )
                    ),
                )

                data.ctrl[:] = 0.0

                data.ctrl[
                    m["aids"]
                ] = tau

                mujoco.mj_step(
                    model,
                    data,
                )

                if (
                    not np.isfinite(
                        data.qpos
                    ).all()
                    or not np.isfinite(
                        data.qvel
                    ).all()
                ):
                    raise RuntimeError(
                        "NaN/Inf in MuJoCo "
                        "state at "
                        f"raw={raw_frame}, "
                        f"ref={ref_frame}"
                    )

                if viewer is not None:

                    if not viewer.is_running():
                        return 0

                    # Follow pelvis with camera.
                    viewer.cam.lookat[:] = (
                        data.xpos[
                            m["pelvis"]
                        ]
                    )

                    # CRITICAL for interactive
                    # mouse perturbations.
                    viewer.sync()

                if not args.no_realtime:

                    delay = (
                        wall0
                        + (
                            float(
                                data.time
                            )
                            - sim0
                        )
                        - time.perf_counter()
                    )

                    if delay > 0:
                        time.sleep(
                            delay
                        )

            # One policy/control step finished.
            executed += 1
            frame += 1

            pelvis_z = float(
                data.xpos[
                    m["pelvis"],
                    2,
                ]
            )

            if (
                data.time
                >= next_log
            ):

                stage = stage_name(
                    ref_frame,
                    raw_frame,
                    frames,
                    args.motion_mode,
                )

                abs_qvel = np.abs(
                    data.qvel[
                        m["qvel"]
                    ]
                )

                vel_ratio = float(
                    np.max(
                        abs_qvel
                        / velocity_limits
                    )
                )

                if (
                    args.motion_mode
                    == "loop"
                ):
                    cycle = (
                        raw_frame
                        // frames
                    )
                else:
                    cycle = 0

                print(
                    f"t={data.time:7.3f}s "
                    f"step={executed:06d} "
                    f"raw={raw_frame:06d} "
                    f"ref={ref_frame:03d} "
                    f"cycle={cycle:03d} "
                    f"stage={stage:>10s} "
                    f"pelvis_z={pelvis_z:6.3f} "
                    f"anchor_err="
                    f"{quat_error_deg(data.xquat[m['anchor']], ref_anchor):6.2f}deg "
                    f"max_q_err="
                    f"{np.max(np.abs(data.qpos[m['qpos']] - target)):6.3f} "
                    f"max_action="
                    f"{np.max(np.abs(action)):6.3f} "
                    f"max_tau={max_tau:6.2f} "
                    f"sat={max_sat:02d} "
                    f"vel_ratio={vel_ratio:5.2f} "
                    f"ncon={data.ncon:02d}"
                )

                next_log += 1.0

            if (
                pelvis_z
                < args.fall_height
                and not args.no_stop_on_fall
            ):
                print(
                    "[STOP] "
                    f"pelvis_z={pelvis_z:.3f} "
                    "< "
                    f"{args.fall_height:.3f}"
                )
                return 2

    except KeyboardInterrupt:

        print(
            "Interrupted"
        )

    finally:

        if viewer is not None:
            viewer.close()

    print(
        "Finished: "
        f"control_steps={executed}, "
        "final pelvis="
        f"{data.xpos[m['pelvis']].tolist()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
