#!/usr/bin/env python3
from pathlib import Path
import argparse
import shutil
import time
import yaml
import numpy as np

JOINT_MAP = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
Q_DEFAULT = [-0.312,-0.312,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.669,0.669,0.200,0.200,-0.363,-0.363,0.200,-0.200,0.000,0.000,0.000,0.000,0.600,0.600,0.000,0.000,0.000,0.000,0.000,0.000]
ACTION_SCALE = [0.548,0.548,0.548,0.351,0.351,0.439,0.548,0.548,0.439,0.351,0.351,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.439,0.075,0.075,0.075,0.075]
KP_POLICY = [40.179,40.179,40.179,99.098,99.098,28.501,40.179,40.179,28.501,99.098,99.098,14.251,14.251,28.501,28.501,14.251,14.251,28.501,28.501,14.251,14.251,14.251,14.251,14.251,14.251,16.778,16.778,16.778,16.778]
KD_POLICY = [2.558,2.558,2.558,6.309,6.309,1.814,2.558,2.558,1.814,6.309,6.309,0.907,0.907,1.814,1.814,0.907,0.907,1.814,1.814,0.907,0.907,0.907,0.907,0.907,0.907,1.068,1.068,1.068,1.068]

def to_sdk(values):
    out = [0.0] * 29
    for policy_i, sdk_i in enumerate(JOINT_MAP):
        out[sdk_i] = float(values[policy_i])
    return out

def backup(path, backup_dir):
    path = Path(path)
    if path.exists():
        dst = backup_dir / path.name
        shutil.copy2(path, dst)
        print("[backup]", dst)

def write_generalized_fixstand(path):
    text = r'''// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        ts_ = cfg["ts"].as<std::vector<float>>();
        qs_ = cfg["qs"].as<std::vector<std::vector<float>>>();

        assert(kp_.size() == kd_.size());
        assert(ts_.size() == qs_.size());
    }

    void enter()
    {
        for(int i = 0; i < kp_.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }

        std::vector<float> q0;
        for(int i = 0; i < kp_.size(); ++i)
        {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }

        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t =
            (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3
            - t0_;

        auto q = linear_interpolate(t, ts_, qs_);

        for(int i = 0; i < q.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    double t0_;
    std::vector<float> kp_;
    std::vector<float> kd_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
};

REGISTER_FSM(State_FixStand)
'''
    path.write_text(text)

def patch_keyboard(fsm_h):
    s = fsm_h.read_text()
    tag = "// G9 action-set keyboard shortcuts"
    if tag in s:
        print("[keyboard] G9 shortcuts already present")
        return
    marker = "// register for all states"
    pos = s.find(marker)
    if pos < 0:
        raise SystemExit("ERROR: FSMState.h marker not found")
    insert = r'''
        // G9 action-set keyboard shortcuts
        // g : Passive -> G9Stand
        // k : G9Stand -> G9TailDirectV2
        if(
            state_string == "Passive"
            && FSMStringMap.right.count("G9Stand")
        )
        {
            registered_checks.emplace_back(
                std::make_pair(
                    []()->bool
                    {
                        return FSMState::keyboard
                            && FSMState::keyboard->key() == "g";
                    },
                    FSMStringMap.right.at("G9Stand")
                )
            );
        }

        if(
            state_string == "G9Stand"
            && FSMStringMap.right.count("G9TailDirectV2")
        )
        {
            registered_checks.emplace_back(
                std::make_pair(
                    []()->bool
                    {
                        return FSMState::keyboard
                            && FSMState::keyboard->key() == "k";
                    },
                    FSMStringMap.right.at("G9TailDirectV2")
                )
            );
        }

        '''
    s = s[:pos] + insert + s[pos:]
    fsm_h.write_text(s)
    print("[keyboard] added g -> G9Stand, k -> G9TailDirectV2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", default="/root/gpufree-data/G9_tail_direct_v2_handoff")
    ap.add_argument("--workspace", default="/root/g1_sdk2_sim2sim")
    args = ap.parse_args()

    pkg = Path(args.package_root).expanduser().resolve()
    ws = Path(args.workspace).expanduser().resolve()
    base = ws / "unitree_rl_lab/deploy/robots/g1_29dof"

    policy_src = pkg / "deploy/policy.onnx"
    motion_src = pkg / "motion/G9_rightfeiti_tail_direct_v2.npz"
    config_yaml = base / "config/config.yaml"
    template_yaml = base / "config/policy/mimic/punch46500/params/deploy.yaml"
    mimic_h = base / "include/State_Mimic.h"
    mimic_cpp = base / "src/State_Mimic.cpp"
    fixstand_h = ws / "unitree_rl_lab/deploy/include/FSM/State_FixStand.h"
    fsm_h = ws / "unitree_rl_lab/deploy/include/FSM/FSMState.h"

    required = [policy_src, motion_src, config_yaml, template_yaml, mimic_h, mimic_cpp, fixstand_h, fsm_h]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("ERROR: missing required files:\n" + "\n".join(missing))

    config_before = yaml.safe_load(config_yaml.read_text())
    if "Punch46500" not in config_before["FSM"]["_"]:
        raise SystemExit("ERROR: Punch46500 state not found; refusing to patch.")

    if "has_exact_qdq" not in mimic_h.read_text(errors="ignore"):
        raise SystemExit("ERROR: State_Mimic.h missing 65-column q/dq patch.")

    mimic_cpp_text = mimic_cpp.read_text(errors="ignore")
    if "wbt_time_step" not in mimic_cpp_text:
        raise SystemExit("ERROR: State_Mimic.cpp missing wbt_time_step.")
    if "PUNCH46500_NO_OUTPUT_DEBUG" in mimic_cpp_text:
        raise SystemExit("ERROR: State_Mimic.cpp still contains PUNCH46500_NO_OUTPUT_DEBUG.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = ws / "backups" / ("before_g9_actionset_" + stamp)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in [config_yaml, fixstand_h, fsm_h, template_yaml]:
        backup(p, backup_dir)

    m = np.load(motion_src)
    required_npz = ["joint_pos","joint_vel","body_pos_w","body_quat_w","fps","deployment_stance_joint_pos"]
    for key in required_npz:
        if key not in m:
            raise SystemExit("ERROR: NPZ missing key: " + key)

    q = np.asarray(m["joint_pos"], dtype=np.float32)
    dq = np.asarray(m["joint_vel"], dtype=np.float32)
    body_pos = np.asarray(m["body_pos_w"], dtype=np.float32)
    body_quat = np.asarray(m["body_quat_w"], dtype=np.float32)
    stance = np.asarray(m["deployment_stance_joint_pos"], dtype=np.float32)
    fps = float(np.asarray(m["fps"]).reshape(-1)[0])

    if q.shape != (114, 29):
        raise SystemExit(f"ERROR: expected joint_pos (114,29), got {q.shape}")
    if dq.shape != (114, 29):
        raise SystemExit(f"ERROR: expected joint_vel (114,29), got {dq.shape}")
    if stance.shape != (29,):
        raise SystemExit(f"ERROR: expected stance (29,), got {stance.shape}")
    if abs(fps - 50.0) > 1e-6:
        raise SystemExit(f"ERROR: expected 50 Hz, got {fps}")

    frame0_err = float(np.max(np.abs(q[0] - stance)))
    last_err = float(np.max(np.abs(q[-1] - stance)))
    if frame0_err > 1e-6:
        raise SystemExit(f"ERROR: G9 frame0 != deployment stance: {frame0_err}")
    if last_err > 1e-6:
        raise SystemExit(f"ERROR: G9 last frame != deployment stance: {last_err}")

    q_sdk = np.zeros_like(q)
    dq_sdk = np.zeros_like(dq)
    for policy_i, sdk_i in enumerate(JOINT_MAP):
        q_sdk[:, sdk_i] = q[:, policy_i]
        dq_sdk[:, sdk_i] = dq[:, policy_i]

    root_pos = body_pos[:, 0, :]
    root_wxyz = body_quat[:, 0, :]
    root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
    csv_data = np.concatenate([root_pos, root_xyzw, q_sdk, dq_sdk], axis=1)
    if csv_data.shape != (114, 65):
        raise SystemExit(f"ERROR: generated CSV shape {csv_data.shape}")

    out_dir = base / "config/policy/mimic/g9_tail_direct_v2"
    exported = out_dir / "exported"
    params = out_dir / "params"
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_src, exported / "policy.onnx")
    motion_csv = params / "g9_tail_direct_v2_qdq_50hz.csv"
    np.savetxt(motion_csv, csv_data, delimiter=",", fmt="%.9g")

    deploy = yaml.safe_load(template_yaml.read_text())
    deploy["joint_ids_map"] = JOINT_MAP
    deploy["step_dt"] = 0.02
    deploy["stiffness"] = to_sdk(KP_POLICY)
    deploy["damping"] = to_sdk(KD_POLICY)
    deploy["default_joint_pos"] = Q_DEFAULT

    if "actions" not in deploy or len(deploy["actions"]) != 1:
        raise SystemExit("ERROR: unexpected actions structure in Punch deploy.yaml")
    action_cfg = next(iter(deploy["actions"].values()))
    action_cfg["scale"] = ACTION_SCALE
    action_cfg["offset"] = Q_DEFAULT
    action_cfg["joint_ids"] = None

    if "observations" not in deploy:
        raise SystemExit("ERROR: observations missing from deploy template")
    if set(deploy["observations"].keys()) != {"obs", "time_step"}:
        raise SystemExit("ERROR: deploy template obs groups are not obs + time_step")

    (params / "deploy.yaml").write_text(
        yaml.safe_dump(deploy, sort_keys=False, default_flow_style=False)
    )

    cfg = yaml.safe_load(config_yaml.read_text())
    enabled = cfg["FSM"]["_"]
    existing_ids = set()
    for value in enabled.values():
        if isinstance(value, dict) and "id" in value:
            try:
                existing_ids.add(int(value["id"]))
            except Exception:
                pass

    if "G9Stand" in enabled:
        g9_stand_id = int(enabled["G9Stand"]["id"])
    else:
        g9_stand_id = max(existing_ids | {103}) + 1
        existing_ids.add(g9_stand_id)

    if "G9TailDirectV2" in enabled:
        g9_policy_id = int(enabled["G9TailDirectV2"]["id"])
    else:
        g9_policy_id = max(existing_ids | {g9_stand_id}) + 1

    enabled["G9Stand"] = {"id": g9_stand_id, "type": "FixStand"}
    enabled["G9TailDirectV2"] = {"id": g9_policy_id, "type": "Mimic"}

    original_fix = cfg["FSM"]["FixStand"]
    cfg["FSM"]["G9Stand"] = {
        "transitions": {"Passive": "LT + B.on_pressed"},
        "kp": list(original_fix["kp"]),
        "kd": list(original_fix["kd"]),
        "ts": [0.0, 4.0],
        "qs": [[], [float(x) for x in q_sdk[0]]],
    }
    cfg["FSM"]["G9TailDirectV2"] = {
        "transitions": {"Passive": "LT + B.on_pressed"},
        "fps": 50,
        "motion_file": "config/policy/mimic/g9_tail_direct_v2/params/g9_tail_direct_v2_qdq_50hz.csv",
        "policy_dir": "config/policy/mimic/g9_tail_direct_v2/",
        "time_start": 0.0,
        "hold_last": True,
    }

    config_yaml.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    )

    write_generalized_fixstand(fixstand_h)
    patch_keyboard(fsm_h)

    cfg_check = yaml.safe_load(config_yaml.read_text())
    if "Punch46500" not in cfg_check["FSM"]["_"]:
        raise SystemExit("ERROR: Punch46500 disappeared after patch.")
    if "G9Stand" not in cfg_check["FSM"]["_"]:
        raise SystemExit("ERROR: G9Stand missing after patch.")
    if "G9TailDirectV2" not in cfg_check["FSM"]["_"]:
        raise SystemExit("ERROR: G9TailDirectV2 missing after patch.")
    if cfg_check["FSM"]["FixStand"]["qs"] != config_before["FSM"]["FixStand"]["qs"]:
        raise SystemExit("ERROR: existing Punch FixStand pose was modified.")

    print()
    print("=== G9 ACTION SET PATCHED ===")
    print("Punch46500: PRESERVED")
    print("existing FixStand pose: PRESERVED")
    print("G9Stand state id:", g9_stand_id)
    print("G9 policy state id:", g9_policy_id)
    print("G9 policy:", exported / "policy.onnx")
    print("G9 motion:", motion_csv)
    print("motion shape:", csv_data.shape)
    print("fps:", fps)
    print("frame0 stance error:", frame0_err)
    print("last stance error:", last_err)
    print("backup:", backup_dir)
    print()
    print("Keyboard:")
    print("  f = Passive -> original FixStand")
    print("  m = original FixStand -> Punch46500")
    print("  g = Passive -> G9Stand")
    print("  k = G9Stand -> G9TailDirectV2")
    print("  p = any active state -> Passive")
    print()
    print("Next:")
    print("  cd", base)
    print("  cmake -S . -B build")
    print("  cmake --build build -j4")

if __name__ == "__main__":
    main()
