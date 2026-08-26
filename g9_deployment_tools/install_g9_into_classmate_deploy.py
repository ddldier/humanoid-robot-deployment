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

def backup_file(path: Path, backup_dir: Path):
    dst = backup_dir / path.name
    shutil.copy2(path, dst)
    print('[backup]', dst)

def patch_state_mimic_h(path: Path):
    s = path.read_text()
    if 'const bool has_exact_qdq' in s:
        print('[State_Mimic.h] exact q/dq support already present')
        return
    old = """        for(int i(0); i < num_frames; ++i)
        {
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6],data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }
        dof_velocities = _comupte_raw_derivative(dof_positions);
"""
    new = """        const bool has_exact_qdq =
            !data.empty() && data[0].size() == 65;

        for(int i(0); i < num_frames; ++i)
        {
            if(data[i].size() != data[0].size())
            {
                throw std::runtime_error(\"Motion CSV has inconsistent column counts.\");
            }

            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(
                Eigen::Quaternionf(
                    data[i][6],
                    data[i][3],
                    data[i][4],
                    data[i][5]
                )
            );

            if(has_exact_qdq)
            {
                dof_positions.push_back(
                    Eigen::VectorXf::Map(data[i].data() + 7, 29)
                );
                dof_velocities.push_back(
                    Eigen::VectorXf::Map(data[i].data() + 36, 29)
                );
            }
            else
            {
                dof_positions.push_back(
                    Eigen::VectorXf::Map(
                        data[i].data() + 7,
                        data[i].size() - 7
                    )
                );
            }
        }

        if(!has_exact_qdq)
        {
            dof_velocities = _comupte_raw_derivative(dof_positions);
        }
"""
    if old not in s:
        raise RuntimeError('State_Mimic.h does not match the classmate deploy version.')
    s = s.replace(old, new, 1)
    if '#include <stdexcept>' not in s:
        s = s.replace('#include "FSM/State_RLBase.h"\n', '#include "FSM/State_RLBase.h"\n#include <stdexcept>\n', 1)
    path.write_text(s)
    print('[State_Mimic.h] added 65-column exact q/dq support')

def patch_state_mimic_cpp(path: Path):
    s = path.read_text()
    if 'REGISTER_OBSERVATION(wbt_time_step)' not in s:
        namespace_close = '\n}\n}\n\n\nState_Mimic::State_Mimic'
        pos = s.find(namespace_close)
        if pos < 0:
            raise RuntimeError('State_Mimic namespace closing marker not found.')
        block = r'''
REGISTER_OBSERVATION(wbt_time_step)
{
    long frame =
        std::max<long>(
            0,
            env->episode_length - 1
        );

    if(State_Mimic::motion)
    {
        frame =
            std::min<long>(
                frame,
                State_Mimic::motion->num_frames - 1
            );
    }

    return {
        static_cast<float>(frame)
    };
}
'''
        s = s[:pos] + '\n' + block + s[pos:]
        print('[State_Mimic.cpp] added wbt_time_step observation')
    else:
        print('[State_Mimic.cpp] wbt_time_step already present')

    old_checks = """    const auto & joy = FSMState::lowstate->joystick;
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return (env->episode_length * env->step_dt) > time_range_[1]; }, // time out
            FSMStringMap.right.at(\"Velocity\")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            FSMStringMap.right.at(\"Passive\")
        )
    );
"""
    new_checks = """    const bool hold_last =
        cfg[\"hold_last\"]
        ? cfg[\"hold_last\"].as<bool>()
        : false;

    if(!hold_last)
    {
        const std::string timeout_target =
            cfg[\"timeout_target\"]
            ? cfg[\"timeout_target\"].as<std::string>()
            : \"Passive\";

        if(!FSMStringMap.right.count(timeout_target))
        {
            throw std::runtime_error(
                \"Unknown Mimic timeout target: \" + timeout_target
            );
        }

        this->registered_checks.emplace_back(
            std::make_pair(
                [&]()->bool
                {
                    return
                        (env->episode_length * env->step_dt)
                        > time_range_[1];
                },
                FSMStringMap.right.at(timeout_target)
            )
        );
    }

    bool enable_bad_orientation_check = true;
    if(cfg[\"enable_bad_orientation_check\"])
    {
        enable_bad_orientation_check =
            cfg[\"enable_bad_orientation_check\"].as<bool>();
    }

    if(enable_bad_orientation_check)
    {
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]()->bool
                {
                    return isaaclab::mdp::bad_orientation(
                        env.get(),
                        1.0
                    );
                },
                FSMStringMap.right.at(\"Passive\")
            )
        );
    }
"""
    if 'Unknown Mimic timeout target' not in s:
        if old_checks not in s:
            raise RuntimeError('State_Mimic.cpp timeout block does not match the classmate deploy version.')
        s = s.replace(old_checks, new_checks, 1)
        print('[State_Mimic.cpp] made timeout target configurable')
    else:
        print('[State_Mimic.cpp] configurable timeout already present')
    path.write_text(s)

def build_g9_assets(pkg: Path, base: Path):
    policy_src = pkg / 'deploy' / 'policy.onnx'
    motion_src = pkg / 'motion' / 'G9_rightfeiti_tail_direct_v2.npz'
    m = np.load(motion_src)
    q = np.asarray(m['joint_pos'], dtype=np.float32)
    dq = np.asarray(m['joint_vel'], dtype=np.float32)
    body_pos = np.asarray(m['body_pos_w'], dtype=np.float32)
    body_quat = np.asarray(m['body_quat_w'], dtype=np.float32)
    stance = np.asarray(m['deployment_stance_joint_pos'], dtype=np.float32)
    fps = float(np.asarray(m['fps']).reshape(-1)[0])
    if q.shape != (114, 29) or dq.shape != (114, 29):
        raise RuntimeError(f'Unexpected G9 q/dq shapes: {q.shape} / {dq.shape}')
    if abs(fps - 50.0) > 1e-6:
        raise RuntimeError(f'Expected G9 50 Hz, got {fps}')
    if np.max(np.abs(q[0] - stance)) > 1e-6 or np.max(np.abs(q[-1] - stance)) > 1e-6:
        raise RuntimeError('G9 motion does not start/end at deployment stance')
    q_sdk = np.zeros_like(q)
    dq_sdk = np.zeros_like(dq)
    for policy_i, sdk_i in enumerate(JOINT_MAP):
        q_sdk[:, sdk_i] = q[:, policy_i]
        dq_sdk[:, sdk_i] = dq[:, policy_i]
    root_pos = body_pos[:, 0, :]
    root_wxyz = body_quat[:, 0, :]
    root_xyzw = root_wxyz[:, [1,2,3,0]]
    csv_data = np.concatenate([root_pos, root_xyzw, q_sdk, dq_sdk], axis=1)
    if csv_data.shape != (114, 65):
        raise RuntimeError(f'Generated G9 motion shape {csv_data.shape}, expected (114,65)')
    out_dir = base / 'config' / 'policy' / 'mimic' / 'g9_tail_direct_v2'
    exported = out_dir / 'exported'
    params = out_dir / 'params'
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_src, exported / 'policy.onnx')
    motion_csv = params / 'g9_tail_direct_v2_qdq_50hz.csv'
    np.savetxt(motion_csv, csv_data, delimiter=',', fmt='%.9g')
    deploy = {
        'joint_ids_map': JOINT_MAP,
        'step_dt': 0.02,
        'stiffness': to_sdk(KP_POLICY),
        'damping': to_sdk(KD_POLICY),
        'default_joint_pos': Q_DEFAULT,
        'commands': {},
        'actions': {
            'JointPositionAction': {
                'clip': None,
                'joint_names': ['.*'],
                'scale': ACTION_SCALE,
                'offset': Q_DEFAULT,
                'joint_ids': None,
            }
        },
        'observations': {
            'obs': {
                'motion_command': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
                'motion_anchor_ori_b': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
                'base_ang_vel': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
                'joint_pos_rel': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
                'joint_vel_rel': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
                'last_action': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1},
            },
            'time_step': {
                'wbt_time_step': {'params': {}, 'clip': None, 'scale': None, 'history_length': 1}
            },
        },
    }
    (params / 'deploy.yaml').write_text(yaml.safe_dump(deploy, sort_keys=False, default_flow_style=False))
    return q_sdk, motion_csv, exported / 'policy.onnx', fps

def patch_config(config_path: Path, q_sdk):
    cfg = yaml.safe_load(config_path.read_text())
    fsm = cfg['FSM']
    enabled = fsm['_']
    boxing_q = np.asarray(fsm['BoxingFixStand']['qs'][-1], dtype=np.float64)
    g9_q = np.asarray(q_sdk[0], dtype=np.float64)
    stance_err = float(np.max(np.abs(boxing_q - g9_q)))
    existing_ids = [int(v['id']) for v in enabled.values() if isinstance(v, dict) and 'id' in v]
    state_id = int(enabled['G9Kick']['id']) if 'G9Kick' in enabled else max(existing_ids) + 1
    enabled['G9Kick'] = {'id': state_id, 'type': 'Mimic'}
    fsm['BoxingFixStand'].setdefault('keyboard_transitions', {})
    fsm['BoxingFixStand']['keyboard_transitions']['G9Kick'] = '4'
    fsm['BoxingFixStand'].setdefault('transitions', {})
    fsm['BoxingFixStand']['transitions']['G9Kick'] = 'RB + B.on_pressed'
    fsm['G9Kick'] = {
        'transitions': {
            'Passive': 'LT + B.on_pressed',
            'BoxingFixStand': 'LT + up.on_pressed',
        },
        'keyboard_transitions': {
            'Passive': '0',
            'BoxingFixStand': '1',
        },
        'fps': 50,
        'motion_file': 'config/policy/mimic/g9_tail_direct_v2/params/g9_tail_direct_v2_qdq_50hz.csv',
        'policy_dir': 'config/policy/mimic/g9_tail_direct_v2',
        'time_start': 0.0,
        'hold_last': False,
        'timeout_target': 'BoxingFixStand',
        'enable_bad_orientation_check': False,
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return state_id, stance_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deploy-root', default='/root/classmate_deploy/deploy')
    ap.add_argument('--package-root', default='/root/gpufree-data/G9_tail_direct_v2_handoff_classmate')
    args = ap.parse_args()
    deploy_root = Path(args.deploy_root).resolve()
    pkg = Path(args.package_root).resolve()
    base = deploy_root / 'robots' / 'g1_29dof'
    config_path = base / 'config' / 'config.yaml'
    mimic_h = base / 'include' / 'State_Mimic.h'
    mimic_cpp = base / 'src' / 'State_Mimic.cpp'
    required = [deploy_root / 'include' / 'FSM' / 'FSMState.h', config_path, mimic_h, mimic_cpp, pkg / 'deploy' / 'policy.onnx', pkg / 'motion' / 'G9_rightfeiti_tail_direct_v2.npz']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit('ERROR: missing required files:\n' + '\n'.join(missing))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup_dir = deploy_root.parent / f'classmate_deploy_before_g9_{stamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in [config_path, mimic_h, mimic_cpp]:
        backup_file(p, backup_dir)
    patch_state_mimic_h(mimic_h)
    patch_state_mimic_cpp(mimic_cpp)
    q_sdk, motion_csv, policy_out, fps = build_g9_assets(pkg, base)
    state_id, stance_err = patch_config(config_path, q_sdk)
    print('\n=== G9 INSTALLED INTO CLASSMATE DEPLOY ===')
    print('deploy root :', deploy_root)
    print('G9 state id :', state_id)
    print('policy      :', policy_out)
    print('motion      :', motion_csv)
    print('motion shape: (114, 65)')
    print('fps         :', fps)
    print('BoxingFixStand vs G9 frame0 max error:', stance_err)
    print('backup      :', backup_dir)
    print('\nKeyboard:')
    print('  0 = Passive')
    print('  1 = BoxingFixStand')
    print('  2 = Boxing_Velocity')
    print('  3 = Recovery')
    print('  4 = G9Kick (from BoxingFixStand)')
    print('\nG9 automatically returns to BoxingFixStand after the motion ends.')
    print('\nNext:')
    print('  cd', base)
    print('  cmake -S . -B build')
    print('  cmake --build build -j4')

if __name__ == '__main__':
    main()
