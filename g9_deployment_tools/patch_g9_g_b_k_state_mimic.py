from pathlib import Path
import re, shutil, time
WS=Path('/root/g1_sdk2_sim2sim')
BASE=WS/'unitree_rl_lab/deploy/robots/g1_29dof'
CPP=BASE/'src/State_Mimic.cpp'
FSM=WS/'unitree_rl_lab/deploy/include/FSM/FSMState.h'

s=CPP.read_text()
if '#include <atomic>' not in s:
    anchor='#include "isaaclab/envs/mdp/actions/joint_actions.h"\n'
    s=s.replace(anchor, anchor+'#include <atomic>\n#include <array>\n',1)

anchor='std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;\n'
if '// G9 three-stage control: g -> b -> k' not in s:
    block='''\n// G9 three-stage control: g -> b -> k\n// 0: DEFAULT_POSE, 1: BALANCE_HOLD, 2: KICK\nstatic std::atomic<int> g9_control_mode{2};\nstatic std::atomic<bool> g9_state_active{false};\nstatic std::array<float, 29> g9_entry_q{};\nstatic std::array<float, 29> g9_frame0_q{};\nstatic double g9_entry_t0 = 0.0;\nstatic constexpr double G9_ENTRY_BLEND_SEC = 3.0;\n\n'''
    s=s.replace(anchor,anchor+block,1)

old='''REGISTER_OBSERVATION(wbt_time_step)\n{\n    long frame =\n        std::max<long>(\n            0,\n            env->episode_length - 1\n        );\n\n    if(State_Mimic::motion)\n    {\n        frame =\n            std::min<long>(\n                frame,\n                State_Mimic::motion->num_frames - 1\n            );\n    }\n\n    return {\n        static_cast<float>(frame)\n    };\n}\n'''
new='''REGISTER_OBSERVATION(wbt_time_step)\n{\n    if(g9_state_active.load() && g9_control_mode.load() == 1)\n    {\n        return {0.0f};\n    }\n\n    long frame =\n        std::max<long>(\n            0,\n            env->episode_length - 1\n        );\n\n    if(State_Mimic::motion)\n    {\n        frame =\n            std::min<long>(\n                frame,\n                State_Mimic::motion->num_frames - 1\n            );\n    }\n\n    return {\n        static_cast<float>(frame)\n    };\n}\n'''
if new not in s:
    assert old in s, 'time block mismatch'
    s=s.replace(old,new,1)

anchor='''void State_Mimic::enter()\n{\n    // set gain\n'''
repl='''void State_Mimic::enter()\n{\n    const bool is_g9 = (getStateString() == "G9TailDirectV2");\n    g9_state_active.store(is_g9);\n\n    if(is_g9)\n    {\n        g9_control_mode.store(0);\n        g9_entry_t0 =\n            (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;\n\n        for(int i = 0; i < 29; ++i)\n        {\n            g9_entry_q[i] = lowstate->msg_.motor_state()[i].q();\n        }\n\n        motion_->update(time_range_[0]);\n        auto frame0_q = motion_->joint_pos();\n        for(int i = 0; i < 29 && i < frame0_q.size(); ++i)\n        {\n            g9_frame0_q[i] = frame0_q[i];\n        }\n        spdlog::info("[G9] g-stage DEFAULT_POSE; b=balance hold; k=kick");\n    }\n\n    // set gain\n'''
if repl not in s:
    assert anchor in s,'enter anchor mismatch'
    s=s.replace(anchor,repl,1)

old='''        while (policy_thread_running)\n        {\n            env->robot->update();\n            motion->update(env->episode_length * env->step_dt + time_range_[0]);\n            env->step();\n\n            // -------------------------------------------------\n'''
new='''        int local_g9_mode =\n            (getStateString() == "G9TailDirectV2") ? 0 : 2;\n\n        while (policy_thread_running)\n        {\n            const bool is_g9_thread =\n                (getStateString() == "G9TailDirectV2");\n\n            if(is_g9_thread)\n            {\n                const int mode = g9_control_mode.load();\n\n                if(mode == 0)\n                {\n                    std::this_thread::sleep_until(sleepTill);\n                    sleepTill += dt;\n                    continue;\n                }\n\n                if(mode != local_g9_mode)\n                {\n                    env->robot->update();\n                    motion->reset(env->robot->data, time_range_[0]);\n\n                    auto g9_ref_yaw =\n                        isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();\n                    auto g9_robot_yaw =\n                        isaaclab::yawQuaternion(torso_quat_w(env.get())).toRotationMatrix();\n                    init_quat = g9_robot_yaw * g9_ref_yaw.transpose();\n\n                    env->reset();\n                    local_g9_mode = mode;\n\n                    if(mode == 1)\n                        spdlog::info("[G9] b-stage BALANCE_HOLD: frame0 frozen");\n                    else\n                        spdlog::info("[G9] k-stage KICK: episode reset, reference released");\n                }\n\n                env->robot->update();\n\n                if(mode == 1)\n                    motion->update(time_range_[0]);\n                else\n                    motion->update(env->episode_length * env->step_dt + time_range_[0]);\n\n                env->step();\n            }\n            else\n            {\n                env->robot->update();\n                motion->update(env->episode_length * env->step_dt + time_range_[0]);\n                env->step();\n            }\n\n            // -------------------------------------------------\n'''
if new not in s:
    assert old in s,'loop anchor mismatch'
    s=s.replace(old,new,1)

pat=re.compile(r'void State_Mimic::run\(\)\n\{\n.*?\n\}',re.S)
m=pat.search(s)
assert m,'run missing'
newrun='''void State_Mimic::run()\n{\n    if(getStateString() == "G9TailDirectV2")\n    {\n        int mode = g9_control_mode.load();\n        const std::string key =\n            FSMState::keyboard ? FSMState::keyboard->key() : "";\n\n        if(mode == 0)\n        {\n            if(key == "b")\n            {\n                g9_control_mode.store(1);\n                mode = 1;\n                spdlog::info("[G9] b pressed -> BALANCE_HOLD");\n            }\n            else\n            {\n                const double now =\n                    (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;\n                const float alpha = std::clamp(\n                    static_cast<float>((now - g9_entry_t0) / G9_ENTRY_BLEND_SEC),\n                    0.0f, 1.0f);\n\n                for(int i = 0; i < 29; ++i)\n                {\n                    lowcmd->msg_.motor_cmd()[i].q() =\n                        (1.0f - alpha) * g9_entry_q[i] + alpha * g9_frame0_q[i];\n                }\n                return;\n            }\n        }\n\n        if(mode == 1 && key == "k")\n        {\n            g9_control_mode.store(2);\n            mode = 2;\n            spdlog::info("[G9] k pressed -> KICK");\n        }\n\n        auto action = env->action_manager->processed_actions();\n        const auto & ids = env->robot->data.joint_ids_map;\n\n        if(action.size() >= ids.size())\n        {\n            for(int i = 0; i < ids.size(); ++i)\n                lowcmd->msg_.motor_cmd()[ids[i]].q() = action[i];\n        }\n        else\n        {\n            for(int i = 0; i < 29; ++i)\n                lowcmd->msg_.motor_cmd()[i].q() = g9_frame0_q[i];\n        }\n        return;\n    }\n\n    auto action = env->action_manager->processed_actions();\n    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++)\n    {\n        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];\n    }\n}'''
s=s[:m.start()]+newrun+s[m.end():]
CPP.write_text(s)

f=FSM.read_text()
# replace previous G9 shortcut blocks before universal timeout block
f=re.sub(r'\n\s*// G9 action-set keyboard shortcuts.*?(?=\s*// register for all states)','\n',f,flags=re.S)
f=re.sub(r'\n\s*// G9TailDirectV2 terminal shortcut.*?(?=\s*// register for all states)','\n',f,flags=re.S)
f=re.sub(r'\n\s*// G9 three-stage terminal shortcut.*?(?=\s*// register for all states)','\n',f,flags=re.S)
block='''\n        // G9 three-stage terminal shortcut\n        // g : Passive/FixStand -> G9TailDirectV2\n        // b/k are handled inside State_Mimic.cpp\n        if(\n            (state_string == "Passive" || state_string == "FixStand")\n            && FSMStringMap.right.count("G9TailDirectV2")\n        )\n        {\n            registered_checks.emplace_back(\n                std::make_pair(\n                    []()->bool\n                    {\n                        return FSMState::keyboard\n                            && FSMState::keyboard->key() == "g";\n                    },\n                    FSMStringMap.right.at("G9TailDirectV2")\n                )\n            );\n        }\n\n'''
marker='        // register for all states'
assert marker in f
f=f.replace(marker,block+marker,1)
FSM.write_text(f)
