from pathlib import Path
import re, shutil, time

CPP = Path('/root/g1_sdk2_sim2sim/unitree_rl_lab/deploy/robots/g1_29dof/src/State_Mimic.cpp')
if not CPP.exists():
    raise SystemExit(f'ERROR: missing {CPP}')

s = CPP.read_text()
required = [
    '// G9 three-stage control: g -> b -> k',
    'g9_control_mode',
    'g9_frame0_q',
    '[G9] b pressed -> BALANCE_HOLD',
]
for x in required:
    if x not in s:
        raise SystemExit(f'ERROR: current State_Mimic.cpp is not the expected g/b/k version; missing: {x}')

stamp = time.strftime('%Y%m%d_%H%M%S')
backup = CPP.with_name(f'State_Mimic.cpp.before_static_B_{stamp}')
shutil.copy2(CPP, backup)

# ----- globals -----
old = '''// G9 three-stage control: g -> b -> k
// 0: DEFAULT_POSE, 1: BALANCE_HOLD, 2: KICK
static std::atomic<int> g9_control_mode{2};
static std::atomic<bool> g9_state_active{false};
static std::array<float, 29> g9_entry_q{};
static std::array<float, 29> g9_frame0_q{};
static double g9_entry_t0 = 0.0;
static constexpr double G9_ENTRY_BLEND_SEC = 3.0;
'''
new = '''// G9 three-stage control: g -> b -> k
// 0: DEFAULT_POSE  = exact original G9 asymmetric frame0
// 1: BALANCE_HOLD  = static asymmetric PD pose (NO policy)
// 2: KICK          = original G9 policy + advancing reference
static std::atomic<int> g9_control_mode{2};
static std::atomic<bool> g9_state_active{false};
static std::atomic<bool> g9_kick_ready{false};
static std::array<float, 29> g9_entry_q{};
static std::array<float, 29> g9_frame0_q{};
static std::array<float, 29> g9_balance_q{};
static double g9_entry_t0 = 0.0;
static double g9_balance_t0 = 0.0;
static constexpr double G9_ENTRY_BLEND_SEC = 3.0;
static constexpr double G9_BALANCE_BLEND_SEC = 1.5;

// 0.0 = exact G9 frame0, 1.0 = exact proven Punch FixStand legs.
// Start at 0.45: still clearly asymmetric, but moves the support geometry
// toward a pose already known to stand in this MuJoCo setup.
static constexpr float G9_BALANCE_TO_STABLE = 0.45f;

// First 12 SDK-order leg joints from the already-proven Punch FixStand.
static constexpr std::array<float, 12> G9_STABLE_LEG_ANCHOR = {
    -0.998841822f,  0.269701540f,  0.172225922f,
     0.905094087f, -0.256087214f, -0.062892899f,
    -0.999644101f, -0.268931359f, -0.168453321f,
     0.902647316f, -0.253162414f,  0.064963907f
};

static constexpr std::array<float, 12> G9_BALANCE_KP = {
    100.0f, 100.0f, 100.0f, 150.0f, 40.0f, 40.0f,
    100.0f, 100.0f, 100.0f, 150.0f, 40.0f, 40.0f
};
static constexpr std::array<float, 12> G9_BALANCE_KD = {
    2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 4.0f, 2.0f, 2.0f, 2.0f
};
'''
if old not in s:
    raise SystemExit('ERROR: globals block mismatch; upload current State_Mimic.cpp if you edited it manually.')
s = s.replace(old, new, 1)

# Correct KD to exact FixStand ordering: first 12 are 2,2,2,4,2,2 repeated.
s = s.replace('''static constexpr std::array<float, 12> G9_BALANCE_KD = {
    2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 4.0f, 2.0f, 2.0f, 2.0f
};''', '''static constexpr std::array<float, 12> G9_BALANCE_KD = {
    2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 2.0f
};''')

# ----- enter(): compute the static asymmetric B target -----
old = '''        for(int i = 0; i < 29 && i < frame0_q.size(); ++i)
        {
            g9_frame0_q[i] = frame0_q[i];
        }
        spdlog::info("[G9] g-stage DEFAULT_POSE; b=balance hold; k=kick");
'''
new = '''        for(int i = 0; i < 29 && i < frame0_q.size(); ++i)
        {
            g9_frame0_q[i] = frame0_q[i];
            g9_balance_q[i] = frame0_q[i];
        }

        // Keep the complete G9 upper body exactly unchanged.
        // Only the 12 leg joints are pulled part-way toward the already
        // verified stable FixStand.  Therefore B remains asymmetric.
        for(int i = 0; i < 12; ++i)
        {
            g9_balance_q[i] =
                (1.0f - G9_BALANCE_TO_STABLE) * g9_frame0_q[i]
                + G9_BALANCE_TO_STABLE * G9_STABLE_LEG_ANCHOR[i];
        }

        g9_kick_ready.store(false);
        spdlog::info(
            "[G9] g-stage DEFAULT_POSE; b=STATIC asymmetric balance; k=kick; blend={:.2f}",
            G9_BALANCE_TO_STABLE
        );
'''
if old not in s:
    raise SystemExit('ERROR: enter frame0 block mismatch')
s = s.replace(old, new, 1)

# ----- policy thread: modes 0 and 1 consume NO policy time -----
old = '''                if(mode == 0)
                {
                    std::this_thread::sleep_until(sleepTill);
                    sleepTill += dt;
                    continue;
                }

                if(mode != local_g9_mode)
                {
                    env->robot->update();
                    motion->reset(env->robot->data, time_range_[0]);

                    auto g9_ref_yaw =
                        isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
                    auto g9_robot_yaw =
                        isaaclab::yawQuaternion(torso_quat_w(env.get())).toRotationMatrix();
                    init_quat = g9_robot_yaw * g9_ref_yaw.transpose();

                    env->reset();
                    local_g9_mode = mode;

                    if(mode == 1)
                        spdlog::info("[G9] b-stage BALANCE_HOLD: frame0 frozen");
                    else
                        spdlog::info("[G9] k-stage KICK: episode reset, reference released");
                }

                env->robot->update();

                if(mode == 1)
                    motion->update(time_range_[0]);
                else
                    motion->update(env->episode_length * env->step_dt + time_range_[0]);

                env->step();
'''
new = '''                // g and b are pure PD states.  Do not call env->step(),
                // otherwise last_action/time keep changing before the kick.
                if(mode != 2)
                {
                    std::this_thread::sleep_until(sleepTill);
                    sleepTill += dt;
                    continue;
                }

                if(mode != local_g9_mode)
                {
                    g9_kick_ready.store(false);

                    env->robot->update();
                    motion->reset(env->robot->data, time_range_[0]);

                    auto g9_ref_yaw =
                        isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
                    auto g9_robot_yaw =
                        isaaclab::yawQuaternion(torso_quat_w(env.get())).toRotationMatrix();
                    init_quat = g9_robot_yaw * g9_ref_yaw.transpose();

                    env->reset();
                    local_g9_mode = mode;
                    spdlog::info("[G9] k-stage KICK: episode reset, reference released");
                }

                env->robot->update();
                motion->update(env->episode_length * env->step_dt + time_range_[0]);
                env->step();
                g9_kick_ready.store(true);
'''
if old not in s:
    raise SystemExit('ERROR: policy-thread G9 mode block mismatch')
s = s.replace(old, new, 1)

# ----- replace run() -----
pat = re.compile(r'void State_Mimic::run\(\)\n\{\n.*?\n\}', re.S)
m = pat.search(s)
if not m:
    raise SystemExit('ERROR: State_Mimic::run() not found')

newrun = r'''void State_Mimic::run()
{
    if(getStateString() == "G9TailDirectV2")
    {
        int mode = g9_control_mode.load();
        const std::string key =
            FSMState::keyboard ? FSMState::keyboard->key() : "";

        // ------------------------------------------------------
        // g: exact original asymmetric G9 frame0.
        // ------------------------------------------------------
        if(mode == 0)
        {
            if(key == "b")
            {
                g9_control_mode.store(1);
                mode = 1;
                g9_balance_t0 =
                    (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;

                // Use the gains of the FixStand that is already known
                // to stand in this exact MuJoCo model.
                for(int i = 0; i < 12; ++i)
                {
                    lowcmd->msg_.motor_cmd()[i].kp() = G9_BALANCE_KP[i];
                    lowcmd->msg_.motor_cmd()[i].kd() = G9_BALANCE_KD[i];
                }

                spdlog::info(
                    "[G9] b pressed -> STATIC_BALANCE, leg blend={:.2f}",
                    G9_BALANCE_TO_STABLE
                );
            }
            else
            {
                const double now =
                    (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
                const float alpha = std::clamp(
                    static_cast<float>((now - g9_entry_t0) / G9_ENTRY_BLEND_SEC),
                    0.0f, 1.0f);

                for(int i = 0; i < 29; ++i)
                {
                    lowcmd->msg_.motor_cmd()[i].q() =
                        (1.0f - alpha) * g9_entry_q[i]
                        + alpha * g9_frame0_q[i];
                }
                return;
            }
        }

        // ------------------------------------------------------
        // b: no ONNX inference.  Smoothly move from exact G9 frame0
        // to the static asymmetric balance target and hold it.
        // ------------------------------------------------------
        if(mode == 1)
        {
            if(key == "k")
            {
                // Restore the exact G9 deployment gains before the
                // policy becomes responsible for the robot.
                for(int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
                {
                    lowcmd->msg_.motor_cmd()[i].kp() =
                        env->robot->data.joint_stiffness[i];
                    lowcmd->msg_.motor_cmd()[i].kd() =
                        env->robot->data.joint_damping[i];
                }

                g9_kick_ready.store(false);
                g9_control_mode.store(2);
                spdlog::info("[G9] k pressed -> KICK requested");

                // Hold B until the policy thread has completed reset
                // and produced its first valid action.
                for(int i = 0; i < 29; ++i)
                    lowcmd->msg_.motor_cmd()[i].q() = g9_balance_q[i];
                return;
            }

            const double now =
                (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
            const float alpha = std::clamp(
                static_cast<float>((now - g9_balance_t0) / G9_BALANCE_BLEND_SEC),
                0.0f, 1.0f);

            for(int i = 0; i < 29; ++i)
            {
                lowcmd->msg_.motor_cmd()[i].q() =
                    (1.0f - alpha) * g9_frame0_q[i]
                    + alpha * g9_balance_q[i];
            }
            return;
        }

        // ------------------------------------------------------
        // k: wait for a genuinely fresh G9 action, then policy owns q.
        // ------------------------------------------------------
        if(!g9_kick_ready.load())
        {
            for(int i = 0; i < 29; ++i)
                lowcmd->msg_.motor_cmd()[i].q() = g9_balance_q[i];
            return;
        }

        auto action = env->action_manager->processed_actions();
        const auto & ids = env->robot->data.joint_ids_map;

        if(action.size() >= ids.size())
        {
            for(int i = 0; i < ids.size(); ++i)
                lowcmd->msg_.motor_cmd()[ids[i]].q() = action[i];
        }
        else
        {
            for(int i = 0; i < 29; ++i)
                lowcmd->msg_.motor_cmd()[i].q() = g9_balance_q[i];
        }
        return;
    }

    // Punch / other Mimic states stay exactly as before.
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}'''
s = s[:m.start()] + newrun + s[m.end():]

CPP.write_text(s)
print('=== G9 B CHANGED TO STATIC ASYMMETRIC PD BALANCE ===')
print('backup:', backup)
print('file  :', CPP)
print('G9_BALANCE_TO_STABLE = 0.45')
print('Rebuild required.')
