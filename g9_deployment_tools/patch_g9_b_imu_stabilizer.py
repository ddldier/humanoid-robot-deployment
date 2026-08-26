from pathlib import Path
import re, shutil, time

CPP = Path('/root/g1_sdk2_sim2sim/unitree_rl_lab/deploy/robots/g1_29dof/src/State_Mimic.cpp')
if not CPP.exists():
    raise SystemExit(f'missing {CPP}')

s = CPP.read_text()
stamp = time.strftime('%Y%m%d_%H%M%S')
backup = CPP.with_name(f'State_Mimic.cpp.before_imu_B_{stamp}')
shutil.copy2(CPP, backup)

# includes
if '#include <cmath>' not in s:
    s = s.replace('#include <array>\n', '#include <array>\n#include <cmath>\n#include <utility>\n', 1)

# Replace current G9 globals block down to torso helper.
pat = re.compile(r'// G9 three-stage control: g -> b -> k\n.*?(?=\n\nEigen::Quaternionf torso_quat_w)', re.S)
new_globals = r'''// G9 three-stage control: g -> b -> k
// 0: DEFAULT_POSE = exact original G9 asymmetric frame0
// 1: BALANCE_HOLD = exact G9 nominal pose + small IMU ankle/hip corrections
// 2: KICK         = original G9 policy + advancing reference
static std::atomic<int> g9_control_mode{2};
static std::atomic<bool> g9_state_active{false};
static std::atomic<bool> g9_kick_ready{false};
static std::array<float, 29> g9_entry_q{};
static std::array<float, 29> g9_frame0_q{};
static std::array<float, 29> g9_balance_q{};
static double g9_entry_t0 = 0.0;
static double g9_balance_last_log_t = 0.0;
static constexpr double G9_ENTRY_BLEND_SEC = 3.0;

// B-stage keeps the user's exact G9 frame0 as the nominal q.
// These feedback terms are deliberately small: they only move hip/ankle joints
// enough to keep the floating base near the reference roll/pitch.
static float g9_target_roll = 0.0f;
static float g9_target_pitch = 0.0f;

static constexpr float G9_ROLL_KP = 0.90f;
static constexpr float G9_ROLL_KD = 0.12f;
static constexpr float G9_PITCH_KP = 1.10f;
static constexpr float G9_PITCH_KD = 0.14f;
static constexpr float G9_ROLL_MAX = 0.10f;   // rad joint correction
static constexpr float G9_PITCH_MAX = 0.12f;  // rad joint correction

// If a correction is observed to accelerate the fall instead of arresting it,
// flip only the corresponding sign and rebuild.
static constexpr float G9_ROLL_SIGN = 1.0f;
static constexpr float G9_PITCH_SIGN = 1.0f;

static constexpr std::array<float, 12> G9_BALANCE_KP = {
    100.0f, 100.0f, 100.0f, 150.0f, 55.0f, 55.0f,
    100.0f, 100.0f, 100.0f, 150.0f, 55.0f, 55.0f
};
static constexpr std::array<float, 12> G9_BALANCE_KD = {
    3.0f, 3.0f, 2.0f, 5.0f, 3.0f, 3.0f,
    3.0f, 3.0f, 2.0f, 5.0f, 3.0f, 3.0f
};

static std::pair<float, float> g9_roll_pitch_from_quat(const Eigen::Quaternionf & q)
{
    const auto R = q.normalized().toRotationMatrix();
    const float roll = std::atan2(R(2, 1), R(2, 2));
    const float pitch = std::asin(std::clamp(-R(2, 0), -1.0f, 1.0f));
    return {roll, pitch};
}
'''
if not pat.search(s):
    raise SystemExit('globals block not found')
s = pat.sub(new_globals, s, count=1)

# Replace G9 enter frame0/balance target section.
pat = re.compile(r'''        motion_->update\(time_range_\[0\]\);\n        auto frame0_q = motion_->joint_pos\(\);\n        for\(int i = 0; i < 29 && i < frame0_q.size\(\); \+\+i\)\n        \{.*?        spdlog::info\(\n            "\[G9\] g-stage DEFAULT_POSE;.*?\n        \);''', re.S)
new_enter = r'''        motion_->update(time_range_[0]);
        auto frame0_q = motion_->joint_pos();
        for(int i = 0; i < 29 && i < frame0_q.size(); ++i)
        {
            g9_frame0_q[i] = frame0_q[i];
            g9_balance_q[i] = frame0_q[i];
        }

        // Preserve the reference torso tilt encoded in the motion itself.
        // B will regulate around this roll/pitch, not force an arbitrary upright pose.
        auto target_torso_quat = anchor_quat_w(motion_);
        auto target_rp = g9_roll_pitch_from_quat(target_torso_quat);
        g9_target_roll = target_rp.first;
        g9_target_pitch = target_rp.second;
        g9_balance_last_log_t = 0.0;
        g9_kick_ready.store(false);

        spdlog::info(
            "[G9] g-stage DEFAULT_POSE; b=IMU balance around SAME pose; k=kick; target_rp=({:.3f},{:.3f})",
            g9_target_roll,
            g9_target_pitch
        );'''
if not pat.search(s):
    raise SystemExit('enter G9 frame0 block not found')
s = pat.sub(new_enter, s, count=1)

# Replace whole run() with active-B version.
pat = re.compile(r'void State_Mimic::run\(\)\n\{.*?\n\}', re.S)
m = pat.search(s)
if not m:
    raise SystemExit('run() not found')
new_run = r'''void State_Mimic::run()
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

                for(int i = 0; i < 12; ++i)
                {
                    lowcmd->msg_.motor_cmd()[i].kp() = G9_BALANCE_KP[i];
                    lowcmd->msg_.motor_cmd()[i].kd() = G9_BALANCE_KD[i];
                }

                spdlog::info(
                    "[G9] b pressed -> IMU_BALANCE on exact G9 nominal pose"
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
        // b: same nominal 29D G9 pose, with SMALL active roll/pitch
        // corrections only at hips and ankles.
        // ------------------------------------------------------
        if(mode == 1)
        {
            if(key == "k")
            {
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

                for(int i = 0; i < 29; ++i)
                    lowcmd->msg_.motor_cmd()[i].q() = g9_balance_q[i];
                return;
            }

            // Policy thread is sleeping in mode 1, so it is safe to refresh
            // articulation state here without racing env->step().
            env->robot->update();

            const auto torso_q = torso_quat_w(env.get());
            const auto rp = g9_roll_pitch_from_quat(torso_q);
            const float roll = rp.first;
            const float pitch = rp.second;

            const auto gyro = isaaclab::mdp::base_ang_vel(
                env.get(), YAML::Node()
            );
            const float wx = gyro.size() > 0 ? gyro[0] : 0.0f;
            const float wy = gyro.size() > 1 ? gyro[1] : 0.0f;

            const float e_roll = g9_target_roll - roll;
            const float e_pitch = g9_target_pitch - pitch;

            const float u_roll = G9_ROLL_SIGN * std::clamp(
                G9_ROLL_KP * e_roll - G9_ROLL_KD * wx,
                -G9_ROLL_MAX, G9_ROLL_MAX
            );
            const float u_pitch = G9_PITCH_SIGN * std::clamp(
                G9_PITCH_KP * e_pitch - G9_PITCH_KD * wy,
                -G9_PITCH_MAX, G9_PITCH_MAX
            );

            // Start from the EXACT user/reference pose every cycle.
            std::array<float, 29> q_cmd = g9_frame0_q;

            // SDK order:
            // 0 L hip pitch, 1 L hip roll, 4 L ankle pitch, 5 L ankle roll
            // 6 R hip pitch, 7 R hip roll,10 R ankle pitch,11 R ankle roll
            // Ankle strategy provides most of the correction; hips compensate
            // partially so the visible upper-body pose changes as little as possible.
            q_cmd[4]  += u_pitch;
            q_cmd[10] += u_pitch;
            q_cmd[0]  -= 0.35f * u_pitch;
            q_cmd[6]  -= 0.35f * u_pitch;

            q_cmd[5]  += u_roll;
            q_cmd[11] += u_roll;
            q_cmd[1]  -= 0.30f * u_roll;
            q_cmd[7]  -= 0.30f * u_roll;

            // Keep ankle commands inside the published G1 joint ranges.
            q_cmd[4]  = std::clamp(q_cmd[4],  -0.85f, 0.50f);
            q_cmd[10] = std::clamp(q_cmd[10], -0.85f, 0.50f);
            q_cmd[5]  = std::clamp(q_cmd[5],  -0.25f, 0.25f);
            q_cmd[11] = std::clamp(q_cmd[11], -0.25f, 0.25f);

            g9_balance_q = q_cmd;
            for(int i = 0; i < 29; ++i)
                lowcmd->msg_.motor_cmd()[i].q() = q_cmd[i];

            const double now =
                (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
            if(now - g9_balance_last_log_t > 0.25)
            {
                g9_balance_last_log_t = now;
                spdlog::info(
                    "[G9BAL] target_rp=({:.3f},{:.3f}) rp=({:.3f},{:.3f}) "
                    "err=({:.3f},{:.3f}) gyro=({:.3f},{:.3f}) corr=({:.3f},{:.3f})",
                    g9_target_roll, g9_target_pitch,
                    roll, pitch,
                    e_roll, e_pitch,
                    wx, wy,
                    u_roll, u_pitch
                );
            }
            return;
        }

        // ------------------------------------------------------
        // k: wait for a fresh G9 action, then policy owns q.
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

    // Punch / other Mimic states stay unchanged.
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}'''
s = s[:m.start()] + new_run + s[m.end():]

CPP.write_text(s)
print('patched', CPP)
print('backup', backup)
