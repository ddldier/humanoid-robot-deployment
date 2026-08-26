# G9 Deployment Tools

G9机器人部署和补丁工具集合，用于配置和优化G9机器人的控制系统。

## 工具列表

### 1. patch_g9_b_static_asym_balance.py
G9静态非对称平衡补丁脚本
- 为G9机器人添加静态平衡控制
- 修改State_Mimic.cpp以支持B按键触发的平衡保持模式

### 2. add_g9_to_cpp_action_set.py
将G9动作集添加到C++部署代码
- 集成G9动作到C++控制系统
- 配置动作集映射

### 3. patch_g9_b_imu_stabilizer.py
G9 IMU稳定器补丁
- 添加IMU传感器稳定功能
- 增强机器人姿态控制

### 4. install_g9_into_classmate_deploy.py
将G9集成到同学版本部署
- 安装G9配置到同学项目
- 自动化部署流程

### 5. patch_g9_g_b_k_state_mimic.py
G9三阶段控制补丁 (G -> B -> K)
- 实现三阶段状态转换控制
- G: 初始状态
- B: 平衡保持状态
- K: 运动执行状态

### 6. play_clean_target_push.py
播放清理目标推送脚本
- 用于测试和验证推送动作
- 清理和重置目标状态

## 使用方法

这些脚本需要在G1 SDK环境中运行，通常的使用流程：

```bash
# 1. 确保在正确的SDK环境中
cd /root/g1_sdk2_sim2sim/

# 2. 运行相应的补丁脚本
python3 patch_g9_b_static_asym_balance.py

# 3. 重新编译部署代码
# (具体编译步骤根据SDK文档)
```

## 注意事项

⚠️ **重要提醒**:
- 这些脚本会修改C++源代码
- 运行前会自动创建备份文件
- 确保在正确的SDK版本上使用
- 建议先在测试环境中验证

## 相关项目

- [G9_tail_direct_v2_handoff](https://github.com/ddldier/G9_tail_direct_v2_handoff) - G9飞踢sim2real部署
- [whole_body_tracking_custom](https://github.com/ddldier/whole_body_tracking_custom) - 全身跟踪自定义版本

## 技术栈

- Python 3
- Unitree G1 SDK
- C++ (State_Mimic系统)

## 许可证

MIT License
