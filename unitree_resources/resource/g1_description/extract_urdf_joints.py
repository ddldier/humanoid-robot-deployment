#!/usr/bin/env python3
"""
从 URDF 文件提取活动关节（revolute/prismatic/continuous）的声明顺序
输出为 JSON/txt，供 mstudio CSV 转换脚本使用

用法:
    python extract_urdf_joints.py --urdf g1_29dof.urdf --out g1_joints.json
"""
import xml.etree.ElementTree as ET
import json
import argparse


def extract_active_joints(urdf_path: str):
    """提取 URDF 中所有活动关节（非 fixed）的 name 和 type，按声明顺序"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        name = joint.get("name", "")
        if jtype in ("revolute", "prismatic", "continuous"):
            parent = joint.find("parent").get("link", "") if joint.find("parent") is not None else ""
            child = joint.find("child").get("link", "") if joint.find("child") is not None else ""
            joints.append({
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
            })
    return joints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 URDF 关节顺序")
    parser.add_argument("--urdf", required=True, help="输入 URDF 文件路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args()

    joints = extract_active_joints(args.urdf)

    # 输出 JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(joints, f, ensure_ascii=False, indent=2)

    # 同时打印纯名称列表（方便复制到转换脚本）
    names = [j["name"] for j in joints]
    print(f"提取到 {len(names)} 个活动关节:")
    for i, name in enumerate(names):
        print(f"  [{i:2d}] {name}")

    print(f"\n已保存到: {args.out}")
    print("\n请将以下列表复制到转换脚本的 JOINT_ORDER 变量中:")
    print("JOINT_ORDER = [")
    for name in names:
        print(f'    "{name}",')
    print("]")
