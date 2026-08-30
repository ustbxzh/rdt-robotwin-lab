#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_INFO_FILE="${SCRIPT_DIR}/robot/_robot_info.json"

python3 -c '
import sys, os, json

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]
robot_info_file = sys.argv[3]

robot_action_dim_info = json.load(
    open(robot_info_file, "r", encoding="utf-8")
)[env_cfg_type]

print(sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}" "${ROBOT_INFO_FILE}"
