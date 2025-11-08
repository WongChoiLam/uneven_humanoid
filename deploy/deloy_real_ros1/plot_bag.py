#!/usr/bin/env python3

import argparse
import math
import os
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ROS1 bag
import rosbag
import rospy

def parse_args():
    p = argparse.ArgumentParser(description="Plot cmd/state trajectories from rosbag")
    p.add_argument("--bag", required=True, help="Path to .bag file")
    p.add_argument("--state-topic", default="/state", help="Topic for RobotState (e.g., /robot/state)")
    p.add_argument("--cmd-topic", default="/cmd", help="Topic for RobotCommand (e.g., /robot/cmd)")
    p.add_argument("--imu-topic", default=None, help="Topic for IMU if published separately; if None, read from state_msg.imu")
    p.add_argument("--joints", default=None,
                   help="Joint index selection. Examples: '0:12' (slice), 'all', or '0,3,7,12'. Default: all joints found")
    p.add_argument("--align", choices=["nearest", "linear"], default="nearest",
                   help="Time alignment strategy when plotting cmd vs state")
    p.add_argument("--downsample", type=int, default=1, help="Plot every Nth sample for speed")
    p.add_argument("--t-start", type=float, default=None, help="Start time (sec) relative to bag start")
    p.add_argument("--t-end", type=float, default=None, help="End time (sec) relative to bag start")
    p.add_argument("--show-kpkd", action="store_true", help="Also plot kp/kd from cmd")
    p.add_argument("--no-imu", action="store_true", help="Do not plot IMU")
    p.add_argument("--title", default=None, help="Figure suptitle")
    return p.parse_args()


def detect_topics(bag: rosbag.Bag) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try to detect default topics for state/cmd/imu from bag connection info.
    Returns (state_topic, cmd_topic, imu_topic). Any of them can be None if not found.
    """
    state_guess = None
    cmd_guess = None
    imu_guess = None

    for c in bag.get_type_and_topic_info()[1].items():
        topic, info = c
        msg_type = info.msg_type  # e.g., robot_msgs/RobotState
        t = topic

        if msg_type.endswith("RobotState") or "RobotState" in msg_type or t.endswith("/state"):
            if state_guess is None:
                state_guess = t
        if msg_type.endswith("RobotCommand") or "RobotCommand" in msg_type or t.endswith("/cmd"):
            if cmd_guess is None:
                cmd_guess = t
        if msg_type.endswith("IMU") or "IMU" in msg_type or "imu" in t.lower():
            if imu_guess is None:
                imu_guess = t

    return state_guess, cmd_guess, imu_guess


def to_numpy_time(stamps: List[rospy.Time], t0: Optional[rospy.Time] = None) -> np.ndarray:
    if len(stamps) == 0:
        return np.array([])
    if t0 is None:
        t0 = stamps[0]
    return np.array([ (s - t0).to_sec() for s in stamps ], dtype=float)


def parse_joint_selection(joint_count: int, sel: Optional[str]) -> List[int]:
    if sel is None or sel.lower() == "all":
        return list(range(joint_count))
    sel = sel.strip()
    if ":" in sel and "," not in sel:
        a, b = sel.split(":")
        a = int(a) if a != "" else 0
        b = int(b) if b != "" else joint_count
        return list(range(a, b))
    if "," in sel:
        idxs = [int(s) for s in sel.split(",") if s.strip() != ""]
        return idxs
    # single index
    return [int(sel)]


def extract_state(bag: rosbag.Bag, topic: str, t0: Optional[rospy.Time], t_start: Optional[float], t_end: Optional[float]):
    t_list = []
    q_list = []
    dq_list = []
    ddq_list = []
    tau_est_list = []
    imu_quat = []
    imu_gyro = []
    imu_acc = []

    joint_count = None

    for _, msg, t in bag.read_messages(topics=[topic]):
        if t0 is None:
            t0 = t
        t_rel = (t - t0).to_sec()
        if t_start is not None and t_rel < t_start:
            continue
        if t_end is not None and t_rel > t_end:
            continue

        t_list.append(t)

        # motor_state is a list/vector
        states = msg.motor_state
        if joint_count is None:
            joint_count = len(states)
        # Initialize per-sample arrays
        q = np.full(joint_count, np.nan, dtype=float)
        dq = np.full(joint_count, np.nan, dtype=float)
        ddq = np.full(joint_count, np.nan, dtype=float)
        tau_est = np.full(joint_count, np.nan, dtype=float)

        for i in range(min(joint_count, len(states))):
            ms = states[i]
            # Field names as inferred
            q[i] = getattr(ms, "q", np.nan)
            dq[i] = getattr(ms, "dq", np.nan)
            ddq[i] = getattr(ms, "ddq", np.nan)
            tau_est[i] = getattr(ms, "tau_est", np.nan)

        q_list.append(q)
        dq_list.append(dq)
        ddq_list.append(ddq)
        tau_est_list.append(tau_est)

        # IMU nested in state if available
        if hasattr(msg, "imu"):
            imu = msg.imu
            if hasattr(imu, "quaternion"):
                imu_quat.append(np.array(list(imu.quaternion), dtype=float))
            if hasattr(imu, "gyroscope"):
                imu_gyro.append(np.array(list(imu.gyroscope), dtype=float))
            if hasattr(imu, "accelerometer"):
                imu_acc.append(np.array(list(imu.accelerometer), dtype=float))

    if joint_count is None:
        return {}, None

    data = {
        "t": to_numpy_time(t_list, t0),
        "q": np.vstack(q_list) if len(q_list) else np.empty((0, joint_count)),
        "dq": np.vstack(dq_list) if len(dq_list) else np.empty((0, joint_count)),
        "ddq": np.vstack(ddq_list) if len(ddq_list) else np.empty((0, joint_count)),
        "tau_est": np.vstack(tau_est_list) if len(tau_est_list) else np.empty((0, joint_count)),
        "imu_quat": np.vstack(imu_quat) if len(imu_quat) else np.empty((0, 4)),
        "imu_gyro": np.vstack(imu_gyro) if len(imu_gyro) else np.empty((0, 3)),
        "imu_acc": np.vstack(imu_acc) if len(imu_acc) else np.empty((0, 3)),
        "joint_count": joint_count,
    }
    return data, t0


def extract_cmd(bag: rosbag.Bag, topic: str, t0: Optional[rospy.Time], t_start: Optional[float], t_end: Optional[float]):
    t_list = []
    q_list = []
    dq_list = []
    tau_list = []
    kp_list = []
    kd_list = []

    joint_count = None

    for _, msg, t in bag.read_messages(topics=[topic]):
        if t0 is None:
            t0 = t
        t_rel = (t - t0).to_sec()
        if t_start is not None and t_rel < t_start:
            continue
        if t_end is not None and t_rel > t_end:
            continue

        t_list.append(t)

        cmds = msg.motor_command
        if joint_count is None:
            joint_count = len(cmds)

        q = np.full(joint_count, np.nan, dtype=float)
        dq = np.full(joint_count, np.nan, dtype=float)
        tau = np.full(joint_count, np.nan, dtype=float)
        kp = np.full(joint_count, np.nan, dtype=float)
        kd = np.full(joint_count, np.nan, dtype=float)

        for i in range(min(joint_count, len(cmds))):
            c = cmds[i]
            q[i] = getattr(c, "q", np.nan)
            dq[i] = getattr(c, "dq", np.nan)
            tau[i] = getattr(c, "tau", np.nan)
            kp[i] = getattr(c, "kp", np.nan)
            kd[i] = getattr(c, "kd", np.nan)

        q_list.append(q)
        dq_list.append(dq)
        tau_list.append(tau)
        kp_list.append(kp)
        kd_list.append(kd)

    if joint_count is None:
        return {}, t0

    data = {
        "t": to_numpy_time(t_list, t0),
        "q": np.vstack(q_list) if len(q_list) else np.empty((0, joint_count)),
        "dq": np.vstack(dq_list) if len(dq_list) else np.empty((0, joint_count)),
        "tau": np.vstack(tau_list) if len(tau_list) else np.empty((0, joint_count)),
        "kp": np.vstack(kp_list) if len(kp_list) else np.empty((0, joint_count)),
        "kd": np.vstack(kd_list) if len(kd_list) else np.empty((0, joint_count)),
        "joint_count": joint_count,
    }
    return data, t0


def align_series(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray, mode: str = "nearest") -> np.ndarray:
    """
    Resample y_src(t_src) to y_dst(t_dst).
    - mode 'nearest': pick nearest sample
    - mode 'linear': linear interpolation per column (NaNs are propagated)
    """
    if y_src.size == 0 or t_src.size == 0 or t_dst.size == 0:
        return np.empty((len(t_dst), y_src.shape[1] if y_src.ndim == 2 else 0))

    if mode == "nearest":
        idx = np.searchsorted(t_src, t_dst, side="left")
        idx = np.clip(idx, 0, len(t_src) - 1)
        # Check if closer to previous sample
        prev = np.clip(idx - 1, 0, len(t_src) - 1)
        choose_prev = (np.abs(t_dst - t_src[prev]) <= np.abs(t_dst - t_src[idx]))
        final_idx = np.where(choose_prev, prev, idx)
        return y_src[final_idx, :]

    # linear
    y_out = np.empty((len(t_dst), y_src.shape[1]))
    for j in range(y_src.shape[1]):
        col = y_src[:, j]
        # Handle NaN by simple forwarding: mask out NaNs for interpolation
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            y_out[:, j] = np.nan
            continue
        y_out[:, j] = np.interp(t_dst, t_src[valid], col[valid], left=np.nan, right=np.nan)
    return y_out


def plot_joint_series(t: np.ndarray, y_state: np.ndarray, y_cmd: Optional[np.ndarray],
                      joint_indices: List[int], title: str, ylabel: str, legend_labels=("state", "cmd")):
    n = len(joint_indices)
    if n == 0 or t.size == 0:
        return
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.0), sharex=True)
    axes = np.array(axes).reshape(rows, cols)
    for k, j in enumerate(joint_indices):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        if y_state.size:
            ax.plot(t, y_state[:, j], label=legend_labels[0], lw=1.2)
        if y_cmd is not None and y_cmd.size:
            ax.plot(t, y_cmd[:, j], label=legend_labels[1], lw=1.0, alpha=0.8)
        ax.set_title(f"joint {j}")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.3)
        if r == rows - 1:
            ax.set_xlabel("time [s]")
        if k == 0:
            ax.legend(loc="best")
    # Hide empty subplots
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])


def main():
    args = parse_args()

    if not os.path.exists(args.bag):
        raise FileNotFoundError(args.bag)

    bag = rosbag.Bag(args.bag, "r")

    # Detect topics if not provided
    state_topic = args.state_topic
    cmd_topic = args.cmd_topic
    imu_topic = args.imu_topic

    if state_topic is None or cmd_topic is None or (imu_topic is None and not args.no_imu):
        s_guess, c_guess, i_guess = detect_topics(bag)
        state_topic = state_topic or s_guess
        cmd_topic = cmd_topic or c_guess
        imu_topic = imu_topic or i_guess

    print(f"Using topics -> state: {state_topic}, cmd: {cmd_topic}, imu: {imu_topic}")

    if state_topic is None:
        raise RuntimeError("Cannot find state topic; please pass --state-topic")
    if cmd_topic is None:
        raise RuntimeError("Cannot find cmd topic; please pass --cmd-topic")

    t0 = None
    state, t0 = extract_state(bag, state_topic, t0, args.t_start, args.t_end)
    cmd, t0 = extract_cmd(bag, cmd_topic, t0, args.t_start, args.t_end)

    if not state:
        raise RuntimeError("No state messages found in the selected time window/topic")
    if not cmd:
        raise RuntimeError("No cmd messages found in the selected time window/topic")

    # Downsample for speed/clarity
    ds = max(1, args.downsample)
    t_state = state["t"][::ds]
    q_state = state["q"][::ds]
    dq_state = state["dq"][::ds]
    ddq_state = state["ddq"][::ds]
    tau_est_state = state["tau_est"][::ds]

    t_cmd = cmd["t"][::ds]
    q_cmd = cmd["q"][::ds]
    dq_cmd = cmd["dq"][::ds]
    tau_cmd = cmd["tau"][::ds]
    kp_cmd = cmd["kp"][::ds]
    kd_cmd = cmd["kd"][::ds]

    # Choose timeline for plotting: use state timeline as reference
    t_plot = t_state

    # Align cmd to state time
    q_cmd_al = align_series(t_cmd, q_cmd, t_plot, args.align)
    dq_cmd_al = align_series(t_cmd, dq_cmd, t_plot, args.align)
    tau_cmd_al = align_series(t_cmd, tau_cmd, t_plot, args.align)
    kp_cmd_al = align_series(t_cmd, kp_cmd, t_plot, args.align)
    kd_cmd_al = align_series(t_cmd, kd_cmd, t_plot, args.align)

    joint_count = min(state["joint_count"], cmd["joint_count"])
    joints = parse_joint_selection(joint_count, args.joints)

    if args.title:
        plt.figure().suptitle(args.title)
        plt.close()

    # q
    plot_joint_series(t_plot, q_state[:, joints], q_cmd_al[:, joints], joints,
                      title="Joint position q: state vs cmd", ylabel="q [rad]")
    # dq
    plot_joint_series(t_plot, dq_state[:, joints], dq_cmd_al[:, joints], joints,
                      title="Joint velocity dq: state vs cmd", ylabel="dq [rad/s]")
    # tau vs tau_est
    plot_joint_series(t_plot, tau_est_state[:, joints], tau_cmd_al[:, joints], joints,
                      title="Torque: tau_est (state) vs tau (cmd)", ylabel="tau [Nm]", legend_labels=("tau_est", "tau_cmd"))
    # ddq (only state)
    plot_joint_series(t_plot, ddq_state[:, joints], None, joints,
                      title="Joint acceleration ddq (state)", ylabel="ddq [rad/s^2]")

    # kp/kd (cmd only)
    if args.show_kpkd:
        plot_joint_series(t_plot, kp_cmd_al[:, joints], None, joints,
                          title="Gain kp (cmd)", ylabel="kp")
        plot_joint_series(t_plot, kd_cmd_al[:, joints], None, joints,
                          title="Gain kd (cmd)", ylabel="kd")

    # IMU
    if not args.no_imu:
        imu_quat = state.get("imu_quat", np.empty((0, 4)))
        imu_gyro = state.get("imu_gyro", np.empty((0, 3)))
        imu_acc = state.get("imu_acc", np.empty((0, 3)))
        if imu_quat.size or imu_gyro.size or imu_acc.size:
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            if imu_quat.size:
                axes[0].plot(t_plot[:imu_quat.shape[0]], imu_quat[:len(t_plot), 0], label="qx")
                axes[0].plot(t_plot[:imu_quat.shape[0]], imu_quat[:len(t_plot), 1], label="qy")
                axes[0].plot(t_plot[:imu_quat.shape[0]], imu_quat[:len(t_plot), 2], label="qz")
                axes[0].plot(t_plot[:imu_quat.shape[0]], imu_quat[:len(t_plot), 3], label="qw")
                axes[0].set_ylabel("quat")
                axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.3)
            if imu_gyro.size:
                n = min(len(t_plot), imu_gyro.shape[0])
                axes[1].plot(t_plot[:n], imu_gyro[:n, 0], label="gx")
                axes[1].plot(t_plot[:n], imu_gyro[:n, 1], label="gy")
                axes[1].plot(t_plot[:n], imu_gyro[:n, 2], label="gz")
                axes[1].set_ylabel("gyro [rad/s]")
                axes[1].legend(); axes[1].grid(True, ls="--", alpha=0.3)
            if imu_acc.size:
                n = min(len(t_plot), imu_acc.shape[0])
                axes[2].plot(t_plot[:n], imu_acc[:n, 0], label="ax")
                axes[2].plot(t_plot[:n], imu_acc[:n, 1], label="ay")
                axes[2].plot(t_plot[:n], imu_acc[:n, 2], label="az")
                axes[2].set_ylabel("acc [m/s^2]")
                axes[2].set_xlabel("time [s]")
                axes[2].legend(); axes[2].grid(True, ls="--", alpha=0.3)
            fig.suptitle("IMU signals")
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        else:
            print("No IMU data found inside state; if IMU is on a separate topic, pass --imu-topic and adapt extraction.")

    plt.show()
    bag.close()


if __name__ == "__main__":
    main()