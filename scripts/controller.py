#!/usr/bin/env python3
import numpy as np 
import torch 
import rospy
import actionlib
import math
import pickle
import os

from std_msgs.msg import Float64MultiArray 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point

# Action Imports
from sorosimpp_pick_place_framework.msg import ExecuteTrajectoryAction, ExecuteTrajectoryFeedback, ExecuteTrajectoryResult

# --- CONFIGURATION ---
NODE_FREQUENCY = 20.0 
DT = 1.0 / NODE_FREQUENCY
N_ACT = 3 
REQUIRED_FRAMES = ["cs19", "tip"] 

ACTUATION_LIMIT_MIN = 0.0
ACTUATION_LIMIT_MAX = 18.0
ACTUATION_RATE_LIMIT = 0.6 

# --- TOLERANCES ---
TRANSITION_TOLERANCE = 0.010  # 1cm: Switch NN -> PID
SUCCESS_TOLERANCE    = 0.005  # 2mm: Switch PID -> Success (Done)

# PID GAINS
PID_KP = -20.0  
PID_KI = -20.0   
ACT_ANGLES = [math.radians(90), math.radians(330), math.radians(210)]

# --- MODEL CLASS ---
class MLP_model(torch.nn.Module): 
    def __init__(self, input_flat_size, hidden_units, output_size, num_hidden_layers):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_flat_size, hidden_units) 
        layers = []
        for i in range(num_hidden_layers): 
            layers.append(torch.nn.Linear(hidden_units, hidden_units)) 
            layers.append(torch.nn.ReLU()) 
        self.backbone = torch.nn.Sequential(*layers) 
        self.output_layer = torch.nn.Linear(hidden_units, output_size) 
        self.relu = torch.nn.ReLU()
    def forward(self, x): 
        out = self.input_layer(x) 
        out = self.relu(out)
        out = self.output_layer(out) 
        return out

# --- CONTROLLER ACTION SERVER ---
class IK_Action_Server:
    def __init__(self):
        rospy.init_node("ik_controller_action_node")
        
        self.load_resources()
        
        # State Variables
        self.current_u = np.zeros(N_ACT)
        self.integral_error = np.zeros(2) 
        self.pose_buffer = []
        self.init_pose_buf_filled = False
        
        # Modes: INIT -> MOVING -> FINE_TUNING -> HOLDING (Success)
        self.control_mode = "INIT" 
        
        # Trajectory Containers
        self.scaled_des_trajectory = None
        self.traj_counter = 0
        self.final_target_phys = None 
        self.latched_u_base = None    
        
        # ROS Setup
        self.pub_act = rospy.Publisher("/sorosimpp/actuators", Float64MultiArray, queue_size=10) 
        self.sub_tf = rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        self.latest_tf = None

        self._as = actionlib.SimpleActionServer("execute_trajectory", ExecuteTrajectoryAction, execute_cb=self.action_callback, auto_start=False)
        self._as.start()
        
        rospy.loginfo("IK Action Server Started.")
        self.timer = rospy.Timer(rospy.Duration(DT), self.control_loop)

    def load_resources(self):
        script_path = os.path.abspath(__file__) 
        script_dir = os.path.dirname(script_path) 
        model_dir = os.path.join(script_dir, "ik_model_lines_data") 
        
        with open(os.path.join(model_dir, "input_scaler_lines.pkl"), 'rb') as f: self.input_scaler = pickle.load(f)
        with open(os.path.join(model_dir, "state_scaler_lines.pkl"), 'rb') as f: self.state_scaler = pickle.load(f)
        
        self.model = MLP_model(21, 30, 3, 1)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "IK_MLP_lines.pth"), map_location='cpu'))
        self.model.eval()

        phys_lims = np.array([[ACTUATION_LIMIT_MIN]*N_ACT, [ACTUATION_LIMIT_MAX]*N_ACT])
        scaled_lims = self.input_scaler.transform(phys_lims)
        self.u_min_s = np.min(scaled_lims, axis=0)
        self.u_max_s = np.max(scaled_lims, axis=0)
        self.u_rate_s = ACTUATION_RATE_LIMIT * self.input_scaler.scale_ 

    # --- ACTION CALLBACK ---
    def action_callback(self, goal):
        rospy.loginfo(f"Received Trajectory: {len(goal.trajectory)} points.")
        
        raw_traj = np.array([[p.x, p.y] for p in goal.trajectory])
        
        tip_scale = self.state_scaler.scale_[2:4] 
        tip_min = self.state_scaler.min_[2:4]
        self.scaled_des_trajectory = raw_traj * tip_scale + tip_min
        
        self.final_target_phys = raw_traj[-1]
        self.traj_counter = 0
        
        # Reset States for new move
        self.integral_error = np.zeros(2) 
        self.latched_u_base = None
        self.control_mode = "MOVING" # Start Phase 1
        
        rate = rospy.Rate(5)
        feedback = ExecuteTrajectoryFeedback()
        
        # Loop until we reach the final HOLDING state (Phase 3)
        while self.control_mode != "HOLDING":
            
            if self._as.is_preempt_requested():
                rospy.logwarn("Trajectory Preempted!")
                self._as.set_preempted()
                return

            # Get current distance for feedback
            curr_dist = 999.9
            if self.latest_tf:
                raw = self.get_latest_pose()
                if raw:
                    curr_dist = np.linalg.norm(np.array(raw[2:4]) - self.final_target_phys)

            feedback.dist_to_goal = curr_dist
            feedback.state = self.control_mode
            self._as.publish_feedback(feedback)
            
            rate.sleep()
            
        # If we exit loop, we are in HOLDING -> Success
        res = ExecuteTrajectoryResult()
        res.success = True
        self._as.set_succeeded(res)
        rospy.loginfo(f"Action Succeeded. Final Error: {curr_dist:.4f}m")

    # --- MAIN CONTROL LOOP ---
    def control_loop(self, event):
        if self.latest_tf is None: return
        raw_pose = self.get_latest_pose()
        if raw_pose is None: return
        
        scaled_pose = self.state_scaler.transform(np.array(raw_pose).reshape(1, -1))
        
        if not self.init_pose_buf_filled:
            self.pose_buffer.append(scaled_pose)
            if len(self.pose_buffer) == 3: self.init_pose_buf_filled = True
            return 
        else:
            self.pose_buffer = self.pose_buffer[1:] + [scaled_pose]

        # Calculate Distance to Goal
        dist = 999.0
        if self.final_target_phys is not None:
            dist = np.linalg.norm(np.array(raw_pose[2:4]) - self.final_target_phys)

        # ----------------------------------
        # STATE MACHINE LOGIC
        # ----------------------------------

        # PHASE 1: MOVING (Neural Network)
        if self.control_mode == "MOVING":
            # Transition Logic: Reached 1cm? -> Go to Phase 2
            if dist < TRANSITION_TOLERANCE:
                rospy.loginfo(f"Within 1cm ({dist:.4f}). Switching to FINE_TUNING (PID).")
                self.control_mode = "FINE_TUNING"
                self.latched_u_base = self.current_u.copy()
                self.integral_error = np.zeros(2) 
                return # Skip one tick to let state settle

            # NN Execution
            in_feat = self.prepare_features(scaled_pose, self.scaled_des_trajectory)
            with torch.no_grad():
                net_out = self.model(torch.tensor(in_feat, dtype=torch.float32).unsqueeze(0)).numpy().flatten()
            self.update_current_u(net_out)

        # PHASE 2: FINE_TUNING (PID Active)
        elif self.control_mode == "FINE_TUNING":
            # Strict Logic: Only exit if we truly hit the target (2mm)
            if dist < SUCCESS_TOLERANCE:
                rospy.loginfo(f"Within fine tolerance ({dist:.4f}). Converged! Switching to HOLDING.")
                self.control_mode = "HOLDING"
                return

            # PID Execution (Keeps running forever if not converged)
            delta_pid = self.calculate_pid_correction(raw_pose)
            target_u = self.latched_u_base + delta_pid
            self.update_current_u(target_u, is_absolute_target=True)

        # PHASE 3: HOLDING (Passive Station Keeping)
        elif self.control_mode == "HOLDING":
            # Continue PID to maintain position
            if self.final_target_phys is not None:
                delta_pid = self.calculate_pid_correction(raw_pose)
                target_u = self.latched_u_base + delta_pid
                self.update_current_u(target_u, is_absolute_target=True)

        # ----------------------------------

        self.publish_actuation()

    # --- HELPERS ---
    def get_latest_pose(self):
        cs, tip = None, None
        for t in self.latest_tf.transforms:
            if t.child_frame_id == "cs19": cs = t.transform.translation
            if t.child_frame_id == "tip": tip = t.transform.translation
        if cs and tip: return [cs.x, cs.y, tip.x, tip.y]
        return None

    def prepare_features(self, current_pose_scaled, traj_scaled):
        if self.traj_counter < len(traj_scaled):
            target = traj_scaled[self.traj_counter]
            self.traj_counter += 1
        else:
            target = traj_scaled[-1]
        buf_flat = np.array(self.pose_buffer).flatten()
        curr_flat = current_pose_scaled.flatten()
        return np.concatenate((target, curr_flat, buf_flat, self.current_u), axis=0)

    def calculate_pid_correction(self, raw_pose):
        current_tip = np.array(raw_pose[2:4])
        err = self.final_target_phys - current_tip
        
        self.integral_error += err * DT
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5)
        
        delta_phys = np.zeros(N_ACT)
        for i, angle in enumerate(ACT_ANGLES):
            p = err[0]*math.cos(angle) + err[1]*math.sin(angle)
            i_term = self.integral_error[0]*math.cos(angle) + self.integral_error[1]*math.sin(angle)
            delta_phys[i] = (PID_KP * p) + (PID_KI * i_term)
            
        return delta_phys * self.input_scaler.scale_

    def update_current_u(self, target_val, is_absolute_target=False):
        if is_absolute_target:
            raw_diff = target_val - self.current_u
        else:
            raw_diff = target_val - self.current_u
        clamped_diff = np.clip(raw_diff, -self.u_rate_s, self.u_rate_s)
        new_u = self.current_u + clamped_diff
        self.current_u = np.clip(new_u, self.u_min_s, self.u_max_s)

    def publish_actuation(self):
        msg = Float64MultiArray()
        msg.data = self.input_scaler.inverse_transform(self.current_u.reshape(1,-1)).flatten().tolist()
        self.pub_act.publish(msg)

    def tf_callback(self, msg):
        relevant = [t for t in msg.transforms if t.child_frame_id in REQUIRED_FRAMES]
        if relevant: self.latest_tf = msg 

if __name__ == '__main__':
    IK_Action_Server()
    rospy.spin()