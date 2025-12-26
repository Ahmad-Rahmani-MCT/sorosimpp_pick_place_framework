#!/usr/bin/env python3
import numpy as np 
import torch 
import rospy
import actionlib
import math
import pickle
import os

# importing ROS messages
from std_msgs.msg import Float64MultiArray 
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point

# importing action messages
from sorosimpp_controller.msg import ExecuteTrajectoryAction, ExecuteTrajectoryFeedback, ExecuteTrajectoryResult

# configurations
NODE_FREQUENCY = 20.0 
DT = 1.0 / NODE_FREQUENCY
N_ACT = 3 
REQUIRED_FRAMES = ["cs19", "tip"] 

ACTUATION_LIMIT_MIN = 0.0
ACTUATION_LIMIT_MAX = 18.0
ACTUATION_RATE_LIMIT = 0.6 
TARGET_TOLERANCE = 0.01 

# PI gains
PID_KP = -20.0  
PID_KI = -20.0   
ACT_ANGLES = [math.radians(90), math.radians(330), math.radians(210)]

# model class
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

# controller action server
class IK_Action_Server:
    def __init__(self):
        rospy.init_node("ik_controller_action_node")
        
        # loading models and scalers
        self.load_resources()
        
        # initial variables and buffers
        self.current_u = np.zeros(N_ACT)
        self.integral_error = np.zeros(2) 
        self.pose_buffer = []
        self.init_pose_buf_filled = False
        
        # modes: we define "INIT" (Waiting for buffer), "MOVING" (Following Traj), "HOLDING" (PID holding)
        self.control_mode = "INIT" 
        
        # other variables 
        self.scaled_des_trajectory = None
        self.traj_counter = 0
        self.final_target_phys = None # physical (x,y) goal
        self.latched_u_base = None    # bias for PI
        
        # seting up ROS
        self.pub_act = rospy.Publisher("/sorosimpp/actuators", Float64MultiArray, queue_size=10) 
        self.sub_tf = rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        self.latest_tf = None

        # starting the action server 
        self._as = actionlib.SimpleActionServer("execute_trajectory", ExecuteTrajectoryAction, execute_cb=self.action_callback, auto_start=False)
        self._as.start()
        rospy.loginfo("IK Action Server Started. Waiting for path")

        # starting control loop timer
        # ensures the robot stays active/holding even when no action is running
        self.timer = rospy.Timer(rospy.Duration(DT), self.control_loop)

    def load_resources(self):
        script_path = os.path.abspath(__file__) 
        script_dir = os.path.dirname(script_path) 
        model_dir_name = "ik_model_lines_data" 
        model_dir = os.path.join(script_dir,model_dir)
        
        # loading scalars
        input_scaler_name = "input_scaler_lines.pkl"
        state_scaler_name = "state_scaler_lines.pkl"
        with open(os.path.join(model_dir, input_scaler_name), 'rb') as f: self.input_scaler = pickle.load(f)
        with open(os.path.join(model_dir, state_scaler_name), 'rb') as f: self.state_scaler = pickle.load(f)
        
        # loading the model
        input_flat_size = 21
        hidden_units = 30
        output_size = 3
        num_hidden_layers = 1
        self.model = MLP_model(input_flat_size=input_flat_size, hidden_units=hidden_units, output_size=output_size, num_hidden_layers=num_hidden_layers)
        model_state_dict_filename = "IK_MLP_lines.pth"
        self.model.load_state_dict(torch.load(os.path.join(model_dir, model_state_dict_filename), map_location='cpu'))
        self.model.eval()

        # limits
        phys_lims = np.array([[ACTUATION_LIMIT_MIN]*N_ACT, [ACTUATION_LIMIT_MAX]*N_ACT])
        scaled_lims = self.input_scaler.transform(phys_lims)
        self.u_min_s = np.min(scaled_lims, axis=0)
        self.u_max_s = np.max(scaled_lims, axis=0)
        self.u_rate_s = ACTUATION_RATE_LIMIT * self.input_scaler.scale_ 

    # action callback 
    def action_callback(self, goal):
        rospy.loginfo(f"Received Trajectory with {len(goal.trajectory)} points.")
        
        # extract and scale the received trajectory
        # Convert ROS point message type to Numpy array
        raw_traj = np.array([[p.x, p.y] for p in goal.trajectory])
        
        # scale
        tip_scale = self.state_scaler.scale_[2:4] 
        tip_min = self.state_scaler.min_[2:4]
        self.scaled_des_trajectory = raw_traj * tip_scale + tip_min
        
        # update targets
        self.final_target_phys = raw_traj[-1]
        self.traj_counter = 0
        
        # switch operation mode to MOVING
        # reset PID integral error for the new move
        self.integral_error = np.zeros(2) 
        self.latched_u_base = None
        self.control_mode = "MOVING"
        
        # monitoring progress
        # control_loop handles the movement 
        # monitor completion to return success/failure
        rate = rospy.Rate(10)
        feedback = ExecuteTrajectoryFeedback()
        
        while self.control_mode == "MOVING":
            if self._as.is_preempt_requested():
                rospy.logwarn("Trajectory Preempted!")
                self._as.set_preempted()
                return

            # feedback
            feedback.current_step = self.traj_counter
            feedback.total_steps = len(raw_traj)
            # feedback.error_distance = ... (can be calculated here)
            self._as.publish_feedback(feedback)
            
            rate.sleep()
            
        # if loop exits and we are in HOLDING, it means we have finished the trajectory
        if self.control_mode == "HOLDING":
            res = ExecuteTrajectoryResult()
            res.success = True
            self._as.set_succeeded(res)
            rospy.loginfo("Trajectory Completed. Holding Position.")

    # main control loop 
    def control_loop(self, event):
        # processing the received data
        if self.latest_tf is None: return
        raw_pose = self.get_latest_pose()
        if raw_pose is None: return
        
        scaled_pose = self.state_scaler.transform(np.array(raw_pose).reshape(1, -1))
        
        # managing the pose buffer
        if not self.init_pose_buf_filled:
            self.pose_buffer.append(scaled_pose)
            if len(self.pose_buffer) == 3: self.init_pose_buf_filled = True
            return # wait for buffer filling
        else:
            # updating the buffer as FIFO
            self.pose_buffer = self.pose_buffer[1:] + [scaled_pose]

        # logic selection based on Mode
        
        # MODE: MOVING (phase 1: NN "global coarse approach") 
        if self.control_mode == "MOVING":
            # checking the tolerance
            dist = np.linalg.norm(np.array(raw_pose[2:4]) - self.final_target_phys)
            
            if dist < TARGET_TOLERANCE:
                # Reached Goal -> Switch to HOLDING !! [TO BE MODIFIED]
                self.control_mode = "HOLDING"
                self.latched_u_base = self.current_u.copy()
                self.integral_error = np.zeros(2) # Reset I-term for clean holding
                return

            # run NN
            in_feat = self.prepare_features(scaled_pose, self.scaled_des_trajectory)
            with torch.no_grad():
                net_out = self.model(torch.tensor(in_feat, dtype=torch.float32).unsqueeze(0)).numpy().flatten()
            
            # apply rate limit and update U
            self.update_current_u(net_out)

        # MODE: HOLDING (Phase 2: PID pose keeping and fine tuning)
        elif self.control_mode == "HOLDING":
            if self.final_target_phys is not None:
                # run PID
                delta_pid = self.calculate_pid_correction(raw_pose)
                
                # base is latched value (from last NN) + PID correction
                target_u = self.latched_u_base + delta_pid
                
                # applying rate limit
                self.update_current_u(target_u, is_absolute_target=True)
                
        # --- MODE: INIT --- [TO BO MODIFIED]
        else:
            # Do nothing or publish zero? 
            # Ideally publish last known 'current_u' to keep chambers filled.
            pass

        # 3. Publish
        self.publish_actuation()

    # helper methods
    def get_latest_pose(self):
        # (Same TF logic as before)
        cs, tip = None, None
        for t in self.latest_tf.transforms:
            if t.child_frame_id == "cs19": cs = t.transform.translation
            if t.child_frame_id == "tip": tip = t.transform.translation
        if cs and tip: return [cs.x, cs.y, tip.x, tip.y]
        return None

    def prepare_features(self, current_pose_scaled, traj_scaled):
        # Trajectory Logic
        if self.traj_counter < len(traj_scaled):
            target = traj_scaled[self.traj_counter]
            self.traj_counter += 1
        else:
            target = traj_scaled[-1] # Hold last point
            
        # Flatten buffer
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
        """ Handles Rate Limiting and Absolute Clamping """
        if is_absolute_target:
            raw_diff = target_val - self.current_u
        else:
            # If target_val is the raw NN output, the diff is (NN - current)
            raw_diff = target_val - self.current_u
            
        clamped_diff = np.clip(raw_diff, -self.u_rate_s, self.u_rate_s)
        new_u = self.current_u + clamped_diff
        self.current_u = np.clip(new_u, self.u_min_s, self.u_max_s)

    def publish_actuation(self):
        msg = Float64MultiArray()
        msg.data = self.input_scaler.inverse_transform(self.current_u.reshape(1,-1)).flatten().tolist()
        self.pub_act.publish(msg)

    def tf_callback(self, msg):
        # Simple filter for efficiency
        relevant = [t for t in msg.transforms if t.child_frame_id in REQUIRED_FRAMES]
        if relevant: 
            # We need to construct a partial message or store locally
            # Simplification: Just store the whole msg if it has what we need
            self.latest_tf = msg 

if __name__ == '__main__':
    IK_Action_Server()
    rospy.spin()