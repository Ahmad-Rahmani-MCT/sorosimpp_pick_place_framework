#!/usr/bin/env python3
import rospy
import actionlib
from geometry_msgs.msg import Point
from sorosimpp_pick_place_framework.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal # action message types
from sorosimpp_pick_place_framework.srv import GetTrajectory # service messsage types

# this node is a client to the service and action server 
# service for getting the path 
# action for the control to the desired pose
class PickPlaceSupervisor: 
    def __init__(self):
        rospy.init_node('supervisor_node')
        
        # setting up planner client (service)
        rospy.wait_for_service('get_trajectory')
        self.get_traj_service = rospy.ServiceProxy('get_trajectory', GetTrajectory)
        
        # setting up controller client (action)
        self.client = actionlib.SimpleActionClient('execute_trajectory', ExecuteTrajectoryAction)
        rospy.loginfo("Waiting for Controller Action Server")
        self.client.wait_for_server()
        rospy.loginfo("Controller Connected")

        # defining the task
        # defining object positions and drop zones 
        self.home_pose = Point(0.0, 0.0, 0.0) 
        self.objects = [
            Point(0.02, 0.06, 0.0),  # object 1 pose
            Point(0.05, 0.05, 0.0) # object 2 pose
        ]
        self.drop_zone = Point(-0.05, -0.05, 0.0)
 
    def move_robot(self, start, end, steps=40): 
        # geting path from planner
        try:
            resp = self.get_traj_service(start, end, steps)
            if not resp.success: 
                rospy.logerr("Planner failed to generate path.") 
                return False
            
            # sending path to controller
            goal = ExecuteTrajectoryGoal() # creating trajectory goal message type
            goal.trajectory = resp.trajectory
            
            rospy.loginfo(f"Sending Goal: Move to ({end.x:.2f}, {end.y:.2f})")
            self.client.send_goal(goal, feedback_cb=self.feedback_callback)    
            
            # waiting for results
            self.client.wait_for_result() 
            return self.client.get_result().success  
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def feedback_callback(self, feedback):
        # Optional: Print progress bars [TO BE MODIFIED]
        # We throttle the log to avoid flooding the terminal (e.g., every 0.5s)
        # feedback.state is "MOVING" or "FINE_TUNING"
        # feedback.dist_to_goal is the distance in meters
        rospy.loginfo_throttle(0.5, f"Status: {feedback.state} | Dist: {feedback.error_distance:.4f}m")

    def run_mission(self):
        # you need to know where the robot initial pose is. 
        # for the very first move, we assume we are at Home (0,0) or read you have read tf 
        current_pose = self.home_pose  

        for i, obj_loc in enumerate(self.objects):
            rospy.loginfo(f"PROCESSING OBJECT {i+1}")
            
            # go to object
            if self.move_robot(current_pose, obj_loc):
                rospy.loginfo("Reached Object. GRIPPING (5s wait)")
                rospy.sleep(5.0) # Simulate gripping time
                current_pose = obj_loc
            else:
                rospy.logerr("Failed to reach object. Aborting.")
                return

            # return home (Reset Dynamics)
            if self.move_robot(current_pose, self.home_pose):
                rospy.loginfo("Returned Home.")
                rospy.sleep(5.0)
                current_pose = self.home_pose
            
            # going to drop zone
            if self.move_robot(current_pose, self.drop_zone):
                rospy.loginfo("Reached Drop Zone. RELEASING...")
                rospy.sleep(5.0)
                current_pose = self.drop_zone
            
            # returning home (ready for next object)
            self.move_robot(current_pose, self.home_pose)
            rospy.sleep(5.0)
            current_pose = self.home_pose

        rospy.loginfo("MISSION COMPLETE!")

    def run_mission_direct(self):
        # you need to know where the robot initial pose is. 
        # for the very first move, we assume we are at Home (0,0) or read you have read tf 
        current_pose = self.home_pose  

        for i, obj_loc in enumerate(self.objects):
            rospy.loginfo(f"PROCESSING OBJECT {i+1} (DIRECT MODE)")
            
            # go to object (from wherever we are)
            if self.move_robot(current_pose, obj_loc):
                rospy.loginfo("Reached Object. GRIPPING (5s wait)")
                rospy.sleep(5.0) # Simulate gripping time
                current_pose = obj_loc
            else:
                rospy.logerr("Failed to reach object. Aborting.")
                return

            # go directly to drop zone
            if self.move_robot(current_pose, self.drop_zone):
                rospy.loginfo("Reached Drop Zone. RELEASING...")
                rospy.sleep(5.0)
                current_pose = self.drop_zone
            else:
                rospy.logerr("Failed to reach drop zone. Aborting.")
                return
            
            # loop continues, next start point is drop_zone

        # final return home
        rospy.loginfo("All objects placed. Returning Home.")
        self.move_robot(current_pose, self.home_pose)
        rospy.loginfo("MISSION COMPLETE!")

if __name__ == '__main__':
    supervisor = PickPlaceSupervisor()
    # supervisor.run_mission()
    supervisor.run_mission_direct()