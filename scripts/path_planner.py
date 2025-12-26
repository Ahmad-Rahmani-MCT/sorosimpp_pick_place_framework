#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Point
from sorosimpp_pick_place_framework.srv import GetTrajectory, GetTrajectoryResponse

def lin_path_gen(start, goal, n_steps): 
    # linear interpolation
    x = np.linspace(start.x, goal.x, n_steps)
    y = np.linspace(start.y, goal.y, n_steps)
    
    # convert to ROS points
    trajectory = []
    for i in range(n_steps):
        p = Point()
        p.x = x[i]
        p.y = y[i]
        p.z = 0.0 # plnar motion for now
        trajectory.append(p)
    return trajectory

def handle_get_trajectory(req):
    rospy.loginfo(f"Planner: Generating path from ({req.start.x:.2f}, {req.start.y:.2f}) to ({req.goal.x:.2f}, {req.goal.y:.2f})")
    
    try:
        path = lin_path_gen(req.start, req.goal, req.num_steps)
        return GetTrajectoryResponse(trajectory=path, success=True)
    except Exception as e:
        rospy.logerr(f"Planner Failed: {e}")
        return GetTrajectoryResponse(success=False)

def planner_server(): 
    rospy.init_node('motion_planner_node')
    s = rospy.Service('get_trajectory', GetTrajectory, handle_get_trajectory)
    rospy.loginfo("Motion Planner Service Ready.")
    rospy.spin()

if __name__ == "__main__":
    planner_server()