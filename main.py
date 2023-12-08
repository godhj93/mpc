import airsim
from utils import euler_to_quaternion
import time
from controller import ModelPredictiveController
import numpy as np
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Initialize the quadcopter position and orientation
pose = client.simGetVehiclePose()
pose.position.x_val = 225 
pose.position.z_val = -1
x,y,z,w = euler_to_quaternion(0,0,90)
pose.orientation.x_val = x
pose.orientation.y_val = y
pose.orientation.z_val = z
pose.orientation.w_val = w
client.simSetVehiclePose(pose, True)

# Takeoff
client.takeoffAsync().join()

# Define Goal Position
goal_x = 225
goal_y = 100
goal_z = -1
# Set MPC
mpc = ModelPredictiveController(goal_position=(goal_x, goal_y, goal_z))
# Turn off mpc logging
mpc.controller.settings.supress_ipopt_output()



# Check if the quadcopter is at the goal position in radius 5m
while not ((pose.position.x_val - goal_x)**2 + (pose.position.y_val - goal_y)**2)**0.5 < 5:
    # Update the quadcopter position
    pose = client.simGetVehiclePose()
    velocity = client.getMultirotorState().kinematics_estimated.linear_velocity
    # print(pose)
    # print(velocity)
    
    mpc.state = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val, velocity.x_val, velocity.y_val, velocity.z_val])
    
    u = mpc.solve()

    print(f"Control Input: {u[0][0], u[1][0], u[2][0]}")
    print(f"Distance from Goal: {((pose.position.x_val - goal_x)**2 + (pose.position.y_val - goal_y)**2)**0.5}")
    # move the quadcopter in the direction of the goal position using acceleration
    
    client.moveByVelocityAsync(vx=u[0][0], vy=u[1][0], vz=-u[2][0], duration = 0.1, drivetrain= airsim.DrivetrainType.ForwardOnly).join()
    # client.moveByVelocityAsync(vx=1.0, vy=0, vz=0, duration = 10).join()
    # Logging the quadcopter position and distance from the goal position
    
    # break
print("Goal Reached")
print("Done")

client.reset()