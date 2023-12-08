from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(roll, pitch, yaw):
    # Create a rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    # Convert to quaternion and return
    return r.as_quat()
