import numpy as np

def quaternionToEuler(q):
    '''Assumes q = [x, y, z, w], returns [roll, pitch, yaw]'''
    qx, qy, qz, qw = q[:,0], q[:,1], q[:,2], q[:,3]

    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.sqrt(1 + 2*(qw*qy - qx*qz))
    cosp = np.sqrt(1 - 2*(qw*qy - qx*qz))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi/2

    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)

