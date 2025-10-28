import numpy as np

# 3D rotation matrixes
def rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]])
def ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])
def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def rotate(rot_mat, vec):
    return rot_mat @ vec


def angle_between(v1, v2):
    v1_v2 = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    angle_Z = np.arccos(v1_v2 / (mag_v1 * mag_v2))
    return np.degrees(angle_Z)


def angle_finder(LMC_velocity_components): # x, y, and z components of the LMC velocity vector
    vel_vec_x =(42, 0, 0) # random vector pointing towards x axis. x = 42 because it is the answer to the ultimate question of life, the universe and everything.
    v_LMC = LMC_velocity_components
    Rz = angle_between([vel_vec_x[0], vel_vec_x[1]], [v_LMC[0], v_LMC[1]]) # angle between the two vectors in 2D
    
        # -------------------------------------------------------------------------#
    if (((v_LMC[0] < 0) & (v_LMC[1] <= 0 )) | ((v_LMC[0] > 0) & (v_LMC[1] <= 0))):
        Rz = -Rz
    else:
        Rz = Rz
        # -------------------------------------------------------------------------#  
           
    rotated_2D = rotate(rz(np.radians(Rz)), vel_vec_x) # rotation matrix around z axis

    Ry = angle_between(rotated_2D, v_LMC) # angle between the two vectors in 3D
        
        # -------------------------------------------------------------------------#
    if (((v_LMC[0] < 0) & (v_LMC[2] >= 0 )) | ((v_LMC[0] > 0) & (v_LMC[2] <= 0 ))): # to determine the direction of rotation
        Ry = -Ry
    else:
        Ry = Ry
        # -------------------------------------------------------------------------#
    print(f"Rz = {Rz}, Ry = {Ry}")
    return Rz, Ry