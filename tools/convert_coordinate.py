import numpy as np
import open3d as o3d
import math

def convert_orthogonal_to_spherical(orthogonal):
    x_lidar = orthogonal[:, 0]
    y_lidar = orthogonal[:, 1]
    z_lidar = orthogonal[:, 2]
    intensity_lidar = orthogonal[:, 3]

    r_lidar = np.sqrt(x_lidar * x_lidar + y_lidar * y_lidar + z_lidar * z_lidar)
    theta_lidar = np.arccos(z_lidar / r_lidar)
    pi_lidar = np.arctan2(y_lidar, x_lidar)

    pi_lidar = np.where(pi_lidar < 0, 2 * math.pi + pi_lidar,pi_lidar)

    spherical = np.stack((r_lidar, theta_lidar, pi_lidar, intensity_lidar), axis = 1)

    spherical = np.delete(spherical, np.where(r_lidar == 0), axis = 0)

    return spherical

def convert_spherical_to_orthogonal(spherical):    
    r_lidar = spherical[:, 0]
    theta_lidar = spherical[:, 1]
    pi_lidar = spherical[:, 2]
    intensity_lidar = spherical[:, 3]

    x_lidar = r_lidar * np.sin(theta_lidar) * np.cos(pi_lidar)
    y_lidar = r_lidar * np.sin(theta_lidar) * np.sin(pi_lidar)
    z_lidar = r_lidar * np.cos(theta_lidar)

    orthogonal = np.stack((x_lidar, y_lidar, z_lidar, intensity_lidar), axis = 1)

    return orthogonal