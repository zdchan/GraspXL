import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use('TkAgg')


def from_to_as_rot(from_vect, to_vect):
    """
    Computes the rotation to rotate from_vect such that it has the same direction/orientation
    as to_vect.

    :param from_vect:
    :param to_vect: target orientation
    :return: rotation vector as euler angles
    """
    from_vect = from_vect / np.linalg.norm(from_vect)
    to_vect = to_vect / np.linalg.norm(to_vect)

    # rotation axis is orthogonal to the plane spanned by the two vectors
    rot_axis = np.cross(from_vect, to_vect)
    rot_axis_len = np.linalg.norm(rot_axis)
    rot_axis = rot_axis / rot_axis_len

    angle = np.arccos(np.dot(from_vect, to_vect))

    vect_axis_angle = rot_axis * angle

    return R.from_rotvec(vect_axis_angle).as_euler("XYZ")


def _plot_vect(vect, start=[0, 0, 0], color='r', size=3):
    arrow = ax.quiver(start[0], start[1], start[2], vect[0], vect[1], vect[2], [size], color=color)
    # ax.quiverkey(arrow,vect[0], vect[1], 0.8, label=label)


if __name__ == '__main__':
    vect_to_rot = [0.2, 0.8, 0.6]

    vect1 = np.asarray([0, -1.5, 1])
    vect2 = np.asarray([0.75, 0, 1])

    direction = vect2 - vect1

    euler_vect = from_to_as_rot(vect_to_rot, direction)

    rotation = R.from_euler("XYZ", euler_vect)

    rotated_vect = rotation.apply(vect_to_rot)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _plot_vect(direction, start=vect1, color='r')
    _plot_vect(vect1, color='g')
    _plot_vect(vect2, color='b')

    _plot_vect(vect_to_rot, color='y', size=5)
    _plot_vect(rotated_vect, start=vect1, color='orange', size=5)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.grid(True)

    plt.show()

    plt.close()
