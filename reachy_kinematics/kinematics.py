"""Compute the kinematics of Reachy's arm using the KDL library and its URDF definition."""
from typing import Tuple

import numpy as np
import PyKDL as kdl

from .kdl_parser_py import urdf


def generate_solver(urdf_str: str):
    """Create an FK/IK solvers for each arm (left/right)."""
    success, urdf_tree = urdf.treeFromString(urdf_str)
    if not success:
        raise IOError('Could not parse the URDF!')

    chains = {}
    fk_solvers = {}
    ik_solvers = {}

    for side in ['left', 'right']:
        chains[side] = urdf_tree.getChain('torso', f'{side}_tip')
        fk_solvers[side] = kdl.ChainFkSolverPos_recursive(chains[side])

        ik_solvers[side] = kdl.ChainIkSolverPos_LMA(
            chains[side],
            eps=1e-5,
            _maxiter=500,
            _eps_joints=1e-15,
        )

    return chains, fk_solvers, ik_solvers


def forward_kinematics(fk_solver, joints: np.ndarray, nb_joints: int) -> Tuple[float, np.ndarray]:
    """Compute the forward kinematics of the given arm.

    The function assumes the number of joints is correct!
    """
    q = kdl.JntArray(nb_joints)
    for i, j in enumerate(joints):
        q[i] = j

    pose = kdl.Frame()
    res = fk_solver.JntToCart(q, pose)

    M = np.eye(4)
    M[:3, 3] = list(pose.p)
    for i in range(3):
        for j in range(3):
            M[i, j] = pose.M[i, j]

    return res, M


def inverse_kinematics(ik_solver, q0: np.ndarray, target_pose: np.ndarray, nb_joints: int) -> Tuple[float, np.ndarray]:
    """Compute the inverse kinematics of the given arm.

    The function assumes the number of joints is correct!
    """
    x, y, z = target_pose[:3, 3]
    R = target_pose[:3, :3].flatten().tolist()

    _q0 = kdl.JntArray(nb_joints)
    for i, q in enumerate(q0):
        _q0[i] = q

    pose = kdl.Frame()
    pose.p = kdl.Vector(x, y, z)
    pose.M = kdl.Rotation(*R)

    sol = kdl.JntArray(nb_joints)
    res = ik_solver.CartToJnt(_q0, pose, sol)
    sol = list(sol)

    return res, sol
