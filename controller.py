from sys import argv
import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack


# same time as RaceCar.time_step
time_step = 0.1

# gains for velocity PID controller
KP_V = 6.0
KI_V = 0.3
KD_V = 4

# gains for steering angle PID controller
KP_DELTA = 10.0
KI_DELTA = 0.0
KD_DELTA = 0.7


# Lookahead distance along the raceline 
BASE_LOOKAHEAD = 10.0   # min lookahead (m)
MAX_LOOKAHEAD  = 25.0   # max lookahead (m)

_prev_v_error = 0.0
_int_v_error = 0.0

_prev_delta_error = 0.0
_int_delta_error = 0.0

# add raceline 
raceline = np.loadtxt(argv[2], comments="#", delimiter=",")
raceline = raceline[:, 0:2]

def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _find_lookahead_point(
    path_points: np.ndarray, position: np.ndarray, lookahead_distance: float
) -> np.ndarray:
    """Find the reference point roughly `lookahead_distance` ahead from the 
    current position of the vehicle. 
    """
    # find nearest point on the path
    dists = np.linalg.norm(path_points - position[None, :], axis=1)
    idx_nearest = int(np.argmin(dists))

    N = path_points.shape[0]
    acc_dist = 0.0
    i = idx_nearest

    # walk forward along the path until we've accumulated the lookahead distance
    while acc_dist < lookahead_distance:
        j = (i + 1) % N
        step = np.linalg.norm(path_points[j] - path_points[i])
        acc_dist += step
        i = j
        if step == 0:
            break  # avoid potential infinite loops

    return path_points[i]


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    
    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    sx, sy = state[0], state[1]
    v = state[3]
    phi = state[4]
    position = np.array([sx, sy])

    speed_factor = np.clip(v / max(parameters[5], 1e-3), 0.0, 1.0)
    lookahead = BASE_LOOKAHEAD + (MAX_LOOKAHEAD - BASE_LOOKAHEAD) * speed_factor

    lookahead_point = _find_lookahead_point(raceline, position, lookahead)

    # calculate heading  
    vec_to_lookahead = lookahead_point - position
    desired_heading = np.arctan2(vec_to_lookahead[1], vec_to_lookahead[0])

    # calculate difference from current heading
    heading_error = _wrap_to_pi(desired_heading - phi)

    # assume the vechile will go about in a circle
    L = parameters[0]  
    L_d = np.linalg.norm(vec_to_lookahead)
    curvature = 2.0 * np.sin(heading_error) / max(L_d, 1e-3)
    delta_ref = np.arctan(L*curvature)

    delta_min = parameters[1] 
    delta_max = parameters[4] 
    delta_ref = np.clip(delta_ref, delta_min, delta_max) # clamp the steering angle

    if abs(curvature) < 1e-3:
        v_ref = parameters[5]  # max vel on straight
    else:
        curvature_factor = min(1.0, abs(heading_error) / (np.pi / 4.0))
        v_ref = 60 * (1.0 - 0.70 * curvature_factor)
        v_ref = max(v_ref, 31)

    # clamp the velocity according to limits
    v_min = parameters[2]
    v_max = parameters[5]
    v_ref = np.clip(v_ref, v_min, v_max)

    return np.array([delta_ref, v_ref], dtype=float)


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    global _prev_v_error, _int_v_error
    global _prev_delta_error, _int_delta_error

    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    assert desired.shape == (2,)

    delta = state[2]
    v = state[3]

    delta_ref = desired[0]
    v_ref = desired[1]

    # delta PID 
    delta_error = _wrap_to_pi(delta_ref - delta)
    _int_delta_error += delta_error * time_step
    delta_error_dot = (delta_error - _prev_delta_error) / time_step 
    _prev_delta_error = delta_error

    v_delta = (
        KP_DELTA * delta_error
        + KI_DELTA * _int_delta_error
        + KD_DELTA * delta_error_dot
    )

    # clamp the steering angle velocity 
    v_delta_min = parameters[7]
    v_delta_max = parameters[9]
    v_delta = np.clip(v_delta, v_delta_min, v_delta_max)

    # Speed PID 
    v_error = v_ref - v
    _int_v_error += v_error * time_step
    v_error_dot = (v_error - _prev_v_error) / time_step
    _prev_v_error = v_error

    a = KP_V * v_error + KI_V * _int_v_error + KD_V * v_error_dot

    # clamp the acceleration 
    a_min = parameters[8]
    a_max = parameters[10]
    a = np.clip(a, a_min, a_max)

    return np.array([v_delta, a], dtype=float)
