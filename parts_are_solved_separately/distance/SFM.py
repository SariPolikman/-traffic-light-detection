import numpy as np

X = 0
Y = 1
Z = 2


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    """ transform pixels into normalized pixels using the focal length and principle point """
    return np.array([[(x[0] - pp[0]) / focal, (x[1] - pp[1]) / focal] for x in pts])


def unnormalize(pts, focal, pp):
    """transform normalized pixels into pixels using the focal length and principle point"""
    return [[(x[0] + pp[0]) * focal, (x[1] + pp[1]) * focal] for x in pts]



def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    t = EM[:3, 3]
    R = EM[:3, :3]
    foe = (t[X] / t[Z], t[Y] / t[Z])
    return R, foe, t[Z]


def rotate(pts, R):
    # rotate the points - pts using R
    return np.array([p @ R for p in pts])


def find_corresponding_points(p, norm_pts_rot, foe):
    """compute the epipolar line between p and foe
    run over all norm_pts_rot and find the one closest to the epipolar line
    return the closest point and its index"""

    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    d = [abs(m * pts[0] + n - pts[1] / (m ** 2 + 1) ** 2) for pts in norm_pts_rot]

    idx = np.argmin(d)
    return norm_pts_rot[np.argmin(idx)], idx


def calc_dist(p_curr, p_rot, foe, tZ):
    """calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    combine the two estimations and return estimated Z"""
    z_x = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    z_y = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])
    dis_x = abs(p_curr[0] - p_rot[0])
    dis_y = abs(p_curr[1] - p_rot[1])
    return (z_x * dis_x + z_y * dis_y) / (dis_x + dis_y)




