import numpy as np
from model.traffic_lights.distance import SFM


def visualize(fig, prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

    fig.set_title("distances")
    fig.imshow(curr_container.img)
    curr_p = curr_container.traffic_light

    fig.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

    for i in range(len(curr_p)):
        fig.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if curr_container.valid[i]:
            fig.text(curr_p[i, 0], curr_p[i, 1],
                     r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r', fontsize=6)
    fig.plot(foe[0], foe[1], 'r+')
    fig.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')


class FrameContainer(object):
    def __init__(self, img):
        self.img = img
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


def get_distance(fig, prev_img, curr_img, prev_coordination, curr_coordination, curr_EM, focal, pp):
    prev_container = FrameContainer(prev_img)
    curr_container = FrameContainer(curr_img)

    #########################
    prev_coordination = [(prev_cord[1], prev_cord[0]) for prev_cord in prev_coordination]
    curr_coordination = [(curr_cord[1], curr_cord[0]) for curr_cord in curr_coordination]
    #########################

    prev_container.traffic_light = np.array(prev_coordination)
    curr_container.traffic_light = np.array(curr_coordination)

    curr_container.EM = curr_EM
    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)

    visualize(fig, prev_container, curr_container, focal, pp)

    return curr_container.corresponding_ind
