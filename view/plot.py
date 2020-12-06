

def mark_tfl(image, candidates, aux, fig, title):
    sum_red = aux.count('R')
    assert len(aux) == sum_red + aux.count('G')

    x = [r[0] for r in candidates]
    y = [r[1] for r in candidates]

    fig.set_title(title)
    fig.imshow(image)

    fig.plot(y[:sum_red], x[:sum_red], 'ro', color='r', markersize=4)
    fig.plot(y[sum_red:], x[sum_red:], 'ro', color='g', markersize=4)


def show_img(image, title, fig):
    fig.set_title(title)
    fig.imshow(image)
# def mark_distances(image, candidates, fig, foe, distances, rot_pts):
#     fig.set_title("distances")
#     fig.imshow(image)
#     fig.plot(candidates[:, 0],candidates[:, 1], 'b+')
#     for i in range(len(candidates)):
#         fig.plot([candidates[i, 0], foe[0]], [candidates[i, 1], foe[1]], 'b', linewidth=0.2)
#         fig.text(candidates[i, 0], candidates[i, 1], r'{0:.1f}'.format(distances[i, 2]), color='r', fontsize=5)
#     fig.plot(foe[0], foe[1], 'r+')
#     fig.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
