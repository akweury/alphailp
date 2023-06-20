import copy
import torch
import numpy as np
import matplotlib

import config

from aitk.utils import data_utils
from sklearn.linear_model import LinearRegression

ness_index = config.score_type_index["ness"]
suff_index = config.score_type_index["suff"]
sn_index = config.score_type_index["sn"]


def is_sn(score):
    if score[sn_index] == 1:
        return True
    return False


def is_sn_th_good(score, threshold):
    if score[sn_index] > threshold:
        return True
    else:
        return False


def is_nc(score):
    if score[ness_index] == 1:
        return True
    else:
        return False


def is_nc_th_good(score, threshold):
    if score[ness_index] > threshold:
        return True
    else:
        return False


def is_sc(score):
    if score[suff_index] == 1:
        return True
    else:
        return False


def is_sc_th_good(score, threshold):
    if score[suff_index] > threshold:
        return True
    else:
        return False


def check_clu_result(clu_result):
    is_done = False
    for pred, res in clu_result.items():
        if res["result"] > 0.99:
            is_done = True
            break
    return is_done


def get_circle_error(c, r, points):
    dists = torch.sqrt(((points[:, [0, 2]] - c) ** 2).sum(1))
    return (dists - r) ** 2


def get_conic_error(poly_coef, center, points):
    # intersections
    k = (points[:, 2] - center[1]) / (points[:, 0] - center[0])
    b = points[:, 2] - k * points[:, 0]

    # solve intersections
    inter_coef_0 = poly_coef[0] + poly_coef[1] * k + poly_coef[2] * k ** 2
    inter_coef_1 = poly_coef[1] * b + 2 * k * b * poly_coef[2] + poly_coef[3] + poly_coef[4] * k
    inter_coef_2 = b ** 2 * poly_coef[2] + poly_coef[4] * b + (-1)

    x_intersects = []
    y_intersects = []
    for x_i, _x in enumerate(points[:, 0]):
        p = np.poly1d([inter_coef_0[x_i], inter_coef_1[x_i], inter_coef_2[x_i]])
        x_roots = p.r
        if _x > center[0]:
            x_intersects.append(x_roots.max())
        else:
            x_intersects.append(x_roots.min())

        if np.abs(x_intersects[x_i].imag) > 0.1:
            x_intersects[x_i] = 100
        else:
            x_intersects[x_i] = x_intersects[x_i].real

        y_intersects.append(x_intersects[x_i] * k[x_i] + b[x_i])

    intersects = torch.cat((torch.tensor(x_intersects).unsqueeze(1), torch.tensor(y_intersects).unsqueeze(1)), 1)
    dist = calc_dist(points[:, [0, 2]], intersects)

    # dist_real = []
    # for i in range(dist.shape[0]):
    #     if torch.abs(dist[i].imag) < 0.1:
    #         dist_real.append(dist[i].real)
    #     else:
    #         dist_real.append(100)
    return dist


def get_group_distribution(points, center):
    def cart2pol(x, y):
        rho = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        phi = torch.rad2deg(phi)
        return (rho, phi)

    round_divide = points.shape[1]
    points_2d = points[:, :, [0, 2]]
    area_angle = int(360 / round_divide)
    dir_vec = points_2d - center.unsqueeze(0).unsqueeze(0)
    dir_vec[:, :, 1] = -dir_vec[:, :, 1]
    rho, phi = cart2pol(dir_vec[:, :, 0], dir_vec[:, :, 1])
    phi[phi < 0] = 360 - torch.abs(phi[phi < 0])
    zone_id = (phi) // area_angle % round_divide

    is_even_distribution = []
    for g in zone_id:
        if len(torch.unique(g)) > round_divide - 1:
            is_even_distribution.append(True)
        else:
            is_even_distribution.append(False)

    return torch.tensor(is_even_distribution)


def eval_score(positive_score, negative_score):
    res_score = positive_score.pow(50) * (1 - negative_score.pow(50))
    return res_score


def metric_mse(data, axis):
    error = ((data - data.mean(dim=axis)) ** 2).mean(dim=axis)
    return error


def metric_count_mse(data, axis, epsilon=1e-10):
    counter = data_utils.op_count_nonzeros(data, axis, epsilon)
    error = ((counter - counter.mean()) ** 2).mean()
    return error


def fit_circle(data, args):
    min_group_indices = data_utils.get_comb(torch.tensor(range(data.shape[0])), 3).tolist()
    centers = torch.zeros(len(min_group_indices), 2)
    radius = torch.zeros(len(min_group_indices))
    for g_i, group_indices in enumerate(min_group_indices):
        c, r = calc_circles(data[group_indices], args.cir_error_th)
        if c is not None:
            centers[g_i] = c
            radius[g_i] = r
    centers = centers.mean(dim=0)
    radius = radius.mean()

    cir = {"center": centers, "radius": radius}
    return cir


def fit_conic(point_groups):
    matplotlib.use('TkAgg')

    # https://stackoverflow.com/a/47881806
    X = point_groups[:, 0:1]
    Y = point_groups[:, 1:2]

    # Formulate and solve the least squares problem ||Px - b ||^2
    P = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(P, b)[0].squeeze().astype(np.float64)

    A, B, C, D, E, F = x[0], x[1], x[2], x[3], x[4], -1
    # conic center:   https://en.wikipedia.org/wiki/Ellipse
    c_x = (2 * x[2] * x[3] - x[1] * x[4]) / (x[1] ** 2 - 4 * x[0] * x[2])
    c_z = (2 * x[0] * x[4] - x[1] * x[3]) / (x[1] ** 2 - 4 * x[0] * x[2])
    center = torch.tensor([c_x, c_z])
    a = -np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
            (A + C) + np.sqrt((A - C) ** 2 + B ** 2))) / (B ** 2 - 4 * A * C)
    b = -np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * F) * (
            (A + C) - np.sqrt((A - C) ** 2 + B ** 2))) / (B ** 2 - 4 * A * C)
    axis = torch.tensor([a * 2, b * 2])

    # if a <= 0 or b <= 0:
    #     print("debug ellipse axis ")

    # Print the equation of the ellipse in standard form
    # print(f'Ellipse: {x[0]:.3}x^2 + {x[1]:.3}xy+{x[2]:.3}y^2+{x[3]:.3}x+{x[4]:.3}y = 1, center:{center}.')

    conics = {"coef": x, "center": center, "axis": axis}

    return conics


def calc_circles(point_groups, collinear_th):
    # https://math.stackexchange.com/a/3503338
    complex_point_real = point_groups[:, 0]
    complex_point_imag = point_groups[:, 1]

    complex_points = torch.complex(complex_point_real, complex_point_imag)

    a, b, c = complex_points[0], complex_points[1], complex_points[2]
    if torch.abs(a - b).sum() < collinear_th or torch.abs(b - c).sum() < collinear_th or torch.abs(
            a - c).sum() < collinear_th:
        return None, None

    def f(z):
        return (z - a) / (b - a)

    def f_inv(w):
        return a + (b - a) * w

    w3 = f(c)
    if torch.abs(w3.imag) < collinear_th:
        # print("collinear point groups")
        return None, None
    center_complex = f_inv((w3 - w3 * w3.conj()) / (w3 - w3.conj()))
    r = torch.abs(a - center_complex)
    center = torch.tensor([center_complex.real, center_complex.imag])
    return center, r


def calc_colinearity(obj_tensors, indices_position):
    if obj_tensors.shape[1] < 3:
        raise ValueError

    # sort the objects by x or z axis
    for group_i in range(obj_tensors.shape[0]):
        x_range = obj_tensors[group_i, :, 0].max() - obj_tensors[group_i, :, 0].min()
        z_range = obj_tensors[group_i, :, 2].max() - obj_tensors[group_i, :, 2].min()

        if x_range > z_range:
            values, indices = torch.sort(obj_tensors[group_i, :, 0])
        else:
            values, indices = torch.sort(obj_tensors[group_i, :, 2])
        obj_tensors[group_i] = obj_tensors[group_i, indices]

    indices_a = list(range(1, obj_tensors.shape[1]))
    indices_b = list(range(obj_tensors.shape[1] - 1))
    indices_pos = indices_position

    collinearities = 0
    for i in range(len(indices_a)):
        diff = (obj_tensors[:, indices_a[i], indices_pos] - obj_tensors[:, indices_b[i], indices_pos])
        collinearities += torch.sqrt(torch.sum(diff ** 2, dim=-1))
    collinearities -= torch.sqrt(
        torch.sum((obj_tensors[:, 0, indices_pos] - obj_tensors[:, -1, indices_pos]) ** 2, dim=-1))
    return collinearities


def calc_dist(points, center):
    distance = torch.sqrt(torch.sum((points - center) ** 2, dim=-1))
    return distance


def calc_avg_dist(obj_tensors, indices_position):
    if obj_tensors.shape[1] < 3:
        raise ValueError

    indices_a = list(range(obj_tensors.shape[1] - 1, 0, -1))
    indices_b = list(range(obj_tensors.shape[1] - 2, -1, -1))
    distances = []
    for i in range(len(indices_a)):
        point_1 = obj_tensors[:, indices_a[i], indices_position]
        point_2 = obj_tensors[:, indices_b[i], indices_position]
        distance = torch.sqrt(torch.sum((point_1 - point_2) ** 2, dim=-1))
        distances.append(distance.tolist())

    error = torch.mean((torch.tensor(distances) - torch.mean(torch.tensor(distances), dim=0)) ** 2, dim=0)

    return error


def predict_dots():
    return None


def is_even_distributed_points(args, points_, shape):
    points = copy.deepcopy(points_)

    if shape == "line":
        points_sorted_x = points[points[:, 0].sort()[1]]
        delta_x = torch.abs((points_sorted_x.roll(-1, 0) - points_sorted_x)[:-1, :])
        distribute_error_x = (
                torch.abs(delta_x[:, 0] - delta_x[:, 0].mean(dim=0)).sum(dim=0) / (points.shape[0] - 1)).sum()

        points_sorted_y = points[points[:, 2].sort()[1]]
        delta_y = torch.abs((points_sorted_y.roll(-1, 0) - points_sorted_y)[:-1, :])
        distribute_error_y = (
                torch.abs(delta_y[:, 2] - delta_y[:, 2].mean(dim=0)).sum(dim=0) / (points.shape[0] - 1)).sum()

        if distribute_error_x < args.distribute_error_th and distribute_error_y < args.distribute_error_th:
            return True
        else:
            return False
    elif shape == "circle":
        raise NotImplementedError
    else:
        raise ValueError


def eval_clause_on_test_scenes(NSFR, args, clause, group_pred, ):
    V_T = NSFR.clause_eval_quick(group_pred)[0, 0]
    preds = [clause.head.pred.name]

    score = NSFR.get_test_target_prediciton(V_T, preds, args.device)
    score[score == 1] = 0.99

    return score


def eval_data(data):
    # first metric: mse
    value_diff = metric_mse(data, axis=0)
    # second metric
    type_diff = metric_count_mse(data, axis=1)
    return value_diff, type_diff


def fit_line(point_group):
    # https://stackoverflow.com/a/47881806
    X = point_group[:, 0:1]
    Z = point_group[:, 1:2]

    line_model = LinearRegression().fit(X, Z)
    slope = line_model.coef_
    intercept = line_model.intercept_

    c_x = X.mean()
    c_z = Z.mean()
    center = torch.tensor([c_x, c_z])

    end_A, end_B = None, None
    if X.max() - X.min() > Z.max() - Z.min():
        sorted_x, sorted_x_indices = X.sort(dim=0)
        end_A = torch.tensor([sorted_x[0], Z[sorted_x_indices[0]]])
        end_B = torch.tensor([sorted_x[-1], Z[sorted_x_indices[-1]]])
    else:
        sorted_z, sorted_z_indices = Z.sort(dim=0)
        end_A = torch.tensor([X[sorted_z_indices[0]], sorted_z[0]])
        end_B = torch.tensor([X[sorted_z_indices[-1]], sorted_z[-1]])

    # Print the equation of the line
    # print(f'Line: y = {slope[0][0]} * x + {intercept[0]}.')
    line = {"center": center, "slope": slope, "end_A": end_A, "end_B": end_B}

    return line
