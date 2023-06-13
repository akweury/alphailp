import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

import config
from aitk import ai_interface
from aitk.utils import eval_utils


# --------------------------- evaluate operations ----------------------------------------------------------------------

# https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if vector.sum() == 0:
        return np.zeros(vector.shape)
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    # inner = np.sum(v1.reshape(-1, 3) * v2.reshape(-1, 3), axis=1)
    # norms = np.linalg.norm(v1, axis=1, ord=2) * np.linalg.norm(v2, axis=1, ord=2)
    v1_u = v1 / (np.linalg.norm(v1, axis=1, ord=2, keepdims=True) + 1e-9)
    v2_u = v2 / (np.linalg.norm(v2, axis=1, ord=2, keepdims=True) + 1e-9)

    rad = np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))
    deg = np.rad2deg(rad)
    deg[deg > 90] = 180 - deg[deg > 90]
    return deg


def radian_between_tensor(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    # inner = np.sum(v1.reshape(-1, 3) * v2.reshape(-1, 3), axis=1)
    # norms = np.linalg.norm(v1, axis=1, ord=2) * np.linalg.norm(v2, axis=1, ord=2)
    v1_u = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-20)
    v2_u = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-20)

    rad = torch.arccos(torch.clip(torch.sum(v1_u * v2_u, dim=1), -1.0, 1.0))

    # deg = torch.rad2deg(rad)
    # deg[deg > 90] = 180 - deg[deg > 90]
    return rad.float()


def avg_angle_between_tensor(v1, v2):
    # deg_diff = angle_between(v1.to("cpu").detach().numpy(), v2.to("cpu").detach().numpy())
    # deg_diff = np.sum(np.abs(deg_diff))
    mask = v1.sum(dim=-1) != 0
    radian_diff = radian_between_tensor(v1, v2)
    deg_diff = torch.rad2deg(radian_diff)
    deg_diff = torch.abs(deg_diff)
    deg_diff[deg_diff > 90] = 180 - deg_diff[deg_diff > 90]

    deg_diff_5 = deg_diff > 5
    deg_diff_5 = deg_diff_5.sum() / torch.count_nonzero(mask)

    deg_diff_11d25 = deg_diff > 11.25
    deg_diff_11d25 = deg_diff_11d25.sum() / torch.count_nonzero(mask)

    deg_diff_22d5 = deg_diff > 22.5
    deg_diff_22d5 = deg_diff_22d5.sum() / torch.count_nonzero(mask)

    deg_diff_30 = deg_diff > 30
    deg_diff_30 = deg_diff_30.sum() / torch.count_nonzero(mask)

    avg_angle = deg_diff.sum() / deg_diff.size()[0]
    median_angle = torch.median(deg_diff)

    return avg_angle.to("cpu").detach().numpy(), \
        median_angle.to("cpu").detach().numpy(), \
        deg_diff_5.to("cpu").detach().numpy(), \
        deg_diff_11d25.to("cpu").detach().numpy(), \
        deg_diff_22d5.to("cpu").detach().numpy(), \
        deg_diff_30.to("cpu").detach().numpy()


def avg_angle_between_np(output, target):
    mask = target.sum(axis=2) != 0
    deg_diff = np.zeros(target.shape[:2])
    deg_diff[mask] = angle_between(target[mask], output[mask])
    # deg_diff = angle_between(v1.to("cpu").detach().numpy(), v2.to("cpu").detach().numpy())
    # deg_diff = np.abs(deg_diff)
    # radian_diff = radian_between_tensor(v1, v2)
    # deg_diff = torch.rad2deg(radian_diff)
    # deg_diff[deg_diff > 90] = 180 - deg_diff[deg_diff > 90]

    deg_diff_5 = deg_diff > 5
    deg_diff_5 = deg_diff_5.sum() / np.count_nonzero(mask)

    deg_diff_11d25 = deg_diff > 11.25
    deg_diff_11d25 = deg_diff_11d25.sum() / np.count_nonzero(mask)

    deg_diff_22d5 = deg_diff > 22.5
    deg_diff_22d5 = deg_diff_22d5.sum() / np.count_nonzero(mask)

    deg_diff_30 = deg_diff > 30
    deg_diff_30 = deg_diff_30.sum() / np.count_nonzero(mask)

    avg_angle = deg_diff.sum() / np.count_nonzero(mask)
    median_angle = np.median(deg_diff[mask])

    return avg_angle, \
        median_angle, \
        deg_diff_5, \
        deg_diff_11d25, \
        deg_diff_22d5, \
        deg_diff_30

    # return avg_angle.to("cpu").detach().numpy(), \
    #        median_angle.to("cpu").detach().numpy(), \
    #        deg_diff_5.to("cpu").detach().numpy(), \
    #        deg_diff_11d25.to("cpu").detach().numpy(), \
    #        deg_diff_22d5.to("cpu").detach().numpy(), \
    #        deg_diff_30.to("cpu").detach().numpy()


def vertex2light_direction(vertex_map, light_sorce):
    light_direction = -vertex_map + light_sorce
    light_direction_map = light_direction / np.sqrt(np.sum(light_direction ** 2, axis=-1, keepdims=True))

    return light_direction_map


def vertex2light_direction_tensor(vertex_map, light_sorce):
    light_direction = light_sorce - vertex_map
    light_direction_map = light_direction / (torch.norm(light_direction, p=2, dim=1, keepdim=True) + 1e-20)

    return light_direction_map


def albedo_tensor(I, N, L):
    rho = I.reshape(1, 1, 512, 512) / (torch.sum(N * L, dim=1, keepdim=True) + 1e-20)
    return rho


def albedo(I, mask, G, tranculate_threshold):
    albedo_norm = I / (G + 1e-20)
    # tranculation
    albedo_norm[albedo_norm > 255] = 255
    albedo_norm[albedo_norm < 1e-2] = 1e-2
    # norm
    # albedo_norm = (albedo_norm + tranculate_threshold) / (tranculate_threshold * 2)
    albedo_norm[mask] = 0
    return albedo_norm


def angle_between_2d(m1, m2):
    """ Returns the angle in radians between matrix 'm1' and 'm2'::"""
    m1_u = m1 / (np.linalg.norm(m1, axis=2, ord=2, keepdims=True) + 1e-9)
    m2_u = m2 / (np.linalg.norm(m2, axis=2, ord=2, keepdims=True) + 1e-9)

    rad = np.arccos(np.clip(np.sum(m1_u * m2_u, axis=2), -1.0, 1.0))
    deg = np.rad2deg(rad)
    deg[deg > 90] = 180 - deg[deg > 90]
    return deg


def radians_between_2d_tensor(t1, t2, mask=None):
    """ Returns the angle in radians between matrix 'm1' and 'm2'::"""
    # t1 = t1.permute(0, 2, 3, 1).to("cpu").detach().numpy()
    # t2 = t2.permute(0, 2, 3, 1).to("cpu").detach().numpy()
    # mask = mask.to("cpu").permute(0, 2, 3, 1).squeeze(-1)
    t1 = t1.permute(0, 2, 3, 1)
    t2 = t2.permute(0, 2, 3, 1)

    mask = mask.permute(0, 2, 3, 1).squeeze(-1)
    if mask is not None:
        t1 = t1[mask]
        t2 = t2[mask]
    t1_u = t1 / (torch.norm(t1, dim=-1, keepdim=True) + 1e-9)
    t2_u = t2 / (torch.norm(t2, dim=-1, keepdim=True) + 1e-9)
    # rad = torch.arccos(torch.clip(torch.sum(t1_u * t2_u, dim=-1), -1.0, 1.0))
    rad = torch.arccos(torch.clip(torch.sum(t1_u * t2_u, dim=-1), -1.0, 1.0))
    assert torch.sum(rad != rad) == 0
    # print(f"\t output normal: ({t1[0, 0].item():.2f},{t1[0, 1].item():.2f}, {t1[0, 2].item():.2f})")
    # print(f"\t target normal: ({t2[0, 0].item():.2f},{t2[0, 1].item():.2f},{t2[0, 2].item():.2f}\n"
    #       f"\t rad:{rad[0].item():.2f}\n"
    #       f"\t mse:{F.mse_loss(t1[0, :], t2[0, :]):.2f}\n")

    # deg = torch.rad2deg(rad)
    # deg[deg > 90] = 180 - deg[deg > 90]

    return rad


def mse(img_1, img_2, valid_pixels=None):
    """
    :param img_1: np_array of image with shape height*width*channel
    :param img_2: np_array of image with shape height*width*channel
    :return: mse error of two images in range [0,1]
    """
    if img_1.shape != img_2.shape:
        print("MSE Error: img 1 and img 2 do not have the same shape!")
        raise ValueError

    h, w, c = img_1.shape
    diff = np.sum(np.abs(img_1 - img_2))
    if valid_pixels is not None:
        # only calculate the difference of valid pixels
        diff /= (valid_pixels * c)
    else:
        # calculate the difference of whole image
        diff /= (h * w * c)

    return diff


def get_valid_pixels(img):
    return np.count_nonzero(np.sum(img, axis=2) > 0)


def get_valid_pixels_idx(img):
    return np.sum(img, axis=2) != 0


def canny_edge_filter(img_path):
    img = cv.imread(img_path, 0)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    edges = cv.Canny(img, 1000, 0, apertureSize=7, L2gradient=True)
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edge '), plt.xticks([]), plt.yticks([])

    edges = cv.Canny(img, 1000, 0, apertureSize=7, L2gradient=False)
    plt.subplot(133), plt.imshow(edges, cmap='gray')
    plt.title('Edge'), plt.xticks([]), plt.yticks([])

    # plt.savefig(str(Path(config.ws_path) / f"canny_comparison.png"), dpi=1000)


def binary(img):
    # h, w = img.shape[:2]
    img_permuted = img.permute(0, 2, 3, 1)

    mask = img_permuted.sum(dim=3) == 0
    c_permute = torch.zeros(size=img_permuted.shape)

    c_permute[~mask] = 1

    c = c_permute.permute(0, 3, 1, 2)
    return c


def bi_interpolation(lower_left, lower_right, upper_left, upper_right, x, y):
    return lower_left * (1 - x) * (1 - y) + lower_right * x * (1 - y) + upper_left * (1 - x) * y + upper_right * x * y


def filter_noise(numpy_array, threshold):
    if len(threshold) != 2:
        raise ValueError
    threshold_min, threshold_max = threshold[0], threshold[1]
    numpy_array[numpy_array < threshold_min] = threshold_min
    numpy_array[numpy_array > threshold_max] = threshold_max
    return numpy_array


def normalize3channel(numpy_array):
    mins, maxs = [], []
    if numpy_array.ndim != 3:
        raise ValueError
    h, w, c = numpy_array.shape
    for i in range(c):
        numpy_array[:, :, i], min, max = normalize(numpy_array[:, :, i], data=None)
        mins.append(min)
        maxs.append(max)
    return numpy_array, mins, maxs


def normalize(numpy_array, data=None):
    if numpy_array.ndim != 2:
        if numpy_array.shape == (512, 512):
            numpy_array = numpy_array.reshape(512, 512, 1)

    mask = numpy_array == 0
    if data is not None:
        min, max = data["minDepth"], data["maxDepth"]
    else:
        min, max = numpy_array.min(), numpy_array.max()
        if min == max:
            return numpy_array, 0, 1

    numpy_array[~mask] = (numpy_array[~mask] - min).astype(np.float32) / (max - min)
    return numpy_array, min, max


def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int32(patch_width / 2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)


def lightVisualization():
    points = []
    for px in range(-5, 6):
        for py in range(-5, 6):
            for pz in range(-5, 6):
                p = np.array([px, py, pz]).astype(np.float32) / 10
                if np.linalg.norm(p) > 0:
                    points.append(p / np.linalg.norm(p))
    return points


def cameraVisualization():
    points = []
    for p in range(-50, 51):
        ps = float(p) / 100.0
        points.append([ps, 0.5, 0.5])
        points.append([ps, -0.5, 0.5])
        points.append([ps, 0.5, -0.5])
        points.append([ps, -0.5, -0.5])
        points.append([0.5, ps, 0.5])
        points.append([-0.5, ps, 0.5])
        points.append([0.5, ps, -0.5])
        points.append([-0.5, ps, -0.5])
        points.append([0.5, 0.5, ps])
        points.append([0.5, -0.5, ps])
        points.append([-0.5, 0.5, ps])
        points.append([-0.5, -0.5, ps])

    for p in range(-30, 31):
        ps = float(p) / 100.0
        points.append([ps, 0.3, 0.3 + 0.8])
        points.append([ps, -0.3, 0.3 + 0.8])
        points.append([ps, 0.3, -0.3 + 0.8])
        points.append([ps, -0.3, -0.3 + 0.8])
        points.append([0.3, ps, 0.3 + 0.8])
        points.append([-0.3, ps, 0.3 + 0.8])
        points.append([0.3, ps, -0.3 + 0.8])
        points.append([-0.3, ps, -0.3 + 0.8])
        points.append([0.3, 0.3, ps + 0.8])
        points.append([0.3, -0.3, ps + 0.8])
        points.append([-0.3, 0.3, ps + 0.8])
        points.append([-0.3, -0.3, ps + 0.8])

    return points


def normalize2_32bit(img_scaled, data=None):
    if img_scaled.ndim == 2:
        raise ValueError

    if img_scaled.shape[2] == 1:
        normalized_img, mins, maxs = normalize(img_scaled.sum(axis=-1), data)
    elif img_scaled.shape[2] == 3:
        normalized_img, mins, maxs = normalize3channel(img_scaled)
    else:
        print("image scaled shape: " + str(img_scaled.shape))
        raise ValueError

    img_32bit = cv.normalize(normalized_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_32bit


def normalize2_16bit(img):
    newImg = img.reshape(512, 512)
    normalized_img, _, _ = normalize(newImg)
    img_16bit = cv.normalize(normalized_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_16bit


# --------------------------- convert operations -----------------------------------------------------------------------


def normal_point2view_point(normal, point, view_point):
    if np.dot(normal, (view_point.reshape(3) - point)) < 0:
        normal = -normal
    return normal


def compute_normal(vertex, cam_pos, mask, k):
    normals = np.zeros(shape=vertex.shape)
    for i in range(k, vertex.shape[0]):
        for j in range(k, vertex.shape[1]):
            if mask[i, j]:
                neighbors = vertex[max(i - k, 0): min(i + k + 1, vertex.shape[1] - 1),
                            max(0, j - k): min(j + k + 1, vertex.shape[0] - 1)]  # get its k neighbors
                # neighbors = vertex[i - k:i + k, j - k:j + k]
                neighbors = neighbors.reshape(neighbors.shape[0] * neighbors.shape[1], 3)

                # neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)  # delete center vertex

                # delete background vertex
                if neighbors.ndim == 2 and neighbors.shape[0] > 2:
                    neighbors = np.delete(neighbors, np.where(neighbors == np.zeros(3)), axis=0)

                if neighbors.shape[0] > 1:
                    neighbor_base = neighbors[0, :]
                    neighbors = neighbors[1:, :]
                else:
                    neighbor_base = vertex[i, j]

                plane_vectors = neighbors - neighbor_base

                u, s, vh = np.linalg.svd(plane_vectors)
                normal = vh.T[:, -1]
                normal = normal_point2view_point(normal, vertex[i][j], cam_pos)
                # if np.linalg.norm(normal) != 1:
                #     normal = normal / np.linalg.norm(normal)
                normals[i, j] = normal

    return normals


def generate_normals_all(vectors, vertex, normal_gt, svd_rank=3, MAX_ATTEMPT=1000):
    point_num, neighbors_num = vectors.shape[:2]
    random_idx = np.random.choice(neighbors_num, size=(point_num, MAX_ATTEMPT, svd_rank))
    candidate_neighbors = np.zeros(shape=(point_num, MAX_ATTEMPT, svd_rank, 3))
    # candidate_neighbors = vectors[random_idx]
    for i in range(point_num):
        for j in range(MAX_ATTEMPT):
            candidate_neighbors[i, j] = vectors[i][random_idx[i, j]]

    u, s, vh = np.linalg.svd(candidate_neighbors)
    candidate_normal = np.swapaxes(vh, -2, -1)[:, :, :, -1]
    vertex = np.repeat(np.expand_dims(vertex, axis=1), MAX_ATTEMPT, axis=1)
    normal = normal_point2view_point(candidate_normal, vertex, np.zeros(shape=vertex.shape))
    # if np.linalg.norm(normal, axis=1, ord=2) != 1:
    #     normal = normal / np.linalg.norm(normal)
    normal_gt_expended = np.repeat(np.expand_dims(normal_gt, axis=1), repeats=MAX_ATTEMPT, axis=1)
    error = angle_between_2d(normal, normal_gt_expended)
    best_error = np.min(error, axis=1)
    best_error_idx = np.argmin(error, axis=1)
    best_normals_idx = np.zeros(shape=(point_num, svd_rank))
    best_normals = np.zeros(shape=(point_num, 3))
    for i in range(point_num):
        best_normals_idx[i] = random_idx[i, best_error_idx[i]]
        best_normals[i] = normal[i, best_error_idx[i]]

    return best_normals, best_normals_idx, best_error


# --------------------------- convert functions -----------------------------------------------------------------------

def array2RGB(numpy_array, mask):
    # convert normal to RGB color
    min, max = numpy_array.min(), numpy_array.max()
    if min != max:
        numpy_array[mask] = ((numpy_array[mask] - min) / (max - min))

    return (numpy_array * 255).astype(np.uint8)


def normal2RGB(normals):
    mask = np.sum(np.abs(normals), axis=2) != 0
    rgb = np.zeros(shape=normals.shape)
    # convert normal to RGB color
    rgb[mask] = normals[mask] * 0.5 + 0.5
    rgb[:, :, 2][mask] = 1 - rgb[:, :, 2][mask]
    rgb[mask] = rgb[mask] * 255

    # rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = np.rint(rgb)
    return rgb.astype(np.uint8)


def light2RGB(lights):
    mask = np.sum(np.abs(lights), axis=2) != 0
    rgb = np.zeros(shape=lights.shape)
    # convert normal to RGB color
    rgb[mask] = lights[mask] * 0.5 + 0.5
    rgb[mask] = rgb[mask] * 255

    # rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = np.rint(rgb)
    return rgb.astype(np.uint8)


def normal2RGB_torch(normals):
    if normals.size() != (3, 512, 512):
        raise ValueError

    normals = normals.permute(1, 2, 0)
    mask = torch.sum(torch.abs(normals), dim=2) != 0
    rgb = torch.zeros(size=normals.shape).to(normals.device)

    # convert normal to RGB color
    rgb[mask] = normals[mask] * 0.5 + 0.5
    rgb[:, :, 2][mask] = 1 - rgb[:, :, 2][mask]
    rgb[mask] = rgb[mask] * 255

    # rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = torch.round(rgb)
    rgb = rgb.permute(2, 0, 1)
    return rgb.byte()


def normal2RGB_single(normal):
    normal = normal * 0.5 + 0.5
    normal[2] = 1 - normal[2]
    normal = normal * 255
    rgb = cv.normalize(normal, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return rgb


def rgb2normal(color):
    mask = color.sum(axis=2) == 0
    color_norm = np.zeros(shape=color.shape)
    h, w, c = color.shape
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                color_norm[i, j] = color[i, j] / 255.0
                color_norm[i, j, 2] = 1 - color_norm[i, j, 2]
                color_norm[i, j] = (color_norm[i, j] - 0.5) / 0.5
    # color_norm = color_norm / (np.linalg.norm(color_norm, axis=2, ord=2, keepdims=True)+1e-8)
    return color_norm


def rgb2normal_tensor(color):
    color = color.permute(0, 2, 3, 1)
    mask = color.sum(dim=-1) == 0
    color_norm = torch.zeros(color.shape).to(color.device)
    color_norm[~mask] = color[~mask] / 255.0
    color_norm[~mask][:, 2] = 1 - color_norm[~mask][:, 2]
    color_norm[~mask] = (color_norm[~mask] - 0.5) / 0.5

    return color_norm


def depth2vertex(depth, K, R, t):
    c, h, w = depth.shape

    camOrig = -R.transpose(0, 1) @ t.unsqueeze(1)

    X = torch.arange(0, depth.size(2)).repeat(depth.size(1), 1) - K[0, 2]
    Y = torch.transpose(torch.arange(0, depth.size(1)).repeat(depth.size(2), 1), 0, 1) - K[1, 2]
    Z = torch.ones(depth.size(1), depth.size(2)) * K[0, 0]
    Dir = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), 2)

    vertex = Dir * (depth.squeeze(0) / torch.norm(Dir, dim=2)).unsqueeze(2).repeat(1, 1, 3)
    vertex = R.transpose(0, 1) @ vertex.permute(2, 0, 1).reshape(3, -1)
    vertex = camOrig.unsqueeze(1).repeat(1, h, w) + vertex.reshape(3, h, w)
    vertex = vertex.permute(1, 2, 0)
    return np.array(vertex)


def vertex2normal(vertex, mask, cam_pos, k_idx):
    # mask = np.sum(np.abs(vertex), axis=2) != 0
    start = time.time()
    normals = compute_normal(vertex, cam_pos, mask, k_idx)
    gpu_time = time.time() - start
    normals_rgb = normal2RGB(normals)
    return normals, normals_rgb, gpu_time


def depth2normal(depth, mask, cam_pos, k_idx, K, R, t):
    if depth.ndim == 2:
        depth = np.expand_dims(depth, axis=2)
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(K),
                          torch.tensor(R).float(),
                          torch.tensor(t).float())
    return vertex2normal(vertex, mask, cam_pos, k_idx)


# -------------------------------------- openCV Utils ------------------------------------------
def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'upper_right':
        position = (w - 250, 80)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


def addCustomText(img, text, pos, font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos[0] > h or pos[0] < 0 or pos[1] > w or pos[1] < 0:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=pos,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
def addHist(img):
    h, w = img.shape[:2]
    fig = plt.figure()
    color = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
    color_ranges = []
    for i, col in enumerate(color):
        hist_min, hist_max = img[:, :, i].min().astype(np.int), img[:, :, i].max().astype(np.int)
        color_ranges.append([int(hist_min), int(hist_max)])

        if hist_max - hist_min < 2:
            return "..."
        histr, histr_x = np.histogram(img[:, :, i], bins=np.arange(hist_min, hist_max + 1))
        histr = np.delete(histr, np.where(histr == histr.max()), axis=0)

        # plot histogram on the image
        thick = 2
        histr = histr / max(histr.max(), 100)
        for i in range(histr.shape[0]):
            height = int(histr[i] * 50)
            width = int(w / histr.shape[0])
            img[max(h - 1 - height - thick, 0):min(h - 1, h - height + thick),
            max(0, i * width - thick):min(w - 1, i * width + thick)] = col
    plt.close('all')
    return color_ranges


def rgb_diff(img1, img2):
    diff = np.zeros(shape=(3, 256))
    h, w = img1.shape[:2]
    color = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
    for i, col in enumerate(color):
        histr1, histr_x1 = np.histogram(img1[:, :, i], bins=256, range=(0, 255))
        histr2, histr_x2 = np.histogram(img2[:, :, i], bins=256, range=(0, 255))

        histr1[0] = 0
        histr2[0] = 0

        diff[i, :] = histr1 - histr2
    return diff


def normal_histo(normal):
    mask = np.sum(normal, axis=2) == 0
    histo = np.zeros(shape=(3, 100))
    hist_x = None
    for i in range(3):
        histr1, hist_x = np.histogram(normal[:, :, i][~mask], bins=100, range=(-2, 2))
        histo[i, :] = histr1
    return histo, hist_x


def normal_diff(img1, img2):
    diff = np.zeros(shape=(3, 256))
    h, w = img1.shape[:2]
    color = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
    hist_x = None
    for i, col in enumerate(color):
        histr1, hist_x = np.histogram(img1[:, :, i], bins=256, range=(-1, 1))
        histr2, hist_x = np.histogram(img2[:, :, i], bins=256, range=(-1, 1))

        histr1[0] = 0
        histr2[0] = 0

        diff[i, :] = histr1 - histr2
    return diff, hist_x


def pure_color_img(color, size):
    img = np.zeros(shape=size).astype(np.uint8)
    img[:] = color
    return img


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'upper_right':
        position = (w - 250, 80)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


# ============================================ public functions ==================================================
def load_24bitImage(root):
    img = cv.imread(root, -1)
    img[np.isnan(img)] = 0

    return img


def get_cv_image(img_name, upper_right=None, font_scale=0.8):
    img = load_24bitImage(img_name)
    img = image_resize(img, width=512, height=512)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    if img.ndim == 2:
        img = cv.merge((img, img, img))

    return img


def show_images(array, title):
    cv.imshow(title, array)
    cv.waitKey(0)
    cv.destroyAllWindows()


def hconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv.resize(img,
                                (int(img.shape[1] * h_min / img.shape[0]),
                                 h_min), interpolation)
                      for img in img_list]

    return cv.hconcat(im_list_resize)


def draw_circles(img, data, radius, color, thickness):
    for point_i, point in enumerate(data):
        img = cv.circle(img, point.to(torch.int16).tolist(), radius[point_i], color[point_i], thickness)

    return img


def draw_lines(image, left_points, right_points, color=None, thickness=None):
    left_points = left_points.to(torch.int16).tolist()
    right_points = right_points.to(torch.int16).tolist()

    if len(left_points) < 2:
        return image
    for i in range(len(left_points)):
        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        image = cv.line(image, left_points[i], right_points[i], color[i], thickness)

    return image


def draw_text(image, text, position="upper_left", font_size=1.6):
    addText(image, text, pos=position, font_size=font_size)
    return image


def draw_custom_text(image, text, position, font_size=1.6):
    addCustomText(image, text, pos=position, font_size=font_size)
    return image


def save_image(final_image, image_output_path):
    cv.imwrite(image_output_path, final_image)


def visual_group_predictions(args, data, input_image, colors, thickness, group_tensor_index):

    # draw points on each objects
    data = torch.tensor(data)
    group_image = copy.deepcopy(input_image)
    indice_center_on_screen_x = group_tensor_index["x_center_screen"]
    indice_center_on_screen_y = group_tensor_index["y_center_screen"]
    indice_radius_on_screen = group_tensor_index["screen_radius"]
    screen_points = data[:, [indice_center_on_screen_x, indice_center_on_screen_y]][:args.group_e, :]
    screen_radius = data[:, indice_radius_on_screen][:args.group_e].to(torch.int16).tolist()
    group_pred_image = draw_circles(group_image, screen_points, radius=screen_radius, color=colors,
                                    thickness=thickness)

    # draw lines
    indice_left_screen_x = group_tensor_index["screen_left_x"]
    indice_left_screen_y = group_tensor_index["screen_left_y"]
    indice_right_screen_x = group_tensor_index["screen_right_x"]
    indice_right_screen_y = group_tensor_index["screen_right_y"]
    # args.group_max_e = 3
    screen_left_points = data[:, [indice_left_screen_x, indice_left_screen_y]][:args.group_e, :]
    screen_right_points = data[:, [indice_right_screen_x, indice_right_screen_y]][:args.group_e, :]
    group_pred_image = draw_lines(group_pred_image, screen_left_points, screen_right_points,
                                  color=colors, thickness=thickness)

    return group_pred_image


def visual_info(lang, image_shape, font_size):
    info_image = np.zeros(image_shape, dtype=np.uint8)

    # predicates info
    pi_c_text_position = [20, 80]
    text_y_shift = 20
    if len(lang.pi_clauses) == 0:
        info_image = draw_custom_text(info_image, f"No invented predicates.", pi_c_text_position, font_size=font_size)
        pi_c_text_position[1] += text_y_shift
    for pi_c in lang.pi_clauses:
        info_image = draw_custom_text(info_image, f"{pi_c}", pi_c_text_position, font_size=font_size)
        pi_c_text_position[1] += text_y_shift
    return info_image


def visualization(args, lang, scores=None, colors=None, thickness=None, radius=None):
    if colors is None:
        # Blue color in BGR
        colors = [
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
        ]
    if thickness is None:
        # Line thickness of 2 px
        thickness = 2
    if radius is None:
        radius = 10

    for data_type in ["true", "false"]:
        for img_i in range(len(args.test_group_pos)):
            data_name = args.image_name_dict['test'][data_type][img_i]
            if data_type == "true":
                data = args.test_group_pos[img_i]
            else:
                data = args.test_group_neg[img_i]

            # calculate scores
            # VM = ai_interface.get_vm(args, lang)
            # FC = ai_interface.get_fc(args, lang, VM)
            # NSFR = ai_interface.get_nsfr(args, lang, FC)

            # evaluate new clauses
            # scores = eval_utils.eval_clause_on_test_scenes(NSFR, args, lang.clauses[0], data.unsqueeze(0))

            visual_images = []
            # input image
            file_prefix = str(config.root / ".." / data_name[0]).split(".data0.json")[0]
            image_file = file_prefix + ".image.png"
            input_image = get_cv_image(image_file)

            # group prediction
            group_pred_image = visual_group_predictions(args, data, input_image, colors, thickness,
                                                        config.group_tensor_index)

            # information image
            info_image = visual_info(lang, input_image.shape, font_size=0.3)

            # adding header and footnotes
            input_image = draw_text(input_image, "input")
            visual_images.append(input_image)

            group_pred_image = draw_text(group_pred_image,
                                         f"group:{round(scores[data_type]['score'][img_i].tolist(), 4)}")
            group_pred_image = draw_text(group_pred_image, f"{scores[data_type]['clause'][img_i]}",
                                         position="lower_left", font_size=0.4)
            visual_images.append(group_pred_image)

            info_image = draw_text(info_image, f"Info:")
            visual_images.append(info_image)

            # final processing
            final_image = hconcat_resize(visual_images)
            final_image_filename = str(
                args.image_output_path / f"{data_name[0].split('/')[-1].split('.data0.json')[0]}.output.png")

            save_image(final_image, final_image_filename)


def visual_lines(args, line_tensors, line_indices, data_type):
    colors = [
        (255, 0, 0),  # Blue
        (255, 255, 0),  # Cyan
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (0, 255, 255),  # Yellow
    ]
    thickness = 3

    if "pos" in data_type:
        dtype = "true"
    else:
        dtype = "false"

    for i in range(len(line_tensors)):
        data_name = args.image_name_dict['test'][dtype][i]
        data = line_tensors[i]
        # input image
        file_prefix = str(config.root / ".." / data_name[0]).split(".data0.json")[0]
        image_file = file_prefix + ".image.png"
        input_image = get_cv_image(image_file)

        # group prediction
        group_pred_image = visual_group_predictions(args, data, input_image, colors, thickness,
                                                    config.group_tensor_index)
        final_image_filename = str(
            args.image_output_path / f"gp_{data_type}_{data_name[0].split('/')[-1].split('.data0.json')[0]}.output.png")

        save_image(group_pred_image, final_image_filename)


def visual_groups(args, group_tensors, data_type):
    colors = [
        (255, 0, 0),  # Blue
        (255, 255, 0),  # Cyan
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (0, 255, 255),  # Yellow
    ]
    thickness = 3

    if "pos" in data_type:
        dtype = "true"
    else:
        dtype = "false"

    for i in range(len(group_tensors)):
        data_name = args.image_name_dict['test'][dtype][i]
        data = group_tensors[i]
        # input image
        file_prefix = str(config.root / ".." / data_name[0]).split(".data0.json")[0]
        image_file = file_prefix + ".image.png"
        input_image = get_cv_image(image_file)

        # group prediction
        group_pred_image = visual_group_predictions(args, data, input_image, colors, thickness,
                                                    config.group_tensor_index)
        final_image_filename = str(
            args.image_output_path / f"gp_{data_type}_{data_name[0].split('/')[-1].split('.data0.json')[0]}.output.png")

        save_image(group_pred_image, final_image_filename)