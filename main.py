import math
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from UZ_utils import *
from a3_utils import *
from a4_utils import *


def subplot_template(images, titles, height, width, cmap):
    plt.figure(figsize=(12, 8))
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()


def eucledian_distance(v1, v2):
    i = (v1 - v2) ** 2
    distance = i.sum()
    distance = np.sqrt(distance)

    return distance


def hell_distance(v1, v2):
    i = (np.sqrt(v1) - np.sqrt(v2)) ** 2
    distance = i.sum()
    distance = distance / 2
    distance = math.sqrt(distance)

    return distance


def partial_derivatives(original_image, sigma):
    g_x = gauss(sigma)
    g_y = g_x.T
    g_xx = gaussdx(sigma)
    g_yy = g_xx.T
    I_x = cv2.filter2D(cv2.filter2D(original_image, -1, -g_y), -1, g_xx)
    I_y = cv2.filter2D(cv2.filter2D(original_image, -1, g_x), -1, -g_yy)
    I_xx = cv2.filter2D(cv2.filter2D(I_x, -1, -g_y), -1, g_xx)
    I_xy = cv2.filter2D(cv2.filter2D(I_x, -1, g_x), -1, -g_yy)
    I_yy = cv2.filter2D(cv2.filter2D(I_y, -1, g_x), -1, -g_yy)

    return I_x, I_y, I_xx, I_xy, I_yy


def non_maxima(matrix):
    padded_matrix = np.pad(matrix, pad_width=1, mode='constant', constant_values=float('-inf'))
    neighborhoods = np.lib.stride_tricks.sliding_window_view(padded_matrix, (3, 3))
    max_values = np.max(neighborhoods, axis=(2, 3))
    matrix[matrix < max_values] = 0
    return matrix


def hessian_points(image, sigma):
    _, _, I_xx, I_xy, I_yy = partial_derivatives(image, sigma)
    det = I_xx * I_yy - I_xy ** 2
    return det


def hessian_points_extended(image, sigma):
    det = hessian_points(image, sigma)
    det = non_maxima(det)
    return det


def show_hessian(image, sigmas, th):
    images_hessian = []
    keypoint_images = []

    for i in range(len(sigmas)):
        image_h = hessian_points(image, sigmas[i])
        images_hessian.append(image_h)
        image_h_nm = non_maxima(image_h.copy())
        keypoints = np.argwhere(image_h_nm > th)
        keypoint_images.append(keypoints)

    plt.figure(figsize=(12, 8))
    for i in range(len(sigmas)):
        plt.subplot(2, len(sigmas), i + 1)
        plt.imshow(images_hessian[i])
        plt.title("sigma = {}".format(sigmas[i]))
    for i in range(len(sigmas)):
        plt.subplot(2, len(sigmas), i + len(sigmas) + 1)
        plt.imshow(image, cmap='gray')
        plt.scatter(keypoint_images[i - len(sigmas)][:, 1], keypoint_images[i - len(sigmas)][:, 0], c='red', marker='x',
                    s=30, linewidths=1)

    plt.tight_layout()
    plt.show()


def correlation_matrix(image, sigma_der, sigma_gauss):
    I_x, I_y, _, _, _ = partial_derivatives(image, sigma_der)

    I_x_squared = I_x ** 2
    I_y_squared = I_y ** 2
    I_x_y = I_x * I_y

    kernel = cv2.getGaussianKernel(2 * math.ceil(3 * sigma_gauss) + 1, sigma_gauss)
    kernel = kernel * kernel.T
    S_x_squared = cv2.filter2D(I_x_squared, -1, kernel)
    S_y_squared = cv2.filter2D(I_y_squared, -1, kernel)
    S_x_y = cv2.filter2D(I_x_y, -1, kernel)

    det_C = (S_x_squared * S_y_squared) - (S_x_y ** 2)
    tr_C = S_x_squared + S_y_squared

    return det_C, tr_C


def harris_detector(image, sigma_der, sigma_gauss, alpha, th):
    det_C, tr_C = correlation_matrix(image, sigma_der, sigma_gauss)
    det_C = non_maxima(det_C.copy())
    tr_C = non_maxima(tr_C.copy())
    harris = det_C - alpha * (tr_C ** 2)
    keypoints = np.argwhere(harris > th)
    return keypoints


def show_harris(image, sigmas_der, sigmas_gauss, alpha, th):
    corr_matrices = []
    keypoint_images = []

    for i in range(len(sigmas_der)):
        det_C, tr_C = correlation_matrix(image, sigmas_der[i], sigmas_gauss[i])
        corr_matrices.append(det_C)
        det_C = non_maxima(det_C.copy())
        tr_C = non_maxima(tr_C.copy())

        harris = det_C - alpha * (tr_C ** 2)

        keypoints = np.argwhere(harris > th)
        keypoint_images.append(keypoints)

    plt.figure(figsize=(12, 8))
    for i in range(len(sigmas_der)):
        plt.subplot(2, len(sigmas_der), i + 1)
        plt.imshow(corr_matrices[i])
        plt.title("sigma = {}".format(sigmas_der[i]))
    for i in range(len(sigmas_der)):
        plt.subplot(2, len(sigmas_der), i + len(sigmas_der) + 1)
        plt.imshow(image, cmap='gray')
        plt.scatter(keypoint_images[i - len(sigmas_der)][:, 1], keypoint_images[i - len(sigmas_der)][:, 0], c='red',
                    marker='x',
                    s=30, linewidths=1)

    plt.tight_layout()
    plt.show()


def find_correspondences(descriptor1, descriptor2):
    distances = []
    for i in range(descriptor1.shape[0]):
        min_dist = np.Inf
        min_ix = 0
        for j in range(descriptor2.shape[0]):
            distance = hell_distance(descriptor1[i], descriptor2[j])
            if distance < min_dist:
                min_dist = distance
                min_ix = j
        distances.append((i, min_ix))
    return distances


def find_matches(image1, image2, show):
    image1_keypoints = harris_detector(image1, 3, 3 * 1.6, 0.06, 0.00001)
    image2_keypoints = harris_detector(image2, 3, 3 * 1.6, 0.06, 0.00001)

    y_array_a = image1_keypoints[:, 0]
    x_array_a = image1_keypoints[:, 1]

    y_array_b = image2_keypoints[:, 0]
    x_array_b = image2_keypoints[:, 1]

    desc_a = simple_descriptors(image1, y_array_a, x_array_a)
    desc_b = simple_descriptors(image2, y_array_b, x_array_b)

    corr = find_correspondences(desc_a, desc_b)
    pts1 = []
    pts2 = []
    for x, y in corr:
        pts1.append(image1_keypoints[x])
        pts2.append(image2_keypoints[y])

    pts1 = np.fliplr(pts1)
    pts2 = np.fliplr(pts2)

    if show:
        display_matches(image1, pts1, image2, pts2)

    return pts1, pts2


def find_symmetrical_matches(image1, image2, show):
    pts1, pts2 = find_matches(image1, image2, False)
    pts1b, pts2b = find_matches(image2, image1, False)

    length = min(len(pts1), len(pts1b))
    pts1_final = []
    pts2_final = []
    for i in range(length):
        (x1a, y1a) = pts1[i]
        (x2a, y2a) = pts2[i]
        for j in range(len(pts1b)):
            (x2b, y2b) = pts1b[j]
            (x1b, y1b) = pts2b[j]
            if ((x2a, y2a) == (x2b, y2b)) and ((x1a, y1a) == (x1b, y1b)):
                pts1_final.append((x1a, y1a))
                pts2_final.append((x2a, y2a))

    if show:
        display_matches(image1, pts1_final, image2, pts2_final)

    return pts1_final, pts2_final


def estimate_homography(pts1, pts2):
    """
[(21.0, 96.0), (246.0, 94.0), (25.0, 185.0), (186.0, 207.0)]
[(64.0, 52.0), (238.0, 195.0), (10.0, 122.0), (121.0, 243.0)]


    [21 96 1    0  0 0 -64*21   -64*96  -64
     0  0  0   21 96 1 -52*21   -52*96  -52
    246 94 1   0  0  0 -238*246 -238*94 -238
     0  0  0 246 94  1 -195*246 -195*94 -195
    25 185 1   0  0  0 -10*25   -10*185 -10
     0  0  0  25 185 1 -122*25 -122*185 -122
    186 207 1   0  0 0 -121*186 -121*207 -121
     0  0  0 186 207 1 -243*186 -243*207 -243]
    """
    A = np.zeros((8, 9))

    for i in range(0, 7, 2):
        A[i][0] = pts1[int(i / 2)][0]
        A[i][1] = pts1[int(i / 2)][1]
        A[i][2] = 1
        A[i][3] = 0
        A[i][4] = 0
        A[i][5] = 0
        A[i][6] = (-1) * pts2[int(i / 2)][0] * pts1[int(i / 2)][0]
        A[i][7] = (-1) * pts2[int(i / 2)][0] * pts1[int(i / 2)][1]
        A[i][8] = (-1) * pts2[int(i / 2)][0]

        A[i + 1][0] = 0
        A[i + 1][1] = 0
        A[i + 1][2] = 0
        A[i + 1][3] = pts1[int(i / 2)][0]
        A[i + 1][4] = pts1[int(i / 2)][1]
        A[i + 1][5] = 1
        A[i + 1][6] = (-1) * pts2[int(i / 2)][1] * pts1[int(i / 2)][0]
        A[i + 1][7] = (-1) * pts2[int(i / 2)][1] * pts1[int(i / 2)][1]
        A[i + 1][8] = (-1) * pts2[int(i / 2)][1]

    # print(A.astype(int))
    A = A.astype(int)
    U, S, VT = np.linalg.svd(A)
    # print(VT.shape)
    h = VT[-1] / VT[-1][-1]
    H = h.reshape((3, 3))
    return H


def show_transformation(image1, image2, points, pts1, pts2, matrix):
    if matrix:
        pts1 = []
        pts2 = []
        for i in range(4):
            pts1.append((points[i][0], points[i][1]))
            pts2.append((points[i][2], points[i][3]))

    display_matches(image1, pts1, image2, pts2)

    H = estimate_homography(pts1, pts2)

    ny_homography = cv2.warpPerspective(image1, H, (image1.shape[1], image1.shape[0]))
    plt.subplot(1, 2, 1)
    plt.imshow(image2, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(ny_homography, cmap='gray')
    plt.tight_layout()
    plt.show()


def ransac(image1, image2, th, k):
    pts1, pts2 = find_symmetrical_matches(image1, image2, False)
    selected_indexes = random.sample(range(len(pts1)), k=4)
    pts1_best = []
    pts2_best = []
    max_no_of_inliers = -np.inf
    for j in range(k):
        pts1_selected = [pts1[i] for i in selected_indexes]
        # pts1_selected = [(231, 131), (15, 97), (155, 124), (223, 163)]
        pts2_selected = [pts2[i] for i in selected_indexes]
        # pts2_selected = [(203, 214), (58, 49), (149, 160), (176, 234)]
        number_of_inliers = 0

        H = estimate_homography(pts1_selected, pts2_selected)
        reprojection_errors = np.zeros(4)

        for i in range(4):
            homog_c = pts1_selected[i] + (1,)
            v1 = np.dot(H, homog_c)
            v2 = pts2_selected[i] + (1,)
            reprojection_errors[i] = eucledian_distance(v1, v2)
            if reprojection_errors[i] < th:
                number_of_inliers += 1

        # final_error = sum(reprojection_errors) / len(reprojection_errors)
        # print(final_error)

        if number_of_inliers > max_no_of_inliers:
            max_no_of_inliers = number_of_inliers
            pts1_best = pts1_selected
            pts2_best = pts2_selected

    # print(pts1_best, pts2_best)
    return pts1_best, pts2_best


def exercise1():
    # a) Implementing a function 'hessian_points' and testing it on 'graf_a.jpg'
    image = imread_gray('data/graf/graf_a.jpg')
    image_h = hessian_points(image, 3)
    # imshow(image_h)

    # Extending the function hessian_points by implementing a non-maximum suppression post-processing step
    image_h2 = hessian_points_extended(image, 3)
    # imshow(image_h2)

    # Creating a function that plots the detected points

    sigmas = [3, 6, 9]
    th = 0.004
    show_hessian(image, sigmas, th)

    # b) Implementing the Harris feature point detector
    sigmas = [3, 6, 9]
    sigmas_gauss = np.array(sigmas)
    sigmas_gauss = sigmas_gauss * 1.6
    alpha = 0.06
    show_harris(image, sigmas, sigmas_gauss, alpha, 0.00001)


def exercise2():
    # a)
    graf_a = imread_gray('data/graf/graf_a_small.jpg')
    graf_b = imread_gray('data/graf/graf_b_small.jpg')
    find_matches(graf_a, graf_b, True)

    # b)

    graf_a_small = imread_gray('data/graf/graf_a_small.jpg')
    graf_b_small = imread_gray('data/graf/graf_b_small.jpg')
    pts1_final, pts2_final = find_symmetrical_matches(graf_a_small, graf_b_small, True)


def exercise3():
    # a) Writing function estimate_homography, that approximates a homography between two images using a given set of
    # matched feature points
    ny_a = imread_gray('data/newyork/newyork_a.jpg')
    ny_b = imread_gray('data/newyork/newyork_b.jpg')
    ny_points = np.loadtxt('data/newyork/newyork.txt')
    show_transformation(ny_a, ny_b, ny_points, [], [], True)
    graf_a = imread_gray('data/graf/graf_a.jpg')
    graf_b = imread_gray('data/graf/graf_b.jpg')
    graf_points = np.loadtxt('data/graf/graf.txt')
    show_transformation(graf_a, graf_b, graf_points, [], [], True)

    # b)
    k = 100
    th = 3
    pts1_best, pts2_best = ransac(ny_a, ny_b, th, k)
    show_transformation(ny_a, ny_b, [], pts1_best, pts2_best, False)


if __name__ == '__main__':
    exercise1()
    exercise2()
    exercise3()
