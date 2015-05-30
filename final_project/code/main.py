import cv2
import cv2.cv as cv
import numpy as np
import math
import fnmatch
import os
import re
from scipy.signal import argrelextrema
import scipy.fftpack

import matplotlib.pyplot as plt


landmarks_dir = "../data/Landmarks/original/"
radiographs_dir = "../data/Radiographs/"


def main():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x, eps=10**-14), landmarks_training_data)
    pca_result = map(lambda x: pca(np.array(x)), gpa_landmarks)

    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 600, 800)
    original_radiographs = read_radiographs(radiographs_dir)
    pre_processed_radiographs = map(pre_process_radiograph, original_radiographs)
    map(lambda x: segment_teeth2(x, 100), pre_processed_radiographs)


def read_radiographs(radiographs_dir):
    radiographs_list = []
    for file_name in sorted(os.listdir(radiographs_dir)):
        if fnmatch.fnmatch(file_name, '*.tif'):
            img = cv2.imread(radiographs_dir + '/' + file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            radiographs_list.append(img)
    return radiographs_list


def pre_process_radiograph(radiograph, show=False):
    crop_p = 0.1

    radiograph = cv2.equalizeHist(radiograph)
    if show:
        cv2.imshow('window', radiograph)
        cv2.waitKey()

    height, width = radiograph.shape
    cropped_img = radiograph[crop_p*height:(1-crop_p)*height, crop_p*width:(1-crop_p)*width]

    blur_result = cv2.medianBlur(cropped_img, 17)
    blur_result = cv2.GaussianBlur(blur_result, (5, 5), 3)
    if show:
        cv2.imshow('window', blur_result)
        cv2.waitKey()

    otsu_threshold, _ = cv2.threshold(blur_result, 0, 255, cv2.THRESH_OTSU)
    canny_result = cv2.Canny(blur_result, 0.2*otsu_threshold, 0.5*otsu_threshold)
    if show:
        cv2.imshow('window', canny_result)
        cv2.waitKey()

    dilate_result = cv2.dilate(canny_result, (21, 21), iterations=10)
    if show:
        cv2.imshow('window', dilate_result)
        cv2.waitKey()

    it_thresh_result = iterative_thresholding(cropped_img, dilate_result, 0.8)
    if show:
        cv2.imshow('window', it_thresh_result)
        cv2.waitKey()

    ad_thresh_result = cv2.adaptiveThreshold(it_thresh_result, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY, 151, 10)
    if show:
        cv2.imshow('window', ad_thresh_result)
        cv2.waitKey()

    result = cv2.bitwise_and(it_thresh_result, ad_thresh_result)
    if show:
        cv2.imshow('window', result)
        cv2.waitKey()

    return result


def iterative_thresholding(radiograph, mask, teeth_pref=0.5):
    threshold = cv2.mean(radiograph, mask)[0]
    while True:
        non_teeth_mean = cv2.mean(radiograph, cv2.threshold(radiograph, threshold, 255, cv2.THRESH_BINARY_INV)[1])[0]
        teeth_mean = cv2.mean(radiograph, cv2.threshold(radiograph, threshold, 255, cv2.THRESH_BINARY)[1])[0]
        prev_threshold = threshold
        threshold = teeth_pref*non_teeth_mean + (1-teeth_pref)*teeth_mean
        if prev_threshold == threshold:
            break
    return cv2.threshold(radiograph, threshold, 1, cv2.THRESH_TOZERO)[1]


def create_landmarks_data(file_dir):
    landmarks_training_data = [[0]*(len(os.listdir(file_dir))/8) for _ in range(0, 8)]
    for file_name in sorted(os.listdir(file_dir)):
        if fnmatch.fnmatch(file_name, '*.txt'):
            incisor_landmarks = parse_landmarks_file(file_name, file_dir)
            match = re.match(".*?(\d+)-(\d+)\.txt", file_name)
            landmarks_training_data[int(match.group(2))-1][int(match.group(1))-1] = incisor_landmarks
    return landmarks_training_data

def segment_teeth2(radiograph, interval):
    height, width = radiograph.shape
    hist = []
    minimal_points = []
    for e, i in enumerate(range(interval, width, interval)):
        #generating histogram
        hist.append([])
        for j in range(0, height, 1):
            hist[e].append((np.sum(radiograph[j][i-interval:i+interval+1]), i, j))

        #smoothing
        w = scipy.fftpack.rfft(map(lambda (intensity, s, t): intensity, hist[e]))
        w[30:] = 0
        smoothed = scipy.fftpack.irfft(w)
        # plt.plot(hist[e])
        # plt.plot(smoothed)
        # plt.show()
        #finding mimima and sort them
        indices = argrelextrema(smoothed, np.less)[0]
        minimal_points_width = []
        for idx in indices:
            minimal_points_width.append(hist[e][idx])
        minimal_points_width.sort()

        #keep the best 3 local minima which lie atleast 200 apart from other point
        count = 0
        to_keep = []
        for p in range(len(minimal_points_width)):
            _, _, d = minimal_points_width[p]
            add = True
            for _, _, b in to_keep:
                if (abs(b-d) < 200 and abs(b-d) != 0) or count >= 3:
                    add = False
            if add:
                count += 1
                to_keep.append(minimal_points_width[p])
        minimal_points.extend(to_keep)
        # minimal_points.extend(minimal_points_width[0:4])

    #plotting
    for v, x, y in minimal_points:
        cv2.circle(radiograph, (x, y), 1, 255, 10)
    cv2.line(radiograph, (400,400),(400,600),255,10)
    cv2.imshow('window', radiograph)
    cv2.waitKey()


def segment_teeth(radiograph, interval):
    height, width = radiograph.shape
    horizontal_line = [0]*width
    hist = [0]*height
    j = width/2
    inter = interval
    for i in range(height):
        hist[i] += (np.sum(radiograph[i, j-inter:j+inter+1]))
    cv2.imshow('window', radiograph)
    plt.plot(hist)
    plt.show()
    # for j in range(width/2, width-interval, 50):
    #     hist = [0]*height
    #     for i in range(height):
    #         hist[i] = (np.sum(radiograph[i, j-interval:j+interval+1])+(abs(height/2-i)))
    #     cv2.imshow('window', radiograph)
    #     plt.plot(hist)
    #     plt.show()



def parse_landmarks_file(file_name, file_dir):
    f = open(file_dir+'/'+file_name, 'r')
    return np.array(map(float, f.readlines()))


def generalized_procrustes_analysis(landmarks_list, eps):
    """
    Performing the generalized Procrustes analysis for the given landmarks list
    """
    reference_mean = normalize_landmarks(landmarks_list[0])
    landmarks_list = map(normalize_landmarks, landmarks_list)
    weight_vector = calculate_weight_vector(landmarks_list)
    landmarks_list = map(lambda x: np.multiply(weight_vector, x), landmarks_list)
    while True:
        landmarks_list = map(lambda x: procrustes_analysis(reference_mean, x), landmarks_list)
        previous_mean = reference_mean
        reference_mean = mean_landmarks_normalized(landmarks_list)
        if rmse(reference_mean, previous_mean) < eps:
            break
    landmarks_list = map(lambda x: normalize_landmarks(np.divide(x, weight_vector)), landmarks_list)
    return landmarks_list


def procrustes_analysis(reference_landmarks, landmarks):
    """
    Performing the Procrcustes analysis for the given landmarks
    """
    mean = get_mean_landmarks(reference_landmarks)
    translated_landmarks = translate_landmarks(landmarks, mean)
    scaled_landmarks = scale_landmarks(translated_landmarks)
    theta = find_optimal_rotation(reference_landmarks, scaled_landmarks)
    rotated_landmarks = rotate_landmarks(scaled_landmarks, theta)
    return rotated_landmarks


def calculate_weight_vector(landmarks_list):
    weight_vector = []
    for i in range(len(landmarks_list[0])/2):
        sum_variances = 0
        for j in range(len(landmarks_list[0])/2):
            distances = []
            for landmarks in landmarks_list:
                x1, y1 = landmarks[2*i], landmarks[2*i+1]
                x2, y2 = landmarks[2*j], landmarks[2*j+1]
                distances.append(math.sqrt((x1-x2)**2+(y1-y2)**2))
            sum_variances += np.var(np.array(distances))
        weight_vector.append(sum_variances**-1)
        weight_vector.append(sum_variances**-1)
    return np.array(weight_vector)


def find_optimal_rotation(reference_landmarks, landmarks):
    """
    Returns the optimal rotation to minimize the root mean square error between the corresponding points
    in the given landmarks
    """
    nominator = 0.0
    denominator = 0.0
    for i in range(0, len(landmarks), 2):
        nominator += landmarks[i] * reference_landmarks[i+1] - landmarks[i+1] * reference_landmarks[i]
        denominator += landmarks[i] * reference_landmarks[i] + landmarks[i+1] * reference_landmarks[i+1]
    return math.atan(nominator / denominator)


def mean_landmarks_normalized(landmarks_list):
    """
    Calculates the mean landmarks vector for the vectors in the given list and normalizes it
    """
    sum = np.copy(landmarks_list[0])
    for i in range(1, len(landmarks_list)):
        sum += landmarks_list[i]
    mean = sum / len(landmarks_list)
    return normalize_landmarks(mean)


def rmse(vector, mean):
    """
    Return the root mean squared error of the two given numpy arrays
    """
    return np.sqrt(np.sum((vector-mean)**2))


def normalize_landmarks(landmarks):
    """
    Normalizes the given landmarks vector by translating and scaling it
    """
    translated_landmarks = translate_landmarks(landmarks, (0, 0))
    scaled_landmarks = scale_landmarks(translated_landmarks)
    return scaled_landmarks


def translate_landmarks(landmarks, origin):
    """
    Translates the given landmark vector so that its mean equals the given origin
    """
    mean_x, mean_y = get_mean_landmarks(landmarks)
    tx = origin[0] - mean_x
    ty = origin[1] - mean_y

    translated_landmarks = []
    for i, landmark in enumerate(landmarks):
        if i % 2 == 0:
            translated_landmarks.append(landmark + tx)
        else:
            translated_landmarks.append(landmark + ty)
    return np.array(translated_landmarks)


def scale_landmarks(landmarks):
    """
    Scales the given landmark vector so that the root mean squared distance is equal to 1
    """
    x_landmarks, y_landmarks = separate_landmarks(landmarks)
    mean_x, mean_y = get_mean_landmarks(landmarks)

    s = np.sqrt((np.sum((x_landmarks - mean_x)**2) + np.sum((y_landmarks - mean_y)**2)) / len(x_landmarks))

    scaled_landmarks = []
    for i, landmark in enumerate(landmarks):
        if i % 2 == 0:
            scaled_landmarks.append((landmark - mean_x) / s)
        else:
            scaled_landmarks.append((landmark - mean_y) / s)
    return np.array(scaled_landmarks)


def rotate_landmarks(landmarks, theta):
    """
    Rotates the given landmarks by the given angle theta
    """

    rotated_landmarks = []
    for i in range(0, len(landmarks), 2):
        rotated_landmarks.append(math.cos(theta) * landmarks[i] - math.sin(theta) * landmarks[i+1])
        rotated_landmarks.append(math.sin(theta) * landmarks[i] + math.cos(theta) * landmarks[i+1])
    return np.array(rotated_landmarks)


def get_mean_landmarks(landmarks):
    """
    Returns the mean x- and y-coordinate of the given landmark
    """
    x_landmarks, y_landmarks = separate_landmarks(landmarks)
    mean_x = np.mean(x_landmarks)
    mean_y = np.mean(y_landmarks)
    return mean_x, mean_y


def separate_landmarks(landmarks):
    """
    Separates the given landmarks vector in x- and y-coordinates
    """
    x_landmarks = [x for i, x in enumerate(landmarks) if i % 2 == 0]
    y_landmarks = [y for i, y in enumerate(landmarks) if i % 2 == 1]
    return np.array(x_landmarks), np.array(y_landmarks)


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


def pca(X, num_components=0):
    """
    This method is copied from the supplied solution of project 3 of the Computer Vision course
    """
    [n,d] = X.shape
    if (num_components <= 0) or (num_components>n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[:, i]/np.linalg.norm(eigenvectors[:, i])
    # or simply perform an economy size decomposition
    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


if __name__ == '__main__':
    main()