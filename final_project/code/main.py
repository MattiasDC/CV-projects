import cv2
import cv2.cv as cv
import numpy as np
import math
import fnmatch
import os
import re


landmarks_dir = "../data/Landmarks/original/"
radiographs_dir = "../data/Radiographs/"


def main():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x, eps=10**-14), landmarks_training_data)


def create_landmarks_data(file_dir):
    landmarks_training_data = [[] for _ in range(0, 8)]
    for file_name in os.listdir(file_dir):
        if fnmatch.fnmatch(file_name, '*.txt'):
            incisor_landmarks = parse_landmarks_file(file_name, file_dir)
            incisor_id = int(re.match(".*-(\d+)\.txt", file_name).group(1))
            landmarks_training_data[incisor_id-1].append(incisor_landmarks)
    return landmarks_training_data


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
        return np.dot(X,W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
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
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
    # or simply perform an economy size decomposition
    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


if __name__ == '__main__':
    main()