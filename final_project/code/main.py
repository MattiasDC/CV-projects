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


# Directories
landmarks_dir = "../data/Landmarks/original/"
radiographs_dir = "../data/Radiographs/"

# Precision procrustes
eps = 10**-12

# Parameters for mapping model points to segment
locality_search = 20
wide_search = 30


def main():
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 600, 800)
    original_radiographs = read_radiographs(radiographs_dir)
    cropped_radiographs = map(crop_radiograph, original_radiographs)

    pre_processed_radiographs = map(pre_process_radiograph, cropped_radiographs)
    # map(lambda x: segment_teeth(x, 50), pre_processed_radiographs)

    landmarks_preprocessed = map(pre_process_for_landmarks, original_radiographs)
    cropped_landmarks_preprocessed = map(crop_radiograph, landmarks_preprocessed)

    landmarks_training_data = create_landmarks_data(landmarks_dir)
    landmarks_neigbourhoods = map(lambda x: learn_landmark_neighbourhood(x, landmarks_preprocessed),
                                  landmarks_training_data)
    models = get_models(landmarks_training_data)

    f(cropped_landmarks_preprocessed, models, landmarks_neigbourhoods, pre_processed_radiographs)


def f(cropped_landmarks_preprocessed, models, landmarks_neigbourhoods, pre_processed_radiographs):
    print "Which picture? (number between 0 and %d)" % (len(cropped_landmarks_preprocessed)-1)
    picture = int(raw_input())
    segment = get_segment(pre_processed_radiographs[picture], cropped_landmarks_preprocessed[picture])
    for i in range(len(models)):
        vector = fit_model(models[i], landmarks_neigbourhoods[i], segment, cropped_landmarks_preprocessed[picture])
        d = np.sum(np.divide(vector**2, models[i][3]))
        print "model %d, d %f, dmax %f, result: %s" % (i, d, models[i][2], str(vector))
    f(cropped_landmarks_preprocessed, models, landmarks_neigbourhoods, pre_processed_radiographs)


# =====================================================
#            RADIOGRAPHS
# =====================================================


def read_radiographs(radiographs_dir):
    radiographs_list = []
    for file_name in sorted(os.listdir(radiographs_dir)):
        if fnmatch.fnmatch(file_name, '*.tif'):
            img = cv2.imread(radiographs_dir + '/' + file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            radiographs_list.append(img)
    return radiographs_list


def crop_radiograph(radiograph, percentage=0.1):
    height, width = radiograph.shape
    return radiograph[percentage*height:(1-percentage)*height, percentage*width:(1-percentage)*width]


def pre_process_radiograph(radiograph, show=False):
    """
    Pre-processes the given radiograph before segmentation
    """
    # radiograph = cv2.equalizeHist(radiograph)
    if show:
        cv2.imshow('window', radiograph)
        cv2.waitKey()

    blur_result = cv2.medianBlur(radiograph, 17)
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

    it_thresh_result = iterative_thresholding(radiograph, dilate_result, 0.8)
    if show:
        cv2.imshow('window', it_thresh_result)
        cv2.waitKey()

    ad_thresh_result = cv2.adaptiveThreshold(it_thresh_result, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY, 301, 10)
    if show:
        cv2.imshow('window', ad_thresh_result)
        cv2.waitKey()

    result = cv2.bitwise_and(it_thresh_result, ad_thresh_result)
    if show:
        cv2.imshow('window', result)
        cv2.waitKey()

    return result


def iterative_thresholding(radiograph, mask, teeth_pref=0.5):
    """
    Finds a treshold to divide the radiograph in teeth and non-teeth,
    using the intensity for the white parts of the given mask as initial treshold
    """
    threshold = cv2.mean(radiograph, mask)[0]
    while True:
        non_teeth_mean = cv2.mean(radiograph, cv2.threshold(radiograph, threshold, 255, cv2.THRESH_BINARY_INV)[1])[0]
        teeth_mean = cv2.mean(radiograph, cv2.threshold(radiograph, threshold, 255, cv2.THRESH_BINARY)[1])[0]
        prev_threshold = threshold
        threshold = teeth_pref*non_teeth_mean + (1-teeth_pref)*teeth_mean
        if prev_threshold == threshold:
            break
    return cv2.threshold(radiograph, threshold, 1, cv2.THRESH_TOZERO)[1]


def path_intensity(radiograph, path):
    prev = path[0]
    intensity = 0
    for i in range(1, len(path)):
        intensity += line_intensity(radiograph, prev, path[i])
        prev = path[i]
    return intensity


def line_intensity(radiograph, p1, p2):
    li = cv.InitLineIterator(cv.fromarray(radiograph), p1, p2)
    intensity = 0
    for i in li:
        intensity += i
    return intensity


def remove_short_paths(paths, width, ratio):
    """
    Removes all paths smaller than width*ratio
    """
    minimal_length = width*ratio
    to_keep = []
    for path in paths:
        length = path_length(path)
        if length >= minimal_length:
            to_keep.append(path)
    return to_keep


def path_length(path):
    prev = path[0]
    length = 0
    for i in range(1, len(path)):
        length += math.sqrt((prev[0]-path[i][0])**2+(prev[1]-path[i][1])**2)
        prev = path[i]
    return length


def trim_path(radiograph, path):
    """
    Trims the outer parts of the path based on the average intensity of the line segments
    """
    mean_intensity = path_intensity(radiograph, path)/path_length(path)
    while len(path) > 2:
        if mean_intensity > line_intensity(radiograph, path[0], path[1])/path_length([path[0], path[1]]):
            del(path[0])
        else:
            break
    while len(path) > 2:
        if mean_intensity > line_intensity(radiograph, path[-1], path[-2])/path_length([path[-1], path[-2]]):
            del(path[-1])
        else:
            break
    return path


def draw_path(radiograph, path, color=255):
    prev = path[0]
    for i in range(1, len(path)):
        cv2.line(radiograph, prev, path[i], color, 5)
        prev = path[i]


# =====================================================
#            SEGMENT
# =====================================================


def get_segment(preprocessed_radiograph, radiograph):
    """
    Used to retrieve a tooth segment by manually drawing a box
    """
    tmp_radiograph = preprocessed_radiograph.copy()
    cv2.imshow('window', tmp_radiograph)
    clicks = []

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            cv2.circle(tmp_radiograph, (x, y), 1, 255, 10)
            clicks.append((x, y))

    cv2.setMouseCallback('window', mouse_event)
    while len(clicks) < 4:
        cv2.imshow('window', tmp_radiograph)
        cv2.waitKey(20)

    (x, y), (w, h), angle = cv2.minAreaRect(np.array(clicks))

    box = cv2.cv.BoxPoints(((x, y), (w, h), angle))
    box = np.array(map(lambda (x, y): (np.int(x), np.int(y)), box))
    cv2.drawContours(tmp_radiograph, [box], 0, 255, 10)
    cv2.imshow('window', tmp_radiograph)
    cv2.waitKey()

    if angle < -45:
        angle += 90
        tmp = w
        w = h
        h = tmp

    # m = cv2.getRotationMatrix2D((int(x), int(y)), angle, 1)
    # r_h, r_w = radiograph.shape
    # rotated = cv2.warpAffine(radiograph, m, (r_w, r_h), flags=cv2.INTER_CUBIC)
    # return cv2.getRectSubPix(rotated, (int(w), int(h)), (int(x), int(y)))

    return (int(w), int(h)), (int(x), int(y))


def segment_teeth(radiograph, interval):
    """
    Attempts to segment the teeth on the radiograph
    """
    # TODO vertical segmentation
    height, width = radiograph.shape
    _, mask = cv2.threshold(radiograph, 0, 1, cv2.THRESH_BINARY)
    mask = 255-radiograph

    hist = []
    minimal_points = []
    if width % 2 == 0:
        mask = mask[:, :-1]
    width = mask.shape[1]
    filter = gaussian_filter(450, width)
    mask = np.multiply(mask, filter)
    mask2 = mask * (255/mask.max())
    mask2 = mask2.astype('uint8')

    for e, i in enumerate(range(interval, width, interval)):
        #generating histogram
        hist.append([])
        for j in range(0, height, 1):
            hist[e].append((np.sum(mask[j][i-interval:i+interval+1]), i, j))

        #smoothing
        w = scipy.fftpack.rfft(map(lambda (intensity, s, t): intensity, hist[e]))
        w[30:] = 0
        smoothed = scipy.fftpack.irfft(w)

        #finding mimima and sort them
        indices = argrelextrema(smoothed, np.greater)[0]
        minimal_points_width = []
        for idx in indices:
            minimal_points_width.append(hist[e][idx])
        minimal_points_width.sort(reverse=True)

        #keep the best 3 local minima which lie atleast 200 apart from other point
        count = 0
        to_keep = []
        for p in range(len(minimal_points_width)):
            _, _, d = minimal_points_width[p]
            add = True
            for _, _, b in to_keep:
                if (abs(b-d) < 150 and abs(b-d) != 0) or count >= 4:
                    add = False
            if add:
                count += 1
                to_keep.append(minimal_points_width[p])
        minimal_points.extend(to_keep)
        # minimal_points.extend(minimal_points_width[0:4])

    edges = []
    for _, x, y in minimal_points:
        minimal_intensity = 20000000
        minimal_coords = -1, -1
        for _, u, v in minimal_points:
            cache = line_intensity(mask, (x, y), (u, v))
            if x >= u or cache >= minimal_intensity or abs(v-y) > 0.1*height:
                continue
            minimal_intensity = cache
            minimal_coords = (u, v)
        if minimal_coords != (-1, -1):
            edges.append(((x, y), (minimal_coords[0], minimal_coords[1])))

    paths = []
    for edge in edges:
        new_path = True
        for path in paths:
            if path[-1] == edge[0]:
                new_path = False
                path.append(edge[1])
        if new_path:
            paths.append([edge[0], edge[1]])

    draw_image = mask2.copy()

    paths = map(lambda x: trim_path(mask2, x), paths)
    paths = remove_short_paths(paths, width, 0.3)
    best_path = sorted(map(lambda x: (path_intensity(radiograph, x)/(path_length(x)), x), paths))[0][1]
    map(lambda x: draw_path(draw_image, x, color=150), paths)
    map(lambda x: cv2.putText(draw_image, str(int(path_intensity(radiograph, x)/(path_length(x)))), x[0], cv.CV_FONT_HERSHEY_PLAIN, 5, 255), paths)
    draw_path(draw_image, best_path)

    #plotting
    for v, x, y in minimal_points:
        cv2.circle(draw_image, (x, y), 1, 150, 10)
    cv2.imshow('window', draw_image)
    cv2.waitKey()


def gaussian_function(sigma, u):
    return 1/(math.sqrt(2*math.pi)*sigma)*math.e**-(u**2/(2*sigma**2))


def gaussian_filter(sigma, filter_length=None):
    '''
    Given a sigma, return a 1-D Gaussian filter.
    @param     sigma:         float, defining the width of the filter
    @param     filter_length: optional, the length of the filter, has to be odd
    @return    A 1-D numpy array of odd length,
               containing the symmetric, discrete approximation of a Gaussian with sigma
               Summation of the array-values must be equal to one.
    '''
    if filter_length is None:
        #determine the length of the filter
        filter_length = math.ceil(sigma*5)
        #make the length odd
        filter_length = 2*(int(filter_length)/2) + 1

    #make sure sigma is a float
    sigma = float(sigma)

    #create the filter
    result = np.zeros(filter_length)

    #do your best!
    result = np.asarray(map(lambda u: gaussian_function(sigma, u), range(-(filter_length/2), filter_length/2 + 1, 1)))

    result = result / result.sum()

    #return the filter
    return result


# =====================================================
#            MODEL
# =====================================================

def pre_process_for_landmarks(radiograph, show=False):
    """
    Pre-processes the given radiograph before learning for the landmarks and fitting models
    """
    # TODO improve
    blur_result = cv2.medianBlur(radiograph, 3)
    hist_result = cv2.equalizeHist(blur_result)
    if show:
        cv2.imshow('window', hist_result)
        cv2.waitKey()
    return hist_result


def get_models(landmarks_training_data, multiplier=1.2):
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x), landmarks_training_data)
    models = []
    for i, tooth_landmarks in enumerate(gpa_landmarks):
        eigenvalues, eigenvectors, _ = pca(np.array(tooth_landmarks))
        eigenvectors = np.transpose(eigenvectors)
        projection = np.transpose(np.array([eigenvectors[0], eigenvectors[1], eigenvectors[2]]))
        mean_landmarks = mean_landmarks_normalized(tooth_landmarks)

        max_ds = -1
        for j in range(len(landmarks_training_data[i])):
            b_vector = np.dot(projection.T, (normalize_landmarks(landmarks_training_data[i][j]) - mean_landmarks))
            d = np.sum(np.divide(b_vector**2, eigenvalues[0:len(b_vector)]))
            if max_ds < d:
                max_ds = d*multiplier

        models.append((mean_landmarks, projection, max_ds, eigenvalues[0:3]))
    return models


def fit_model(model, landmarks_neighbourhood, segment, radiograph):
    """
    Fits the given model using an iterative method and returns the vector needed to project the model to the image
    """
    # TODO break when b parameters are violated
    landmarks_segment = initial_fit_model(model, segment)
    vector = [0, 0, 0]
    count = 1
    while True:
        prev = normalize_landmarks(landmarks_segment)
        while True:
            prev_vector = vector
            fitted_model = model[0] + np.dot(model[1], vector)
            _, s = scale_landmarks(landmarks_segment)
            fitted_model, transform_params = procrustes_analysis(landmarks_segment, fitted_model, 1/s)
            # show_model(fitted_model, radiograph, wait=5)
            fitted_landmarks, _ = translate_landmarks(landmarks_segment, transform_params[0])
            fitted_landmarks, _ = scale_landmarks(fitted_landmarks, transform_params[1])
            fitted_landmarks = rotate_landmarks(fitted_landmarks, transform_params[2])
            fitted_landmarks = np.divide(fitted_landmarks, np.dot(fitted_landmarks, model[0]))
            vector = np.dot(model[1].T, (normalize_landmarks(fitted_landmarks) - model[0]))
            if np.sum(abs(vector - prev_vector)) < eps:
                break
        _, s = scale_landmarks(landmarks_segment)
        landmarks_segment, _ = procrustes_analysis(landmarks_segment, model[0] + np.dot(model[1], vector), 1/s)
        t = landmarks_segment.copy()
        landmarks_segment = get_projected_landmarks(landmarks_segment, landmarks_neighbourhood, radiograph, segment, show=False)
        if landmarks_segment is None or count > 1000:
            # print 'too many outlying points'
            vector = np.array([float('inf'), float('inf'), float('inf')])
            break
        # print np.sum(abs(normalize_landmarks(landmarks_segment)-prev))
        if np.sum(abs(normalize_landmarks(landmarks_segment) - prev)) < 0.02*math.log(count, 2) and count > 2:
            # print vector
            break
        count += 1
    show_model(t, radiograph)
    return vector


def show_model(model, segment, color=255, wait=None):
    segment = segment.copy()
    for i in range(0, len(model), 2):
        cv2.circle(segment, (int(model[i]), int(model[i+1])), 3, color)
    cv2.imshow('window', segment)
    if wait is None:
        cv2.waitKey()
    else:
        cv2.waitKey(wait)


def initial_fit_model(model, segment, ratio_width=0.4):
    """
    Returns an initial fit of the model to the segment
    """
    # TODO improve the initial placement using Canny
    width, height = segment[0]
    model = model[0].copy()

    x_model_coords, _ = separate_landmarks(model)
    left_most, right_most = x_model_coords.min(), x_model_coords.max()
    s = abs(left_most - right_most)/(ratio_width*width)
    model, _ = scale_landmarks(model, s)
    _, y_model_coords = separate_landmarks(model)
    return translate_landmarks(model, (segment[1][0], segment[1][1]+height/2-ratio_width*width/2-abs(y_model_coords.min())))[0]


def get_projected_landmarks(fitted_model, landmarks_neighbourhood, radiograph, segment, show=False):
    """
    Returns the optimal projection for each point of the model based on the image and
    the neighbourhood learned from the training data
    """
    draw_image = radiograph.copy()
    projected_landmarks = []
    count = 0
    for i in range(0, len(fitted_model), 2):
        cost_function = lambda x: np.dot(np.dot((x - landmarks_neighbourhood[i/2][0]), landmarks_neighbourhood[i/2][1]),
                                         (x - landmarks_neighbourhood[i/2][0])[np.newaxis].T)
        intensity_vector, normal_vector = get_normal_vector_intensity((fitted_model[i-2], fitted_model[i-1]),
                                                                      (fitted_model[i], fitted_model[i+1]),
                                                                      (fitted_model[(i+2) % len(fitted_model)],
                                                                       fitted_model[(i+3) % len(fitted_model)]),
                                                                      radiograph, wide_search)
        min_cost = float('inf')
        points = (fitted_model[i+1], fitted_model[i])
        k = landmarks_neighbourhood[i/2][0].size/2
        for j in range(k, intensity_vector.size-k):
            fit_intensity_vector = intensity_vector[j-k:j+k+1]
            cost = cost_function(fit_intensity_vector.astype('float64')/np.sum(np.absolute(fit_intensity_vector)))
            if cost < min_cost and normal_vector[j][1] > segment[1][0]-segment[0][0] and normal_vector[j][1] < segment[1][0]+segment[0][0]\
            and normal_vector[j][0] > segment[1][1]-segment[0][1] and normal_vector[j][0] < segment[1][1]+segment[0][1]:
                min_cost = cost
                points = normal_vector[j]
        projected_landmarks.append(points[1])
        projected_landmarks.append(points[0])
        if min_cost == float('inf'):
            count += 1
        if show:
            cv2.line(draw_image, (normal_vector[0][1], normal_vector[0][0]), (normal_vector[-1][1], normal_vector[-1][0]), 120, 1)
            cv2.circle(draw_image, (int(fitted_model[i]), int(fitted_model[i+1])), 1, 255)
            cv2.circle(draw_image, (int(points[1]), int(points[0])), 1, 0)
    if show:
        cv2.imshow('window', draw_image)
        cv2.waitKey(500)
    if count >= len(fitted_model)/8:
        return None
    return np.array(projected_landmarks)


def get_normal_vector_intensity(prev, cur, next, segment, search_length, show=False):
    """
    Returns the intensity and coordinates of the points on the normal for the given coordinates
    """
    height, width = segment.shape
    if next[1] != prev[1]:
        rico = -float(next[0]-prev[0])/float(next[1]-prev[1])
    else:
        rico = 10000
    eq = lambda x: rico*(x-cur[0])+cur[1]

    min_x = int(cur[0]-search_length-1)
    max_x = int(math.ceil(cur[0]+search_length+1))
    min_y = int(eq(min_x))
    max_y = int(math.ceil(eq(max_x)))

    line_points = bresenham_march(segment, (min_x, min_y), (max_x, max_y))

    diff = float(len(line_points)) - (2*search_length + 1)
    line_points = line_points[int(diff/2):len(line_points)-int(math.ceil(diff/2))]

    hist = []
    normal_vector = []
    for x, y in line_points:
        if 0 < y < height and 0 < x < width:
            hist.append(segment[y, x])
            normal_vector.append((y, x))
        else:
            hist.append(0)
            normal_vector.append((y, x))

    if show:
        cv2.line(segment, line_points[0], line_points[-1], 255)
        cv2.circle(segment, (int(cur[0]), int(cur[1])), 1, 0)
        cv2.imshow('window', segment)
        cv2.waitKey()
    return np.array(hist), normal_vector


def create_landmarks_data(file_dir):
    landmarks_training_data = [[0]*(len(os.listdir(file_dir))/8) for _ in range(0, 8)]
    for file_name in sorted(os.listdir(file_dir)):
        if fnmatch.fnmatch(file_name, '*.txt'):
            incisor_landmarks = parse_landmarks_file(file_name, file_dir)
            match = re.match(".*?(\d+)-(\d+)\.txt", file_name)
            landmarks_training_data[int(match.group(2))-1][int(match.group(1))-1] = incisor_landmarks
    return landmarks_training_data


def learn_landmark_neighbourhood(landmarks, radiographs):
    """
    Learn the grayscale neighbourhood of landmark points for later projection use
    """
    landmark_point_local_models = []
    for i in range(0, len(landmarks[0]), 2):
        intensity_vector_list = []
        for j, radiograph in enumerate(radiographs):
            intensity_vector, _ = get_normal_vector_intensity((landmarks[j][i-2], landmarks[j][i-1]),
                                                              (landmarks[j][i], landmarks[j][i+1]),
                                                              (landmarks[j][(i+2) % len(landmarks)],
                                                              landmarks[j][(i+3) % len(landmarks)]),
                                                              radiograph, locality_search)
            intensity_vector_list.append(intensity_vector.astype('float64')/np.sum(np.absolute(intensity_vector)))
        intensity_vector_list = np.array(intensity_vector_list)
        landmark_point_local_models.append((np.mean(intensity_vector_list, axis=0), np.cov(intensity_vector_list.T)**-1))
    return landmark_point_local_models


def parse_landmarks_file(file_name, file_dir):
    f = open(file_dir+'/'+file_name, 'r')
    return np.array(map(float, f.readlines()))


def generalized_procrustes_analysis(landmarks_list):
    """
    Performing the generalized Procrustes analysis for the given landmarks list
    """
    reference_mean = normalize_landmarks(landmarks_list[0])
    landmarks_list = map(normalize_landmarks, landmarks_list)
    weight_vector = calculate_weight_vector(landmarks_list)
    landmarks_list = map(lambda x: np.multiply(weight_vector, x), landmarks_list)
    while True:
        landmarks_list = map(lambda x: procrustes_analysis(reference_mean, x)[0], landmarks_list)
        previous_mean = reference_mean
        reference_mean = mean_landmarks_normalized(landmarks_list)
        if rmse(reference_mean, previous_mean) < eps:
            break
    landmarks_list = map(lambda x: normalize_landmarks(np.divide(x, weight_vector)), landmarks_list)
    return landmarks_list


def procrustes_analysis(reference_landmarks, landmarks, scale=None):
    """
    Performing the Procrustes analysis for the given landmarks
    """
    mean = get_mean_landmarks(reference_landmarks)
    translated_landmarks, translation = translate_landmarks(landmarks, mean)
    scaled_landmarks, s = scale_landmarks(translated_landmarks, scale)
    theta = find_optimal_rotation(reference_landmarks, scaled_landmarks)
    rotated_landmarks = rotate_landmarks(scaled_landmarks, theta)
    return rotated_landmarks, (get_mean_landmarks(landmarks), 1/s, -theta)


def calculate_weight_vector(landmarks_list):
    """
    Weight vector for landmarks based on their variances. Points with high variance are assigned a lower value.
    """
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
    translated_landmarks, _ = translate_landmarks(landmarks, (0, 0))
    scaled_landmarks, _ = scale_landmarks(translated_landmarks)
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
    return np.array(translated_landmarks), (tx, ty)


def scale_landmarks(landmarks, s=None):
    """
    Scales the given landmark vector so that the root mean squared distance is equal to 1
    """
    x_landmarks, y_landmarks = separate_landmarks(landmarks)
    mean_x, mean_y = get_mean_landmarks(landmarks)
    if s is None:
        s = np.sqrt((np.sum((x_landmarks - mean_x)**2) + np.sum((y_landmarks - mean_y)**2)) / len(x_landmarks))

    scaled_landmarks = []
    for i, landmark in enumerate(landmarks):
        if i % 2 == 0:
            scaled_landmarks.append(mean_x + (landmark - mean_x) / s)
        else:
            scaled_landmarks.append(mean_y + (landmark - mean_y) / s)
    return np.array(scaled_landmarks), s


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


def bresenham_march(img, p1, p2):
    """
    Alternative for cv.InitLineIterator that returns the positions
    Source: http://answers.ros.org/question/10160/opencv-python-lineiterator-returning-position-information/
    """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
    if steep:
        t = x1
        x1 = y1
        y1 = t

        t = x2
        x2 = y2
        y2 = t
    also_steep = x1 > x2
    if also_steep:
        t = x1
        x1 = x2
        x2 = t

        t = y1
        y1 = y2
        y2 = t

    dx = x2 - x1
    dy = math.fabs(y2 - y1)
    error = 0.0
    delta_error = 0.0 # Default if dx is zero
    if dx != 0:
        delta_error = math.fabs(dy/dx)

    if y1 < y2:
        y_step = 1
    else:
        y_step = -1

    y = y1
    ret = list([])
    for x in range(x1, x2):
        if steep:
            p = (y, x)
        else:
            p = (x, y)

        ret.append(p)

        error += delta_error
        if error >= 0.5:
            y += y_step
            error -= 1

    if also_steep:
       ret.reverse()

    return ret


if __name__ == '__main__':
    main()