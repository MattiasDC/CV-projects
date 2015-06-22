from main import *


def main():
    print "Which image do you want to use as test data? (number between 0 and 14)"
    image_index = int(raw_input())
    cropped_landmarks_preprocessed, models, landmarks_neighbourhoods_levels, horizontal_paths, test_image = initialise(image_index)
    image = crop_radiograph(pre_process_for_landmarks(test_image))
    height, width = test_image.shape

    landmarks_data = create_landmarks_data(landmarks_dir)
    path = horizontal_paths[image_index]

    for i in range(len(models)):
        draw_image = image.copy()
        landmarks = landmarks_data[i][image_index]
        for j in range(0, len(landmarks), 2):
            cv2.circle(draw_image, (int(landmarks[j]-0.1*width), int(landmarks[j+1]-0.1*height)), 3, 255, thickness=-1)

        segment = get_segment(draw_image)

        mid_height = segment[1][1]
        mid_width = segment[1][0]
        first = 4
        for j in range(0, len(path)-1, 2):
            if path[j][0] <= mid_width <= path[j+1]:
                if mid_height <= (path[j][1]+path[j+1][1])/2:
                    first = 0


        vector, fitted_model = fit_model(models[i], landmarks_neighbourhoods_levels[i], segment, image, (first == 0))
        d = np.sum(np.divide(vector**2, models[i][3]))

        cost = 0
        for j in range(0, len(fitted_model), 2):
            cost_function = lambda x: np.dot(np.dot((x - landmarks_neighbourhoods_levels[i][0][j/2][0]),
                                                    landmarks_neighbourhoods_levels[i][0][j/2][1]),
                                             (x - landmarks_neighbourhoods_levels[i][0][j/2][0])[np.newaxis].T)

            intensity_vector, normal_vector = get_normal_vector_intensity((fitted_model[j-2], fitted_model[j-1]),
                                                                          (fitted_model[j], fitted_model[j+1]),
                                                                          (fitted_model[(j+2) % len(fitted_model)],
                                                                           fitted_model[(j+3) % len(fitted_model)]),
                                                                          image, wide_search)
            fit_intensity_vector = intensity_vector[len(intensity_vector)/2-locality_search:len(intensity_vector)/2+locality_search+1]
            cost += abs(cost_function(fit_intensity_vector.astype('float64')/(np.sum(fit_intensity_vector)+1)))
        cost /= len(fitted_model)/2
        print "model %d, d %f, dmax %f, eucl. cost: %f, b-vector: %s" % (i, d, models[i][2], cost, str(vector))
        for j in range(0, len(fitted_model), 2):
            cv2.circle(draw_image, (int(fitted_model[j]), int(fitted_model[j+1])), 3, 0, thickness=-1)
        cv2.imshow('window', draw_image)
        cv2.waitKey()

if __name__ == '__main__':
    main()