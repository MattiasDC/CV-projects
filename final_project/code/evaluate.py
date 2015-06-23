from main import *


def main():
    # print "Which image do you want to use as test data? (number between 0 and 13)"
    # image_index = int(raw_input())

    costs = [0]*8

    for image_index in range(14):
        print image_index
        # test_image = read_radiographs(radiographs_dir)[image_index]
        # cropped_landmarks_preprocessed, models, landmarks_neighbourhoods_levels, horizontal_paths = initialise()

        cropped_landmarks_preprocessed, models, landmarks_neighbourhoods_levels, horizontal_paths, test_image = initialise(image_index)
        image = crop_radiograph(pre_process_for_landmarks(test_image))
        height, width = test_image.shape

        landmarks_data = create_landmarks_data(landmarks_dir)
        path = segment_teeth(pre_process_radiograph(crop_radiograph(test_image)), 50)
        total_draw = image.copy()

        for i in range(len(models)):
            draw_image = image.copy()
            landmarks = landmarks_data[i][image_index]
            cropped_landmarks = []
            for j in range(0, len(landmarks), 2):
                cropped_landmarks.append(landmarks[j]-0.1*width)
                cropped_landmarks.append(landmarks[j+1]-0.1*height)
                cv2.circle(draw_image, (int(cropped_landmarks[j]), int(cropped_landmarks[j+1])), 3, 255, thickness=-1)
                cv2.circle(total_draw, (int(cropped_landmarks[j]), int(cropped_landmarks[j+1])), 3, 255, thickness=-1)

            segment = get_segment_for_eval(cropped_landmarks)

            mid_height = segment[1][1]
            mid_width = segment[1][0]
            first = 4
            for j in range(0, len(path)-1, 2):
                if path[j][0] <= mid_width <= path[j+1]:
                    if mid_height <= (path[j][1]+path[j+1][1])/2:
                        first = 0


            vector, fitted_model = fit_model(models[i], landmarks_neighbourhoods_levels[i], segment, image, (first == 0), show=False)
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

            cost = calculate_cost_eval(fitted_model, cropped_landmarks)
            costs[i] += cost
            print "model %d, cost %f, d %f, dmax %f, b-vector: %s" % (i, cost, d, models[i][2], str(vector))

            for j in range(0, len(fitted_model), 2):
                cv2.circle(draw_image, (int(fitted_model[j]), int(fitted_model[j+1])), 3, 0, thickness=-1)
                cv2.circle(total_draw, (int(fitted_model[j]), int(fitted_model[j+1])), 3, 0, thickness=-1)
            # cv2.imshow('window', draw_image)
            # cv2.waitKey()
        cv2.imshow('window', total_draw)
        cv2.waitKey()

        # name = raw_input('file name? ')
        # if len(name) > 0:
        #     cv2.imwrite('../figures/'+name+'.png', total_draw)
    print map(lambda x: x/14, costs)


def calculate_cost_eval(fitted_model, landmarks):
    average_pixel_displacement = 0.0
    for i in range(0, len(landmarks), 2):
        average_pixel_displacement += math.sqrt((fitted_model[i]-landmarks[i])**2+(fitted_model[i+1]-landmarks[i+1])**2)
    return average_pixel_displacement/(len(landmarks)/2)


def get_segment_for_eval(landmarks, ratio=1.1):
    x_values, y_values = separate_landmarks(landmarks)
    w = ratio*(max(x_values)-min(x_values))
    h = ratio*(max(y_values)-min(y_values))
    x = min(x_values) + w/2
    y = min(y_values) + h/2
    return (int(w), int(h)), (int(x), int(y))


if __name__ == '__main__':
    main()