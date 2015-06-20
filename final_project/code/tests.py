from main import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def test_landmarks_interpretation():
        landmarks_training_data = create_landmarks_data(landmarks_dir)
        for i in range(14):
            for j in range(8):
                x, y = separate_landmarks(landmarks_training_data[j][i])
                plt.plot(x, y, ".")
                for k, (X, Y) in enumerate(zip(x, y)):
                    plt.annotate('{}'.format(k), xy=(X, Y))
                mx = np.mean(x)
                my = np.mean(y)
                dist, dx, dy = max(map(lambda (a, b): (math.sqrt((mx-a)**2+(my-b)**2), a, b), zip(x, y)))
                plt.annotate('X', xy=(mx, my))
                plt.annotate('tooth' + str(j), xy=(mx+50, my+50))
                plt.plot([dx, mx], [dy, my]),
                plt.axis('equal')
                plt.show()


def test_gpa():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x), landmarks_training_data)

    for i in range(8):
        plt.figure(1)
        for landmarks in gpa_landmarks[i]:
            x, y = separate_landmarks(landmarks)
            plt.plot(x, y, ".")
        x2, y2 = separate_landmarks(mean_landmarks_normalized(gpa_landmarks[i]))
        plt.plot(x2, y2)
        plt.axis('equal')
        plt.show()


def determine_b_parameters():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x), landmarks_training_data)

    for i in range(8):
        tooth_landmarks = np.array(gpa_landmarks[i])
        eigenvalues, eigenvectors, _ = pca(tooth_landmarks)
        eigenvectors = np.transpose(eigenvectors)
        fractions = eigenvalues / sum(eigenvalues)
        print "tooth %d: %s" % (i+1, str(fractions))

        projection = np.transpose(np.array([eigenvectors[0], eigenvectors[1], eigenvectors[2]]))
        mean_landmarks = mean_landmarks_normalized(gpa_landmarks[i])
        x2, y2 = separate_landmarks(mean_landmarks)
        x3, y3 = separate_landmarks(normalize_landmarks(landmarks_training_data[i][0]))

        fig = plt.figure()
        plt.axis('equal')
        plot, = plt.plot(x2, y2, color='blue')
        plt.plot(x2, y2, color='red')
        plt.plot(x3, y3, color='green')

        ax_comp1 = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_comp1 = Slider(ax_comp1, 'PCA 1', -3*math.sqrt(eigenvalues[0]), 3*math.sqrt(eigenvalues[0]), valinit=0)
        ax_comp2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        slider_comp2 = Slider(ax_comp2, 'PCA 2', -3*math.sqrt(eigenvalues[1]), 3*math.sqrt(eigenvalues[1]), valinit=0)
        ax_comp3 = plt.axes([0.25, 0.2, 0.65, 0.03])
        slider_comp3 = Slider(ax_comp3, 'PCA 3', -3*math.sqrt(eigenvalues[2]), 3*math.sqrt(eigenvalues[2]), valinit=0)

        def update(val):
            b1 = slider_comp1.val
            b2 = slider_comp2.val
            b3 = slider_comp3.val

            b = np.transpose(np.array([b1, b2, b3]))
            l_x, l_y = separate_landmarks(mean_landmarks + np.dot(projection, b))

            plot.set_ydata(l_y)
            plot.set_xdata(l_x)
            fig.canvas.draw()

        slider_comp1.on_changed(update)
        slider_comp2.on_changed(update)
        slider_comp3.on_changed(update)

        for j in range(len(landmarks_training_data[i])):
            b_vector = np.dot(projection.T, (normalize_landmarks(landmarks_training_data[i][j]) - mean_landmarks))
            d = np.sum(np.divide(b_vector**2, eigenvalues[0:len(b_vector)]))
            print "D_m^2: %f, b_vector, tooth %d, shape %d: %s" % (d, i, j+1, b_vector)

        plt.show()


if __name__ == '__main__':
    # test_landmarks_interpretation()
    # test_gpa()
    determine_b_parameters()
