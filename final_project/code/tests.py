from main import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def test_gpa():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x, eps=10**-14), landmarks_training_data)

    for i in range(8):
        plt.figure(1)
        for landmarks in gpa_landmarks[i]:
            x, y = separate_landmarks(landmarks)
            plt.plot(x, y, ".")
        x2, y2 = separate_landmarks(mean_landmarks_normalized(gpa_landmarks[i]))
        plt.plot(x2, y2)
        plt.show()


def determine_b_parameters():
    landmarks_training_data = create_landmarks_data(landmarks_dir)
    gpa_landmarks = map(lambda x: generalized_procrustes_analysis(x, eps=10**-14), landmarks_training_data)

    for i in range(8):
        tooth_landmarks = np.array(gpa_landmarks[i])
        eigenvalues, eigenvectors, _ = pca(tooth_landmarks)
        fractions = eigenvalues / sum(eigenvalues)
        print "tooth %d: %s" % (i+1, str(fractions))

        plt.figure(1)
        x2, y2 = separate_landmarks(mean_landmarks_normalized(gpa_landmarks[i]))
        plt.plot(x2, y2, color='red')
        plt.show()



if __name__ == '__main__':
    determine_b_parameters()
