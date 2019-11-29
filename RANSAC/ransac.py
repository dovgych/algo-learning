from data import noisy_line_data
import matplotlib.pyplot as plt


if __name__ == '__main__':

    X, model = noisy_line_data()
    plt.plot(X, 'bo')
    plt.show()

    print('Hi')