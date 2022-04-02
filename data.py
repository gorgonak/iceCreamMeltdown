import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


def predict(X, w, b):
    """
    calculates inputs (X) multiplied by weight (w)
    this supports our error rate calculations (also known as 'loss')

    """
    return X * w + b


def loss(X, Y, w, b):
    """
    calculates the loss

    :param X: inputs
    :param Y:
    :param w: weight
    :return: the loss (also known as error rate)
    """
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    """
    trains our model and supports our training functions

    :param X:
    :param Y:
    :param iterations:
    :param lr:
    :return:
    """
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("iterations %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Could not converge withing %d iterations" % iterations)


def import_training_data():
    """
    imports our training data and returns two arrays (X, Y)
    :return: X and Y

    """
    X, Y = np.loadtxt("icecream.txt", skiprows=1, unpack=True)
    return X, Y


def training(X, Y):
    w, b = train(X, Y, iterations=10000, lr=0.01)
    print("\nw=%.3f" % w, b)
    return w, b


def prediction(outside_temp, w, b):
    """
    this is the key for our user, it shows how many pizzas we can plan on making.
    (the prediction)
    :param outside_temp:
    :param w:
    :return:
    """

    print("Based on %d degree temperatures (x-value), you should sell %.2f ice cream cones (y-value)."
          % (float(outside_temp), predict(float(outside_temp), w, b)))

    return


def plot_chart(X, Y, w, b):
    sns.set()
    plt.plot(X, Y, "ro")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Ice Cream Meltdown", fontsize=20)
    plt.ylabel("Number of Cones", fontsize=20)
    plt.xlabel("Outside Temperature", fontsize=20)
    x_edge, y_edge = 110, 40
    plt.axis([70, x_edge, 0, y_edge])
    plt.plot([0, x_edge], [0, predict(x_edge, w, b)], linewidth=1.0, color="g")
    plt.show()


def welcome_user():
    print("\n\n\t *** Welcome to Ice Cream Meltdown *** \n*** The ice cream sales prediction software. ***\n")
    print("I will help you predict the amount of ice cream cones you will sell based off the temperature outside.")


def clean_up():
    print("\n\n *** Thank you for using Ice Cream Meltdown. "
          "To find out how many ice cream cones you will sell another day, please re-run this program. *** ")
    print("To exit the program, close the graph window and the GUI.")


def main():
    """
    this is the driver function
    :return:
    """
    welcome_user()  # welcomes the user lets them know what is going on
    X, Y = import_training_data()  # imports the data

    # train our model
    print("\n\t *** Training *** \n")
    time.sleep(5)
    w, b = training(X, Y)
    print("\n\t *** Training Completed ***\n")
    time.sleep(5)

    # ask the user to input the number of reservations they have
    outside_temp = input("What is the temperature outside today? ")

    # predict the number of pizza dough units to make based on the number of reservations.
    prediction(outside_temp, w, b)

    # now we clean up and exit
    clean_up()

    # now we can create our graph to show historical data
    plot_chart(X, Y, w, b)

    pass


if __name__ == "__main__":
    """
    this is where our program starts and calls the driver function
    """
    main()
