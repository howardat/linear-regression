import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def loss_function(m, b, points):
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i].tv
        y = points.iloc[i].sales
        total_loss += (y - (m * x + b)) ** 2
    total_error = total_loss / len(points)
    return total_error

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].TV
        y = points.iloc[i].Sales

        m_gradient += - (2 / n) * (y - (m_now * x + b_now)) * x
        b_gradient += - (2 / n) * (y - (m_now * x + b_now))

    m = m_now - (L * m_gradient)
    b = b_now - (L * b_gradient)

    return m, b

if __name__ == "__main__":
    data = pd.read_csv('../../data/Advertising.csv')

    m = 0
    b = 0
    L = 0.00001
    epochs = 1000

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)

    print(m, b)
    plt.scatter(data['TV'], data['Sales'])
    plt.plot(list(range(0, 300)), [m * x + b for x in range(0, 300)], color='red')
    plt.show()