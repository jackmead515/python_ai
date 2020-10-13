import sys
import numpy as np
import matplotlib.pyplot as plot

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

sys.path.append('../')

from tutil import save_pickle, load_pickle

def line_data():
    x = 2 * np.random.rand(200, 1)
    y = 1 + 3 * x + np.random.rand(200, 1)
    return x, y

def poly_data(amount):
    x = 5 * np.random.rand(amount, 1)
    y = 5 * x ** 2 - 3 * x + 10 + np.random.rand(amount, 1)
    return x, y

def sin_data(amount, noise=1):
    x = 2 * np.random.rand(amount, 1)
    y = 3 * np.sin(x)*noise + 5 + np.random.rand(amount, 1)
    return x, y

def linear():
    x, y = line_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)

    plot.plot(x, y, 'b.', color="blue")
    plot.plot(x_test, y_pred, color="red")
    plot.show()
    
def poly():
    x, y = poly_data(500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = make_pipeline(
        PolynomialFeatures(degree=5),
        StandardScaler(),
        LinearRegression()
    )

    train_error, test_error = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_pred = model.predict(x_train[:m])
        y_test_pred = model.predict(x_test)
        train_mse = mean_squared_error(y_train[:m], y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_error.append(train_mse)
        test_error.append(test_mse)

    plot.plot(np.sqrt(train_error), color="red", label="train")
    plot.plot(np.sqrt(test_error), color="blue", label="test")
    plot.show()

    y_pred = model.predict(x_test)
    plot.plot(x, y, 'b.', color="blue")
    plot.plot(x_test, y_pred, 'b.', color="red")
    plot.show()

def linear_bgd():
    x, y = line_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    x = np.c_[np.ones((200, 1)), x]

    lr = 0.1
    epochs = 1000
    m = 200
    theta = np.random.randn(2, 1)

    for _ in range(epochs):
        gradients = 2/m * x.T.dot(x.dot(theta) - y)
        theta = theta - lr * gradients

    print(theta)

def sdg_regressor():
    x, y = sin_data(500, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler()
        #MinMaxScaler(feature_range=(0, 1))
    )

    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = SGDRegressor(
        max_iter=1,
        tol=-np.infty,
        warm_start=True,
        learning_rate='constant',
        eta0=0.0005,
        penalty=None
    )

    epochs = 25
    verr, terr = [], []
    for epoch in range(epochs):
        model.fit(x_train_scaled, y_train.ravel())

        y_test_pred = model.predict(x_test_scaled)
        verr.append(mean_squared_error(y_test, y_test_pred))
        y_train_pred = model.predict(x_train_scaled[:len(x_test_scaled)])
        terr.append(mean_squared_error(y_train[:len(x_test_scaled)], y_train_pred))


    plot.plot(np.sqrt(verr), color="red", label="test")
    plot.plot(np.sqrt(terr), color="blue", label="train")
    plot.legend('upper right')
    plot.show()

    y_pred = model.predict(x_test_scaled)
    plot.plot(x, y, 'b.', color="blue")
    plot.plot(x_test, y_pred, 'b.', color="red")
    plot.show()

def knearest_regressor():
    x, y = sin_data(500, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler()
        #MinMaxScaler(feature_range=(0, 1))
    )

    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = KNeighborsRegressor(
        n_neighbors=5,
        weights='uniform',
        n_jobs=8
    )

    model.fit(x_train_scaled, y_train.ravel())

    y_pred = model.predict(x_test_scaled)
    plot.plot(x, y, 'b.', color="blue")
    plot.plot(x_test, y_pred, 'b.', color="red")
    plot.show()

if __name__ == "__main__":
    knearest_regressor()
