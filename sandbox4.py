from numpy.random import uniform
from numpy import asarray
from numpy.linalg import norm
from csv import writer

r = uniform(low=6, high=20, size=(10000,))
x = uniform(low=-1, high=1, size=(10000,3))
x_norm = norm(x, axis=1)
x_normed = (x.transpose() / x_norm).transpose()

external_points = (x_normed.transpose() * r).transpose()

# print(norm(external_points, axis=1))


with open("test_tetra_outside_points.csv", "w", newline='') as f:
    w = writer(f, delimiter=",")
    for point in external_points:
        w.writerow(point)