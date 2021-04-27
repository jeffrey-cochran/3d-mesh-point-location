from numpy.random import uniform
from numpy import asarray
from numpy.linalg import norm
from csv import writer

x = uniform(size=(100,4))
x_sum = x.sum(axis=1)
x_normed = (x.transpose() / x_sum).transpose()

corners = asarray([
    [0.,  0.,  0.],
    [0., -5.,  0.],
    [0.,  0.,  5.],
    [5.,  0.,  0.]
])

external_points = x_normed.dot(corners)
external_points = (
    external_points.transpose() / norm(external_points, axis=1)
).transpose() * 10.

# print(norm(external_points, axis=1))


with open("test_tetra_outside_points.csv", "w", newline='') as f:
    w = writer(f, delimiter=",")
    for point in external_points:
        w.writerow(point)