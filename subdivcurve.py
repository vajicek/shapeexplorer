import math

EPSILON = 1e-6


def dist(p1, p2):
    return math.sqrt(sum([(a - b) * (a - b) for a, b in zip(p1, p2)]))


def interpolate(p1, p2, t):
    return [(1 - t) * a + t * b for (a, b) in zip(p1, p2)]


def curve_length(polyline_data):
    length = 0
    for i in range(len(polyline_data) - 1):
        length = length + dist(polyline_data[i], polyline_data[i + 1])
    return length


def subdivide_curve(polyline_data, count):
    length = curve_length(polyline_data)
    segment_length = length / (count - 1)
    new_polyline_data = []
    length = 0
    polyline_data_index = 0
    for i in range(count):
        # first point
        if i == 0:
            new_polyline_data.append(polyline_data[polyline_data_index])
            continue

        while length < i * segment_length:
            length = length + \
                dist(polyline_data[polyline_data_index],
                     polyline_data[polyline_data_index + 1])
            polyline_data_index = polyline_data_index + 1

        # last point
        if polyline_data_index == len(polyline_data) - 1:
            new_polyline_data.append(polyline_data[polyline_data_index])
            continue

        # interpolate
        a = polyline_data[polyline_data_index]
        b = polyline_data[polyline_data_index + 1]
        diff = length - i * segment_length
        t = diff / dist(a, b)
        new_polyline_data.append(interpolate(a, b, t))

    return new_polyline_data
