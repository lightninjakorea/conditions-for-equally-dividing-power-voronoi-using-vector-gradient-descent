import numpy as np
from shapely.geometry import Point, LineString

def find_intersection(p, v):
    # 정사각형의 두 꼭지점 좌표
    square_vertices = [(0, 0), (0, 6), (6, 6), (6, 0), (0, 0)]
    v=np.array(v)
    # 반직선의 시작점 P
    P = Point(tuple(p))

    # 반직선을 나타내는 선분
    line_end_point = tuple(np.array(P.coords[0]) + 1000 * v)
    line = LineString([P.coords[0], line_end_point])

    # 정사각형과 반직선의 교점 계산
    intersection_points = line.intersection(LineString(square_vertices))

    for point in intersection_points.coords:
        return point
