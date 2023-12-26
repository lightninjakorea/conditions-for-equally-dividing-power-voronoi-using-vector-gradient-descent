import numpy as np
import matplotlib.pyplot as plt
def perpendicular_vector(A, B, P):
    AB = B - A
    AP = P - A

    projection = A + np.dot(AP, AB) / np.dot(AB, AB) * AB
    perpendicular_vector = projection-P

    return perpendicular_vector
def intersection(v1, p1, point1, point2):
    v2 = point2 - point1

    A = np.array([v1, -v2]).T
    b = point1 - p1

    try:
        # 선형 연립방정식 Ax = b를 푸는 함수
        x = np.linalg.solve(A, b)

        # 교점 계산
        intersection_point = p1 + x[0] * v1
        if 0<=x[0] and 0<=x[1]<=1 : return intersection_point
        else : return None
    except np.linalg.LinAlgError:
        # 행렬이 풀리지 않으면 선분이 평행하거나 겹치지 않는 경우
        return None
def calculate_convex_polygon_area(points): #points : 볼록다각형을 이루는 점들의 리스트
    n = len(points)
    if n < 3:
        raise ValueError("적어도 3개 이상의 정점이 필요합니다.")
    # 볼록다각형의 넓이 계산
    area = 0
    for i in range(n): #신발끈정리
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = 0.5 * np.abs(area)
    return area
def calculate_distance(point1,point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1-point2)
    return distance
def calculate_internal_ratio(point_a, point_b, point_c): #c에서 ab에 그은 수선의 발 H에 대해 AH/AB
    # 넘파이 배열로 변환
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    # 벡터 계산
    vector_ac = point_c - point_a
    vector_ab = point_b - point_a

    # 내분비 계산
    t = np.dot(vector_ac, vector_ab) / np.dot(vector_ab, vector_ab)
    return t
def per_point(A, B, P) :
    AB = B - A
    AP = P - A
    return A + np.dot(AP, AB) / np.dot(AB, AB) * AB

def plot_line(A, B):
    # A와 B의 좌표를 추출
    x_coords = [A[0], B[0]]
    y_coords = [A[1], B[1]]

    # 선분을 그림
    plt.plot(x_coords, y_coords, marker='o',color='blue')

def plot_circle(center, radius):
    # 중심 좌표와 반지름
    cx, cy = center
    r = radius
    # 원 위의 각도 배열 생성
    theta = np.linspace(0, 2*np.pi, 100)
    # 원의 좌표 계산
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    # 원 그리기
    plt.plot(x, y, label=f'원 (중심: {center}, 반지름: {radius})', color='blue')
    # 중심점 표시
    plt.scatter(cx, cy, color='blue', marker='o', label='중심점')

def translate_along_vector(p1, p2, distance_along):
    # P1에서 P2로 향하는 벡터 계산
    vector_p1_to_p2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    # 벡터의 크기 계산
    magnitude = np.linalg.norm(vector_p1_to_p2)

    # 단위 벡터 계산
    unit_vector = vector_p1_to_p2 / magnitude

    # 이동할 새로운 좌표 계산
    new_point = p1 + distance_along * unit_vector

    return new_point



def translate_perpendicular_only(p1, vector_p1_to_p2, distance_perpendicular):
    # 벡터의 크기 계산
    magnitude = np.linalg.norm(vector_p1_to_p2)

    # 단위 벡터 계산
    unit_vector = vector_p1_to_p2 / magnitude

    # 벡터의 수직 방향 계산
    perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])

    # 수직 방향으로 이동할 좌표 계산
    move_perpendicular = p1 + distance_perpendicular * perpendicular_vector

    return move_perpendicular

def translate_along_byvector(p1, vector_p1_to_p2, distance):
    # 벡터의 크기 계산
    magnitude = np.linalg.norm(vector_p1_to_p2)

    # 단위 벡터 계산
    unit_vector = vector_p1_to_p2 / magnitude

    # 수직 방향으로 이동할 좌표 계산
    move_perpendicular = p1 + distance * unit_vector

    return move_perpendicular

def finding_intersection(v1, p1, v2, p2):
    # 벡터의 매개 방정식: p1 + t * v1 = p2 + s * v2
    # 이를 각 좌표에 대해 분리하여 연립 방정식을 풀어 교점을 찾음

    A = np.array([v1, -v2]).T
    b = p2 - p1

    try:
        # 선형 시스템을 해결하여 교점을 찾음
        t, s = np.linalg.solve(A, b)
        intersection_point = p1 + t * v1
        return intersection_point
    except np.linalg.LinAlgError:
        # 해가 없는 경우 (벡터가 평행한 경우)
        return None

def calculate_vector(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2 - p1
    return vector




