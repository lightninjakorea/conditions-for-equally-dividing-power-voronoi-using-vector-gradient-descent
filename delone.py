import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# 랜덤한 점 생성
points = np.array([[2, 1], [2, 3], [2, 5], [5, 3]])

# 들로네 삼각분할 수행
tri = Delaunay(points)

# 결과를 저장할 딕셔너리 초기화
point_lines = {}

# 각 삼각형의 변에 대해 정보 수집
for simplex in tri.simplices:
    for i in range(3):
        # 각 점의 좌표
        x, y = points[simplex[i]]
        # 다음 점의 좌표
        nx, ny = points[simplex[(i + 1) % 3]]

        # 선분 정보 수집
        line_info = f"({x:.2f}, {y:.2f}) to ({nx:.2f}, {ny:.2f})"

        # 딕셔너리에 저장
        if (x, y) not in point_lines:
            point_lines[(x, y)] = [line_info]
        else:
            point_lines[(x, y)].append(line_info)

# 결과 출력 및 시각화
for point, lines in point_lines.items():
    print(f"Point {point}:")
    for line in lines:
        print(f"  {line}")
    print()

# 들로네 삼각형 시각화
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')

# 각 삼각형의 변에 대해 벡터 계산 및 시각화
for simplex in tri.simplices:
    for i in range(3):
        # 각 점의 좌표
        x, y = points[simplex[i]]
        # 다음 점의 좌표
        nx, ny = points[simplex[(i + 1) % 3]]
        # 벡터 계산 및 시각화
        plt.arrow(x, y, nx - x, ny - y, head_width=0.05, head_length=0.1, fc='red', ec='red')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Delaunay Triangulation')
plt.show()
