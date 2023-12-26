import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

def voronoi_rel(points):
    # Voronoi 다이어그램 계산
    vor = Voronoi(points)

    # 결과를 저장할 딕셔너리 초기화
    result_dict = {}

    # 각 초기점에 대해 형성에 영향을 미친 vertices 출력
    for i, point in enumerate(points):
        affected_vertices = []
        for j, ridge_points in enumerate(vor.ridge_points):
            if i in ridge_points:
                ridge_vertices = vor.ridge_vertices[j]
                if -1 in ridge_vertices:
                    ridge_vertices.remove(-1)
                affected_vertices.extend(vor.vertices[ridge_vertices])

        affected_vertices = np.unique(affected_vertices, axis=0)  # 중복 제거
        affected_vertices = [tuple(v) for v in affected_vertices if not np.any(np.isnan(v))]  # NaN 제거

        # 결과 딕셔너리에 추가
        result_dict[tuple(point)] = affected_vertices

    return result_dict