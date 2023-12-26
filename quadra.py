from calculate import *
from square import *

A = np.array([1, 1])
B = np.array([4, 1])
C = np.array([5, 5])
D = np.array([2, 5])

# # 초기에 임의로 선택한 점 P
P1 = np.array([2.5, 2.75], dtype=float)
P2 = np.array([3.5, 3.25], dtype=float)

# 학습률과 에폭 설정
learning_rate = 0.01
epochs = 1000

# 경사하강법으로 P 업데이트
for epoch in range(epochs):
    # 세 변에 수직이 되는 반직선의 벡터
    AB_per = perpendicular_vector(A, B, P1)
    BC_per = perpendicular_vector(B, C, P2)
    CD_per = perpendicular_vector(C, D, P2)
    DA_per = perpendicular_vector(D, A, P1)

    points_edge = [np.array([0, 0]), np.array([6, 0]), np.array([0, 6]), np.array([6, 6])]

    # 정사각형 경계와의 교점
    p_AB=find_intersection(P1, AB_per)
    p_BC=find_intersection(P2, BC_per)
    p_CD=find_intersection(P2, CD_per)
    p_DA=find_intersection(P1, DA_per)

    area_A = calculate_convex_polygon_area([[0, 0], p_AB, P1, p_DA])
    area_B = calculate_convex_polygon_area([[6, 0], p_BC, P2, P1, p_AB])
    area_C = calculate_convex_polygon_area([P2, p_BC, [6, 6], p_CD])
    area_D = calculate_convex_polygon_area([P2, p_CD, [0, 6], p_DA, P1])

    # P로부터 세 변에 수직이 되는 반직선의 방향벡터에 대한 gradient 계산
    gradient_A = (9 - area_A)
    gradient_B = (9 - area_B)
    gradient_C = (9 - area_C)
    gradient_D = (9 - area_D)
    # P 업데이트
    vector_p1_to_p2 = np.array([P2[0] - P1[0], P2[1] - P1[1]])
    P1 = translate_along_vector(P1, P2, gradient_A*learning_rate)
    P2 = translate_along_vector(P2, P1, gradient_C*learning_rate)
    P1 = translate_perpendicular_only(P1, vector_p1_to_p2, gradient_B*learning_rate-gradient_D*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p1_to_p2, gradient_B*learning_rate-gradient_D*learning_rate)
    # 손실 출력
    loss = (9 - area_A) ** 2 + (9 - area_B) ** 2 + (9 - area_C) ** 2 + (9 - area_D) ** 2
    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, P1: {P1}, P2: {P2}, Area A : {area_A}, Area B : {area_B}, Area C : {area_C}, Area D: {area_D}')
    if epoch<=50:
      if epoch<=10 or epoch%10==0:
        plt.plot([P1[0]],[P1[1]],'ro')
        plt.text(P1[0], P1[1], f'{epoch}', fontsize=7, ha='right')
        plt.plot([P2[0]],[P2[1]],'ro')
        plt.text(P2[0], P2[1], f'{epoch}', fontsize=7, ha='right')

P__AB=intersection(AB_per, P1, A,B)
P__BC=intersection(BC_per, P2, B,C)
P__CD=intersection(CD_per, P2, C,D)
P__DA=intersection(DA_per, P1, D,A)


plt.quiver(*P1,*-(P1-P__AB),scale=1)
plt.quiver(*P2,*-(P2-P__BC),scale=1)
plt.quiver(*P2,*-(P2-P__CD),scale=1)
plt.quiver(*P1,*-(P1-P__DA),scale=1)
plot_line(A,B)
plot_line(B,C)
plot_line(C,D)
plot_line(D,A)
plot_line(P1,P2)
plot_line(B,D)

plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.axis('equal')
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()