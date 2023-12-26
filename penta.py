from calculate import *
from square import *

A = np.array([1, 1])
B = np.array([2, 4])
C = np.array([4, 5])
D = np.array([5, 3])
E = np.array([5, 0])

# # 초기에 임의로 선택한 점 P
P1 = np.array([3, 2], dtype=float)
P2 = np.array([3.25, 1.5], dtype=float)
P3 = np.array([3.5, 3.5], dtype=float)

# 학습률과 에폭 설정
learning_rate = 0.01 #점차적 감소
epochs = 1000

# 경사하강법으로 P 업데이트
for epoch in range(epochs):
    # 세 변에 수직이 되는 반직선의 벡터
    AB_per = perpendicular_vector(A, B, P1)
    BC_per = perpendicular_vector(B, C, P3)
    CD_per = perpendicular_vector(C, D, P3)
    DE_per = perpendicular_vector(D, E, P2)
    EA_per = perpendicular_vector(E, A, P2)

    points_edge = [np.array([0, 0]), np.array([6, 0]), np.array([0, 6]), np.array([6, 6])]

    # 정사각형 경계와의 교점
    p_AB=find_intersection(P1, AB_per)
    p_BC=find_intersection(P3, BC_per)
    p_CD=find_intersection(P3, CD_per)
    p_DE=find_intersection(P2, DE_per)
    p_EA = find_intersection(P2, EA_per)

    area_A = calculate_convex_polygon_area([[0, 0], p_AB, P1, P2, p_EA])
    area_B = calculate_convex_polygon_area([[0, 6], p_BC, P3, P1, p_AB])
    area_C = calculate_convex_polygon_area([p_BC, [6,6], p_CD, P3])
    area_D = calculate_convex_polygon_area([P2, P1, P3, p_CD, p_DE])
    area_E = calculate_convex_polygon_area([P2, p_DE,[6,0],p_EA])

    # P로부터 세 변에 수직이 되는 반직선의 방향벡터에 대한 gradient 계산
    gradient_A = (7.2 - area_A)
    gradient_B = (7.2 - area_B)
    gradient_C = (7.2 - area_C)
    gradient_D = (7.2 - area_D)
    gradient_E = (7.2 - area_E)
    # P 업데이트
    vector_p1_to_p3 = -np.array([P3[0] - P1[0], P3[1] - P1[1]])
    vector_p2_to_p1 = -np.array([P1[0] - P2[0], P1[1] - P2[1]])
    vector_p3_to_p2 = -np.array([P2[0] - P3[0], P2[1] - P3[1]])
    P1 = translate_perpendicular_only(P1, vector_p2_to_p1, gradient_A*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p2_to_p1, gradient_A*learning_rate)
    P3 = translate_perpendicular_only(P3, vector_p2_to_p1, gradient_A*learning_rate)

    P1 = translate_perpendicular_only(P1, vector_p1_to_p3, gradient_B*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p1_to_p3, gradient_B*learning_rate)
    P3 = translate_perpendicular_only(P3, vector_p1_to_p3, gradient_B*learning_rate)

    P1 = translate_perpendicular_only(P1, vector_p3_to_p2, gradient_D*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p3_to_p2, gradient_D*learning_rate)
    P3 = translate_perpendicular_only(P3, vector_p3_to_p2, gradient_D*learning_rate)

    learning_rate *= 0.999999

    P2 = translate_along_vector(P2, P1, gradient_E*learning_rate)
    P3 = translate_along_vector(P3, P1, gradient_C*learning_rate)

    # 손실 출력
    loss = (7.2 - area_A) ** 2 + (7.2 - area_B) ** 2 + (7.2 - area_C) ** 2 + (7.2 - area_D) ** 2 + (7.2 - area_E)**2
    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, P1: {P1}, P2: {P2}, P3 : {P3} \n Area A : {area_A}, Area B : {area_B}, Area C : {area_C}, Area D: {area_D}, Area E : {area_E}')
    # if epoch<=50:
    #   if epoch<=10 or epoch%10==0:
    #     plt.plot([P1[0]],[P1[1]],'ro')
    #     plt.text(P1[0], P1[1], f'{epoch}', fontsize=7, ha='right')
    #     plt.plot([P2[0]],[P2[1]],'ro')
    #     plt.text(P2[0], P2[1], f'{epoch}', fontsize=7, ha='right')

plot_line(p_EA,P2)
plot_line(p_DE,P2)
plot_line(p_AB,P1)
plot_line(p_BC,P3)
plot_line(p_CD,P3)

plot_line(A,B)
plot_line(B,C)
plot_line(C,D)
plot_line(D,E)
plot_line(E,A)
plot_line(P1,P2)
plot_line(P1,P3)
plot_line(B,D)
plot_line(D,A)


plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.axis('equal')
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()