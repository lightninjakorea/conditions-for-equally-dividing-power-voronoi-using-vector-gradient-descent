from calculate import *
from square import *

A = np.array([0, 4])
B = np.array([2, 5])
C = np.array([5, 4])
D = np.array([6, 2])
E = np.array([5, 1])
F = np.array([2, 2])

# # 초기에 임의로 선택한 점 P
P1 = np.array([1.5, 3.5], dtype=float)
P2 = np.array([3.16, 3.5], dtype=float)
#P3,P4는 보노로이 점은 아니지만 수직성은 만족하도록 적당히 세팅하였다.
P3 = np.array([3.495, 3], dtype=float)
P4 = np.array([3.665, 3], dtype=float)

# 학습률과 에폭 설정
learning_rate = 0.01 #점차적 감소
epochs = 1000

# 경사하강법으로 P 업데이트
for epoch in range(epochs):
    # 세 변에 수직이 되는 반직선의 벡터
    AB_per = perpendicular_vector(A, B, P1)
    BC_per = perpendicular_vector(B, C, P2)
    CD_per = perpendicular_vector(C, D, P4)
    DE_per = perpendicular_vector(D, E, P4)
    EF_per = perpendicular_vector(E, F, P3)
    FA_per = perpendicular_vector(F, A, P1)

    points_edge = [np.array([0, 0]), np.array([6, 0]), np.array([0, 6]), np.array([6, 6])]

    # 정사각형 경계와의 교점 및 넓이계산
    p_AB=find_intersection(P1, AB_per)
    p_BC=find_intersection(P2, BC_per)
    p_CD=find_intersection(P4, CD_per)
    p_DE=find_intersection(P4, DE_per)
    p_EF = find_intersection(P3, EF_per)
    p_FA = find_intersection(P1, FA_per)

    area_A = calculate_convex_polygon_area([[0, 6], p_AB, P1, p_FA])
    area_B = calculate_convex_polygon_area([p_AB, p_BC, P2, P1])
    area_C = calculate_convex_polygon_area([p_BC, [6,6], p_CD, P4, P3, P2])
    area_D = calculate_convex_polygon_area([p_CD, p_DE, P4])
    area_E = calculate_convex_polygon_area([P4, p_DE,[6,0],p_EF, P3])
    area_F = calculate_convex_polygon_area([P1, P2, P3, p_EF, [0,0], p_FA])

    #gradient 계산
    gradient_A = (6 - area_A)
    gradient_B = (6 - area_B)
    gradient_C = (6 - area_C)
    gradient_D = (6 - area_D)
    gradient_E = (6 - area_E)
    gradient_F = (6 - area_F)
    # P 업데이트
    #for C, F
    vector_p1_to_p3 = np.array([P3[0] - P1[0], P3[1] - P1[1]])
    vector_p4_to_p2 = np.array([P4[0] - P2[0], P4[1] - P2[1]])
    #for B, E
    vector_p2_to_p3 = np.array([P3[0] - P2[0], P3[1] - P2[1]])
    learning_rate*=1001/1000
    #F피드백
    P1 = translate_perpendicular_only(P1, vector_p1_to_p3, gradient_F*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p1_to_p3, gradient_F*learning_rate)
    P3 = translate_perpendicular_only(P3, vector_p1_to_p3, gradient_F*learning_rate)
    P4 = translate_perpendicular_only(P4, vector_p1_to_p3, gradient_F * learning_rate)

    #C피드백
    P1 = translate_perpendicular_only(P1, vector_p4_to_p2, gradient_C*learning_rate)
    P2 = translate_perpendicular_only(P2, vector_p4_to_p2, gradient_C * learning_rate)
    P3 = translate_perpendicular_only(P3, vector_p4_to_p2, gradient_C * learning_rate)
    P4 = translate_perpendicular_only(P4, vector_p4_to_p2, gradient_C * learning_rate)
    learning_rate *= 1000/1001
    #B피드백
    P1 = translate_along_byvector(P1, vector_p2_to_p3, gradient_B*learning_rate)
    P2 = translate_along_byvector(P2, vector_p2_to_p3, gradient_B * learning_rate)

    # E피드백
    P3 = translate_along_byvector(P3, vector_p2_to_p3, -gradient_E * learning_rate)
    P4 = translate_along_byvector(P4, vector_p2_to_p3, -gradient_E * learning_rate)
    # A,D
    P1 = translate_along_vector(P1, P2, gradient_A*learning_rate)
    P4 = translate_along_vector(P4, P3, gradient_D*learning_rate)

    # 손실 출력
    loss = (6 - area_A) ** 2 + (6 - area_B) ** 2 + (6 - area_C) ** 2 + (6 - area_D) ** 2 + (6 - area_E)**2 + (6 - area_F)**2
    print(
        f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, P1: {P1}, P2: {P2}, P3 : {P3} , P4 : {P4}\n Area A : {area_A}, Area B : {area_B}, Area C : {area_C}, Area D: {area_D}, Area E : {area_E}, Area F : {area_F}')
    # if epoch<=50:
    #   if epoch<=10 or epoch%10==0:
    #     plt.plot([P1[0]],[P1[1]],'ro')
    #     plt.text(P1[0], P1[1], f'{epoch}', fontsize=7, ha='right')
    #     plt.plot([P2[0]],[P2[1]],'ro')
    #     plt.text(P2[0], P2[1], f'{epoch}', fontsize=7, ha='right')

plot_line(p_FA,P1)
plot_line(p_AB,P1)
plot_line(p_DE,P4)
plot_line(p_BC,P2)
plot_line(p_CD,P4)
plot_line(p_EF,P3)
#
plot_line(A,B)
plot_line(B,C)
plot_line(C,D)
plot_line(D,E)
plot_line(E,F)
plot_line(F,A)
plot_line(P1,P2)
plot_line(P2,P3)
plot_line(P3,P4)
plot_line(B,F)
plot_line(F,C)
plot_line(C,E)
#
#
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.axis('equal')
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()