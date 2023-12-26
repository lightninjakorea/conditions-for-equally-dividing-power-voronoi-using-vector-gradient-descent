from calculate import *
from square import *

A = np.array([2, 1])
B = np.array([5, 3])
C = np.array([1, 5])

# # 초기에 임의로 선택한 점 P
P = np.array([2,2], dtype=float)

#학습률과 에폭 설정
learning_rate = 0.01
epochs = 1000
i=0
# 경사하강법으로 P 업데이트
for epoch in range(epochs):
    # 세 변에 수직이 되는 반직선의 벡터
    AB_per = perpendicular_vector(A,B,P)
    AC_per = perpendicular_vector(A,C,P)
    BC_per = perpendicular_vector(B,C,P)

    points_edge=[np.array([0,0]), np.array([6,0]), np.array([0,6]), np.array([6,6])]

    # 정사각형 경계와의 교점
    p_AB=find_intersection(P, AB_per)
    p_BC=find_intersection(P, BC_per)
    p_AC=find_intersection(P, AC_per)

    area_AB=calculate_convex_polygon_area([[0 , 6],p_BC, P, p_AC])
    area_BC=calculate_convex_polygon_area([p_AB, [0,0], p_AC, P])
    area_AC = calculate_convex_polygon_area([P, p_AB, [6,0], [6,6], p_BC])

    #P로부터 세 변에 수직이 되는 반직선의 방향벡터에 대한 gradient 계산
    gradient_AB = AB_per/np.linalg.norm(AB_per) * -(12-area_AB)
    gradient_AC = AC_per/np.linalg.norm(AC_per) * -(12-area_AC)
    gradient_BC = BC_per/np.linalg.norm(BC_per) * -(12-area_BC)

    gradient = gradient_AB + gradient_AC + gradient_BC

    # P 업데이트
    P = P - learning_rate * gradient

    # 손실 출력
    loss = (12-area_AB)**2+(12-area_AC)**2+(12-area_BC)**2
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}, P: {P}, Area AB : {area_AB}, Area BC : {area_BC}, Area AC : {area_AC}')
    if epoch<=50:
      if epoch<=10 or epoch%10==0:
        plt.plot([P[0]],[P[1]],'ro')
        plt.text(P[0], P[1], f'{i}', fontsize=7, ha='right')
        i+=1
print(AB_per,BC_per,AC_per)

w1=3
AB_H=per_point(A,B,P)
BC_H=per_point(B,C,P)
CA_H=per_point(C,A,P)

w2=(w1**2-(calculate_distance(A,B)**2)*(2*calculate_internal_ratio(A,B,AB_H)-1))**(1/2)
w3=(w2**2-(calculate_distance(B,C)**2)*(2*calculate_internal_ratio(B,C,BC_H)-1))**(1/2)
print(round(w1,1),round(w2,1),round(w3,1))

P__AB=intersection(AB_per, P, A,B)
P__BC=intersection(BC_per, P, B,C)
P__CA=intersection(AC_per, P, C,A)
print(P__AB,P__BC,P__CA)

plt.quiver(*P,*-(P-P__AB),scale=1)
plt.quiver(*P,*-(P-P__BC),scale=1)
plt.quiver(*P,*-(P-P__CA),scale=1)
plot_line(A,B)
plot_line(B,C)
plot_line(C,A)
plot_circle(A, w1)
plot_circle(B, w2)
plot_circle(C, w3)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.axis('equal')
plt.xlim([0,6])
plt.ylim([0,6])
plt.show()


w1 = 3
AB_H = per_point(A, B, P)
BC_H = per_point(B, C, P)
CA_H = per_point(C, A, P)

w2 = (w1 ** 2 - (calculate_distance(A, B) ** 2) * (2 * calculate_internal_ratio(A, B, AB_H) - 1)) ** (1 / 2)
w3 = (w2 ** 2 - (calculate_distance(B, C) ** 2) * (2 * calculate_internal_ratio(B, C, BC_H) - 1)) ** (1 / 2)
print(round(w1, 1), round(w2, 1), round(w3, 1))

P__AB = intersection(AB_per, P, A, B)
P__BC = intersection(BC_per, P, B, C)
P__CA = intersection(AC_per, P, C, A)
print(P__AB, P__BC, P__CA)

plt.quiver(*P, *-(P - P__AB), scale=1)
plt.quiver(*P, *-(P - P__BC), scale=1)
plt.quiver(*P, *-(P - P__CA), scale=1)
plot_line(A, B)
plot_line(B, C)
plot_line(C, A)
plot_circle(A, w1)
plot_circle(B, w2)
plot_circle(C, w3)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))
plt.axis('equal')
plt.xlim([0, 6])
plt.ylim([0, 6])
plt.show()
