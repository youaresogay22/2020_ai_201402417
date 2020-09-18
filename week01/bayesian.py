from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inlne

'''
Random하게 2개의 Gaussian 을이용해서 2-Class Dataset을 만든다.
예제와 모양이 다르게 만듬(center의 위치 및 분산)
'''
mean1 = [-1, 0]
cov1 = [[0.5, 0], [0, 0.5]]
mean2 = [1, 0]
cov2 = [[0.5, 0], [0, 0.5]]

x1 = np.random.multivariate_normal(mean1, cov1, 500)
x2 = np.random.multivariate_normal(mean2, cov2, 500)

y1 = np.ones(500)
y2 = -1 * np.ones(500)

'''
모양 확인 및 합치기
'''
print(x1.shape)
print(x2.shape)
print(y1.shape)
print(y2.shape)

datax = np.concatenate((x1, x2), axis=0)
datay = np.concatenate((y1, y2), axis=0)

'''
data set이 잘 만들어졌는지 print 및 matplotlib 사용하여 확인
'''
print(datax.shape)
print(datay.shape)

plt.scatter(datax[:, 0], datax[:, 1])
plt.scatter(x1[:, 0], x1[:, 1], c='red')
plt.show()

'''
requirement 1.Data를 train:test로 분할한다.
'''
n_data = datax.shape[0]
p_trn = 0.7
n_trn = round(n_data*p_trn)
print(n_data, p_trn, n_trn)

idx_array = np.array(range(0, n_data))
idx_array_perm = np.random.permutation(idx_array)
print(idx_array_perm[0:100])

trnx = datax[idx_array_perm[0:n_trn], :]
trny = datay[idx_array_perm[0:n_trn]]
tstx = datax[idx_array_perm[n_trn:n_data], :]
tsty = datay[idx_array_perm[n_trn:n_data]]
'''
Permutation 함수는 뒤섞어주는 역할을 한다.
인덱스 배열을 생성해서 섞어준 후 앞에서 부터 끊어서
학습/평가로 분할하는 방법을 사용
'''

'''
잘 분할되었는지 확인
'''
print(trnx.shape)
print(tstx.shape)
print(trny.shape)
print(tsty.shape)

'''
requirement 2
Sklearn의 GaussinNB 함수를 이용해서 Classifier A를 만든다.
'''
clf = GaussianNB()
clf.fit(trnx, trny)
tsty_hat = clf.predict(tstx)
'''
Classifier 객체를 만들고 학습 데이터를 넣어서 fit하면
그 객체가 tuning이 된다. 
그 후 tuning된 객체로 Test를 진행할 수 있음
'''


'''
제대로 predict label이 나왔는지 확인
(가끔 1, 2번 클래스 혹은 1, 0번 클래스 같이 Label 명이 뒤섞이는 경우가 있음)
'''
print(tsty_hat)

'''
requirement 4. A에 대한 Accuracy 계산 파트
'''
dif = tsty-tsty_hat
accuracy = 1 - (np.size(np.where(dif != 0)) / np.size(tsty))
print("Accuracy of Classifier A:", accuracy)



'''
requirement 5. A의 test data에 대한 실제 label 및 
모델의 예측 결과를 그림으로 표현한다.
'''
plt.scatter(tstx[:, 0], tstx[:, 1])
plt.scatter(tstx[np.where(tsty == 1), 0],
            tstx[np.where(tsty == 1), 1], c='red')
plt.scatter(tstx[np.where(tsty_hat == 1), 0],
            tstx[np.where(tsty_hat == 1), 1], c='red', marker='x', s=50)
plt.scatter(tstx[np.where(tsty_hat == -1), 0],
            tstx[np.where(tsty_hat == -1), 1], c='blue', marker='x', s=50)
plt.show()

'''
np.where 함수는 특정 값을 가지는 Index를 반환해줌
실제 label(동그라미)와 예측된 레이블(X 마크)를 동시에 그림
어느 지역의 데이터에 대해 모델이 틀리고 있는지 알 수 있음
'''

'''
requirement 3. Prior를 변경해서 또 다른 Classifier B를 만든다.
Prior의 역할 때문에 빨간색으로 예측되는 영역이 늘어나게 됨
'''
clf2 = GaussianNB(priors=[0.1, 0.9])
clf2.fit(trnx, trny)
tsty_hat2 = clf2.predict(tstx)

'''
requirement 4. B에 대한 Accuracy 계산 파트
'''
dif = tsty-tsty_hat2
accuracy = 1 - (np.size(np.where(dif != 0)) / np.size(tsty))
print("Accuracy of Classifier B:",accuracy)

'''
requirement 5. B의 test data에 대한 실제 label 및 
모델의 예측 결과를 그림으로 표현한다.
'''
plt.scatter(tstx[:, 0], tstx[:, 1])
plt.scatter(tstx[np.where(tsty == 1), 0],
            tstx[np.where(tsty == 1), 1], c='red')
plt.scatter(tstx[np.where(tsty_hat2 == 1), 0],
            tstx[np.where(tsty_hat2 == 1), 1], c='red', marker='x', s=50)
plt.scatter(tstx[np.where(tsty_hat2 == -1), 0],
            tstx[np.where(tsty_hat2 == -1), 1], c='blue', marker='x', s=50)
plt.show()


