# 09.MAIC-Sleep-Staging-Classification
Repository for my experience for MAIC Sleep Staging Classification contest

## 느낀 점
이번 대회는 MAIC 및 서울대학교 병원에서 주최하는 수면 단계 분류 대회였다.

빅데이터 교육과정에서 만난 최성민씨와 다른 두 명이 합쳐 4명이서 팀을 이루어 참가하게 되었다. 참가신청한 팀은 110팀 가까이 되었으며 
이 중에 40팀만이 제한된 서버 자원을 사용할 수 있는 자격을 획득하여 경쟁하게 되는 구조였다. 다행히 우리 팀은 40팀 안에 포함되어 이후
1주일 동안 서버를 사용할 수 있게 되었다.

주어진 문제 상황은 쉽게 말하면 image sequence로부터 label sequence를 학습시킬 수 있는가하는 것이었다. 각각의 환자들에 대해 적용한
수면 다원성 검사 결과는 대략 20여개의 시그널들이 그래프 형식으로 주어지고, 이 그래프들이 시간에 따라 변화하는 양상을 이미지 시퀀스로 표현된 것이었다.
이로부터 각 이미지에 대응하는 수면 단계를 머신러닝을 통해 예측할 수 있는가가 문제였다. 
이 문제는 이전에 경험했던 수력댐기후예측 대회문제와 상당히 유사했다. 즉, 각각의 이미지를 판별하는 것보다는 전체 이미지 시퀀스가 최소단위의 데이터로 간주되고
이로부터 여러개의 라벨을 예측해야 하는 문제였다. 다시 말해 이미지 예측에 특화된 Convolution 과 시계열 예측에 특화된 LSTM 을 잘 조합해서 풀어야 하는 문제였다.

