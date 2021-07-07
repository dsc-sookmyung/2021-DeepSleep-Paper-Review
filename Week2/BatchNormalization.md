## Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

### 개요

- 내용

  - Batch Normalization을 사용하여 internal covariate shift를 줄이고, 심층 신경망 학습을 빠르게 함

- 배경

  - Stochastic gradient descent (SGD) 가 deep network 학습에 효과적이라고 증명되었지만, 
    문제가 있음

  - Internal covariate shift : 학습 중에 이전 layer의 parameter들이 변하므로, 각 input layer의 분포가 변하는 현상

    층이 깊어질 수록 변화가 커져 학습이 어려워짐

  - Internal covariate shift를 완화시키기 위해 Batch Normalization을 사용

  <img src="/Users/seohui/Library/Application Support/typora-user-images/스크린샷 2021-07-07 오후 9.15.46.png" alt="스크린샷 2021-07-07 오후 9.15.46" style="zoom:25%;" /> <img src="/Users/seohui/Library/Application Support/typora-user-images/스크린샷 2021-07-07 오후 9.17.20.png" alt="스크린샷 2021-07-07 오후 9.17.20" style="zoom:25%;" />

- 주요 내용

  - Batch Normalization : normalization을 모델 구조의 일부로 만들고, training mini-batch 별로 normalization 수행
  - Batch Normalization으로 Internal covariate shift 감소
  - Batch Normalization 장점
    - gradient vanishing / exploding 문제 발생 없이, 더 큰 learning rates 사용 가능
    - weight 초기화에 신경을 덜 써도 됨
    - regularizer의 역할도 해서, Dropout의 필요성 낮아짐
  - 성과

- 주요 용어

  - SGD
  - mini-batch
  - internal covariate shift
  - whitening



### Internal Covariate Shift (내부 공변량 변화)

- 개념

  - 심층 신경망의 internal 노드 분포의 변화

     <img src="/Users/seohui/Library/Application Support/typora-user-images/스크린샷 2021-07-07 오후 9.15.46.png" alt="스크린샷 2021-07-07 오후 9.15.46" style="zoom:25%;" />

- 단점

  - nonlinearity의 saturated regime(w의 업데이트가 멈추는 구간)에 빠져서 수렴을 늦춤
  - vanishing / exploding gradients

  이 단점은 ReLU (Rectified Linear Units) 함수 사용, 신중한 초기값 선택, 작은 learninig rates로 일반적으로 해결

  하지만, nonlinearity inputs 분포는 신경망 학습 시에 더 안정적이어서 saturated regime에 빠질 확률이 적으며 학습이 빨라질 것



### Internal covariate shift를 감소시키는 방법

#### Whitening the layer inputs

- Whitening 
  - 기본적으로 들어오는 input의 feature들을 uncorrelated하게 만들어주고, 각각의 분산을 1로 만들어주는 작업
- 의의
  - 각 층의 입력값을 whitening하면 입력값의 고정된 분포를 갖게 되어, internal covariate shift의 나쁜 영향을 없앨 수 있을 것 

- 단점
  - 비용이 많이 듦
  - 모든 곳에서 미분 가능하지 않음



### Full whitening의 문제점을 해결하기 위한 두 가지 Simplifications

#### First Simplification

입력 층과 출력 층의 features를 동시에 whitening 하는 대신에, 각 scalar feature를 평균 0, 분산 1로 독립적으로 정규화

<img src="/Users/seohui/Downloads/IMG_92EE698A627F-1.jpeg" alt="IMG_92EE698A627F-1" style="zoom:10%;" />

- 문제점 1 

  - 각 입력층을 단순히 정규화하는 것은 각 층이 나타내는 것을 변경할 수 있음

    ex) sigmoid의 입력값을 정규화하면 nonlinearity 의 linear regime으로 제한할 수 있음

- 해결책 1

  - make sure that the transformation inserted in the network can represent the identity transform

  - 정규화된 값을 scale & shift

    <img src="/Users/seohui/Downloads/IMG_CEC356F02687-1.jpeg" alt="IMG_CEC356F02687-1" style="zoom:10%;" />

- 문제점 2

  - stochastic optimization을 사용할 때 실용적이지 않음

- 해결책 2

  - second simplification

#### Second simplification

SGD에서 mini-batch를 사용하므로, 각 mini-batch가 각 activation의 평균과 분산 추정치를 만들어 냄
이 방법으로 정규화에 사용된 통계값은 모두 backpropagation에 참여할 수 있음

<img src="/Users/seohui/Downloads/IMG_DECFC4D96C29-1.jpeg" alt="IMG_DECFC4D96C29-1" style="zoom: 33%;" />

BN transform은 미분가능한 transform이며, 정규화된 activations를 network에 갖고 옴

- 적은 internal covariate shift로 학습을 가속화함
- BN transform이 identity transformation을 나타내고, network capacity를 유지할 수 있게 함



### 참고

- Training and Inference with Batch-Normalized Networks

  - mini-batch에 의한 activation 정규화는 효율적인 학습을 가능하게 하지만, 추론에는 바람직하지도 필요하지도 않음 (입력값에 대한 출력값만 필요할 뿐)
  - 한 번 network가 학습되면, mini-batch나 통계치가 아니라 population을 이용한 정규화 사용

  <img src="/Users/seohui/Downloads/IMG_E2BDF86B3927-1.jpeg" alt="IMG_E2BDF86B3927-1" style="zoom:10%;" />

- Batch-Normalized Convolution Networks
  - FC와 합성곱층 커버

- Batch Normalization enables higher learning rates

- Batch Normalization regularizes the model

- Experiements
