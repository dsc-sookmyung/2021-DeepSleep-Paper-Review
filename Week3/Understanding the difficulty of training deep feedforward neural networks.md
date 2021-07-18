- 제목: Understanding the difficulty of training deep feedforward neural networks
- 링크: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

- 주제: 딥러닝에서의 weight 초기화의 중요성
- 배경
  - 왜 랜덤 초기화에 따른 표준 경사하강법이 Deep neural network 학습에서 좋지 못한 성능을 보이는지에 대해 이해
  - Deep neural network에서의 비선형 활성함수의 (부정적인) 영향력 발견
- 내용
  - sigmoid, tanh, softsign을 사용한 네트워크의 각 레이어의 포화도 실험, 분석
  - 활성함수의 선택과 초기화 방법에 대한 평가

---



## Deep Neural Networks

- 딥러닝은 추출한 특징을 이용하여 특징 계층을 학습하는 것을 목표로 한다.

  - '추출한 특징'이란 낮은 수준의 특징들의 합성을 통해서 만들어진 높은 수준의 계층으로부터 추출한 것이다.

- 복잡한 기능의 학습은 <u>고수준의 추상화</u>를 필요로 하는데, 이를 위한 한 가지 방법이 `deep architecture` 다.

  ![그림 3.9](https://ars.els-cdn.com/content/image/3-s2.0-B9780128167182000105-gr009.jpg)

  이런 거..

- 최근 나온 deep architecture를 사용하는 실험 결과들은 `비지도 사전학습`의 효과로 인해 표준 랜덤 초기화와 그레디언트 기반의 optimization을 이용한 표준 지도학습에 비해서 훨씬 더 잘 작동한다.
  - 비지도 사전학습이 최적화 절차에서 regularizer 역할을 하는 덕분에 파라미터 초기화가 더 잘되기 때문이다.
  - 이것은 local minimum이 더 나은 generalization과 연관이 있다는 것을 보여준다.
- **이 논문에서는 깊이가 깊은 다중 인공 신경망이 어떻게 잘못될(잘못 학습될) 수 있는지에 대해 분석하며, 학습 중간에 네트워크의 각 레이어들의 activation과 gradients를 모니터링하고 분석한다. 그 후 (아마도 포화에 영향을 줄 거라고 예상되는) 활성함수의 선택과 초기화 방법에 대하여 평가한다.**





## Experimental Setting and Datasets

1. **유한 데이터셋(온라인 학습)**

deep architecture를 사용하게 되면 큰 학습 셋이나 온라인 학습을 진행할 때 비지도 학습을 통해 초기화를 하게 되면 학습 수가 늘어나도 실질적인 성능 향항이 일어난다. 이런 온라인 학습 시나리오를 테스트하기 위해 많은 예제를 샘플링할 수 있는 데이터셋을 선택했다.

- 32 x 32 크기의 1개 혹은 2개의 2차원 객체를 포함하고 있는 이미지, 삼각형 / 평행사변형 / 타원
- 분류 난이도를 높이기 위해 두 개의 객체로 이미지를 샘플링. 총 9개의 구성 클래스(3(1개) + 3 !(2개))

<img src="/Users/soo/Library/Application Support/typora-user-images/스크린샷 2021-07-16 오후 8.33.52.png" alt="스크린샷 2021-07-16 오후 8.33.52" style="zoom:50%;" />



2. **유한 데이터셋**

- MNIST 숫자 데이터셋

- CIFAR-10 데이터셋 중 10,000개

- Small-ImageNet

  <img src="/Users/soo/Library/Application Support/typora-user-images/스크린샷 2021-07-16 오후 8.40.28.png" alt="스크린샷 2021-07-16 오후 8.40.28" style="zoom:50%;" />



3. **실험 환경 설정**

- 1~5개의 히든 레이어를 가진 feedforward 인공 신경망을 최적화했다.

  - 이 신경망은 레이어 당 1,000개의 숨겨진 유닛을 갖는다.

  - **출력 레이어**는 `softmax logistic regression`을 사용했다.

  - **비용 함수**는 `negative log-likelihood`($-logP(y|x)$)를 사용했다.

    <img src="https://blog.kakaocdn.net/dn/bLuTuS/btqS4M9jFve/ER9MTvaChsQET6gegQaW11/img.png" alt="img" style="zoom:50%;" />

  - **활성함수**는 `sigmoid`, `hyperbolic tangent`, `softsign` 를 각 실험에 사용했으며, 각 모델별로 최고의 hyper-parameters를 찾았다.

  - **biases**는 0으로 초기화했다.
  
  - 각 레이어의 **가중치**는 휴리스틱 방법으로 초기화했다.  <img src="/Users/soo/Library/Application Support/typora-user-images/스크린샷 2021-07-16 오후 9.11.55.png" alt="스크린샷 2021-07-16 오후 9.11.55" style="zoom: 33%;" />





## Effect of Activation Functions and Saturation During Training

### Experiments with the Sigmoid

![img](https://t1.daumcdn.net/cfile/tistory/2366A537583780DE34)

1. 관찰
   - 마지막 레이어(Layer 4)의 활성값은 0으로 빠르게 이동하며 `포화`된다.
   - 다른 레이어들(Layer 1, 2, 3)의 평균 활성값은 0.5 이상이며, 낮은 레이어일수록 이 값은 감소한다.
2. 인사이트
   - 위와 같은 종류의 포화는 인공신경망의 깊이가 깊어질수록 더 오래 지속된다.
   - 이러한 동작은 <u>랜덤 초기화</u>와 <u>0을 출력하는 히든 레이어가 포화된 sigmoid 함수와 일치</u>한다는 점이 혼합되어 발생하는 것 같다. sigmoid를 사용하되, 초기화를 비지도 사전학습으로 진행한 신경망은 이런 포화 동작이 일어나지 않는다. (추가)





### Experiments with the Hyperbolic tangent

![img](https://t1.daumcdn.net/cfile/tistory/212A6C34583B78DB04)

![img](https://t1.daumcdn.net/cfile/tistory/221B0434583B78DB05)

1. 관찰
   - tanh는 0을 중심으로 대칭적이므로 최상위 히든 레이어의 포화 문제를 겪지 않는다.
   - 하지만 표준 가중치 초기화인 $U[-1/\sqrt{n}, 1/\sqrt{n}]$ 을 사용하게 되면 레이어 1에서부터 <u>순차적으로</u> 포화현상이 발생한다.
   - 극한(점근선 -1과 1) 또는 0 부근에서 활성화 분포의 양상을 보인다.
2. 인사이트
   - 나중에 아라보자...





### Experiments with the Softsign

![img](https://t1.daumcdn.net/cfile/tistory/24411B41583BA9911A)

![img](https://t1.daumcdn.net/cfile/tistory/226BD741583BA99227)



1. 관찰
   - tanh와는 다르게 포화가 순차적으로 발생하지 않는다.
   - 평평한 지역은 비선형성이 있지만 gradient가 잘 흐를 수 있는 지역이다.
   - 앞선 두개의 활성함수 그래프와는 다르게 히스토그램 형태이니 해석에 주의하자

<img src="https://paperswithcode.com/media/methods/Screen_Shot_2020-05-27_at_4.35.34_PM_07Nzs7R.png" alt="Softsign Activation Explained | Papers With Code" style="zoom:25%;" />







## Studying Gradients and their Propagation

### **Xavier Initialization 제안**

<img src="https://t1.daumcdn.net/cfile/tistory/27600D4F583FF3952A" alt="img" style="zoom:50%;" />

Bradly(2009)의 연구를 기반: 각 레이어가 linear activation으로 구성된 인공 신경망에서 신경망을 역으로(backward) 학습시킬 때마다 역전파된 그레디언트(back-propagated gradients)의 분산이 감소한다는 결과





### 성능 테스트 

Shapeset 3 x 2 사용, 다른 데이터셋을 사용한 결과도 아래의 결과와 질적으로 동일하다.

1. Activation value (tanh)

<img src="https://t1.daumcdn.net/cfile/tistory/23352F3A583E21401C" alt="img" style="zoom: 50%;" />

<img src="https://t1.daumcdn.net/cfile/tistory/27778F3A583E214129" alt="img" style="zoom: 50%;" />



2. Backpropagated gradients (tanh)

   <img src="https://t1.daumcdn.net/cfile/tistory/2324444F583E245815" alt="img" style="zoom:50%;" />

   <img src="https://t1.daumcdn.net/cfile/tistory/243B2E4F583E245814" alt="img" style="zoom:50%;" />

   - Bradely(2009)의 연구 결과처럼 표준 초기화 방법(위)을 사용할 시 back-propagated gradient의 분산이 아래로 전파됨에 따라서 더 작아지는 것을 볼 수 있다. 
   - 정규화된 초기화 방법(아래)을 사용할 시 분산은 모두 일정하게 유지된다.
   - 학습이 진행됨에 따라 gradient의 크기가 달라지게 되면 학습이 느려지고 좋지 못한 상태(ill-conditioning)를 갖게 되는데, 정규화된 초기화 방법은 gradient의 분산이 일정하게 유지되므로 이 부분에서 강점을 갖는다.

   



## Error Curves and Conclusions

<img src="https://t1.daumcdn.net/cfile/tistory/274ED049583FF6911D" alt="img" style="zoom:67%;" />

<center>5개의 히든 레이어를 사용한 Deep network에서<br/>서로 다른 활성함수와 서로 다른 초기화 방법을 사용했을 때 나온 테스트 에러 표</center>



<img src="https://t1.daumcdn.net/cfile/tistory/27544C4A5840F48D2F" alt="img" style="zoom: 50%;" />

<center>Error curves</center>

- <u>Sigmoid + 표준</u>, <u>tanh + 표준 초기화 방법</u> 조합은 상태가 좋지 않다. 수렴 속도가 느리며, local minima에 취약하다.
- softsign을 사용한 인공 신경망은 더 부드러운 비선형성으로 인해 tanh을 사용한 신경망에 비해 초기화 방법에 크게 영향을 받지 않는다.
- <u>Tanh + 정규화된 초기화 방법</u>은 꽤 유용할 수 있다. 레이어 간 변환이 활성화 및 그레디언트의 크기를 유지하기 때문일 것이다.



![스크린샷 2021-07-18 오전 12.58.24](/Users/soo/Library/Application Support/typora-user-images/스크린샷 2021-07-18 오전 12.58.24.png)

![img](https://t1.daumcdn.net/cfile/tistory/22489A49583FFE1908)

<center>tanh를 사용한 weights gradients 값의 변화(위: 표준, 아래: 정규화)</center>



### 결론

1. 레이어 전반에 걸친 activation과 gradient를 모니터링 하는 것과 반복 학습은 deep net의 학습이 왜 어려운지 이해하는 데 크게 도움이 된다.
2. 작은 랜덤 가중치 값으로 초기화할 때에는 Sigmoid 활성함수를 사용하지 말자. 최상위 히든 레이어의 포화로 인해 학습 능력이 좋지 않다.
3. activaion과 gradient가 모두 잘 흐르도록 레이어 간 변환을 진행하는 것이 도움이 되며, 비지도 사전학습을 통해 훈련된 네트워크와 순수하게 지도 학습으로 훈련된 네트워크 사이의 격차를 줄일 수 있다.

