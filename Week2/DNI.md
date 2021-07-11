# Decoupled Neural Interfaces using Synthetic Gradients



- [Introduction](#Introduction)
- [Decoupled Neural Interfaces](#Decoupled-Neural-Interfaces)
- [Experiments](#Experiments)


## Introduction
- 내용: 작업 멈춤(locking) 현상을 억제하기 위해 신경망 레이어 간 데이터 주입 및 그래디언트 갱신을 비동기적으로 처리할 것을 제안

- 배경
  - 기존 방향성 신경망에는 3가지 locking이 존재
  - 신경망의 한 레이어에서 데이터를 처리할 때 다른 레이어가 작업을 하지 못 하는 상태를 `locking`이라고 표현함
  - `Forward Locking`: 어떤 모듈도 이전 노드에 입력 데이터가 들어오기 전까지는 작업 시작이 불가함
  - `Update Locking`: forward 과정이 끝나기 전까지는 갱신 불가함
  - `Backwards Locking`: forward, backwards 과정이 끝나기 전까지는 갱신 불가함

- 문제
  - 분산 환경이나 레이어에 비동기 방식을 도입하기가 어려움
  - 거대 신경망의 경우 레이어 간의 동기성이 신경망의 학습 속도에 영향을 미침

- 해결
  - 레이어 간의 데이터 전달을 비동기적으로 처리
  - 마지막 레이어에서 에러가 도출되어 역전파 되기를 기다리는 것이 아니라, 각 레이어에서 출력 데이터가 도출되면 이를 갖고 합성 그래디언트를 만들어 레이어의 파라미터를 학습시킴





### Synthetic Gradients
<img src="https://norman3.github.io/papers/images/synthetic_gradients/f01.png" width="600" height="300">



<img src="https://user-images.githubusercontent.com/65005334/125182719-62074400-e24b-11eb-8781-f82a6f97d8cd.png" width="400" height="150">

=> 기존 back propagation

<img src="https://media-exp1.licdn.com/dms/image/C5112AQFmholpGTx6gw/article-inline_image-shrink_1000_1488/0/1524840032813?e=1628726400&v=beta&t=m9mqrhIf-3jmIBzX_41lpyk75-ncgr_-wlQ6ilRew9c" width="400" height="150">



## Decoupled Neural Interfaces
<img src="https://lh3.googleusercontent.com/AW6Mbmx6RLyxDA8y8YKvAMjrEpBQZX_DSQt-gw2GgsVn_BRUILgmqyBR91izb0AKSP7GLckNzqhOMCwsysqcuiXs9zoBr6vKDFMonbk=w2048-rw-v1" width="300" height="300">

- 동작 설명
  - 출력 데이터가 상위 레이어와 그래디언트 모델의 입력 데이터로 전달
  - 그래디언트 모델은 즉각적으로 그래디언트 계산하여 입력을 전달한 레이어의 파라메타를 업데이트함
  - 이는 다시 하위 레이어로 역전파됨(타겟 그래디언트)
  - 타겟 그래디언트를 갖고 현재 그래디언트 모델을 학습시킴
  - 이 과정이 반복되고, 최종 출력의 그래디언트가 충분히 전달될 쯤에는 그래디언트는 상당히 정확한 합성 그래디언트를 추측할 수 있음


  ### Synthetic Gradient for Feed-Forward Networks
  <img src="https://norman3.github.io/papers/images/synthetic_gradients/f06.png" width="500" height="300">

  - 그림 설명
    - 첫번째 레이어인 fi 가 출력 hi를 Mi + 1에 전달하면 합성 그래디언트인 δ^i 를 바로 fi 에게 넘겨줌
    - fi 는 forward 진행과는 무관하게 바로 backprop 갱신을 수행할 수 있게됨
    - fi + 1는 이전 레이어와 동일하게 hi + 1을 Mi + 2 로 전달하여 δ^i + 1 을 얻음
    - δ^i + 1 를 이용하여 δi 을 계산하고 Mi + 1을 업데이트함
    - 이 과정을 계속 반복


### Synthetic Gradient for Recurrent Networks

- 무한히 전개되는 RNN

<img src="https://norman3.github.io/papers/images/synthetic_gradients/f07.png" width="600" height="150">

<img src="https://norman3.github.io/papers/images/synthetic_gradients/f08.png" width="600" height="150">

- 원래 backpropagation throgh time(BPTT)는 그래디언트가 무한대로 발산하거나, 무한히 작아질 수 있는 문제가 있음(= vanishing and exploding gradients)
- 따라서 아래 그림과 BPTT를 일정한 단위(스텝 t에서부터 T까지)로 쪼개어 계산(= truncated BPTT)

<img src="https://user-images.githubusercontent.com/65005334/125182736-8400c680-e24b-11eb-9791-20f5dd9ef709.png" width="600" height="150">

- 임의의 T값을 정한 뒤(위의 그림의 경우는 3) δT=0  로 가정하고 식을 전개함
- 이는 BPTT의 경계를 넘어서는 부분에서는 그래디언트를 전파시킬 수 없음

<img src="https://lh3.googleusercontent.com/It2FZ2smeMjHuhb6jIisLKfY6R4KgUXNb926pPckVsy6Mhn4b4iKpiDsDGda4zZxHkBWqiatfYJkAGXWDeXcl08OZshCtBnB4aIuiw=w2048-rw-v1" width="500" height="100">

- 그러나 합성 그래디언트를 사용하면 δT를 근사할 수 있음
- 즉 순환 신경망의 경우, 합성 그래디언트를 사용하면 비동기화뿐만 아니라 BPTT의 경계를 넘어 신경망이 펼쳐질 수 있다는 장점이 존재



## Experiments

- Results for applying DNI to RNNs

<img src="https://norman3.github.io/papers/images/synthetic_gradients/f14.png" width="600" height="200">


- 모두 LSTM 사용

- 주제: Copy and Repeat Copy task

- Copy, Repeat Copy: 해당 T(shifting size)를 사용하였을 때, 실제 복원되는 문자열의 길이를 의미

- Penn Treebank: 에러값