# Sequence to Sequence Learning with Neural Networks
- 링크: 
- 주제: 두 개의 **LSTM** 모델을 활용한 기존 기계 번역의 한계 개선
- 배경: 기존의 SMT와 RNN 방식으로는 **긴 문장 처리와 어순 구조 변화에 대응**하기 어렵다는 단점이 존재
- 내용
	1. Encoder와 Decoder, 즉 2개의 LSTM을 사용하고 이를 다시 4개의 layer로 쌓아 모델 생성
	=> 파라미터의 수가 늘어나 깊은 학습 가능
	2. 입력 문장의 순서를 뒤집었을 때 결과가 더 좋음



## 배경
- **통계적 기계 번역(Statistical Machine Translation)**
	어떤 단어가 특정 단어로 번역될 통계적 확률을 바탕으로 번역을 수행
	`=> ex) P(나는 학교에서 공부한다 | I study at school) > P(나는 학교를 공부한다 | I study at school)`
	- 모든 문장의 확률을 구해야 하기 때문에 너무 많은 양의 데이터를 요구
	- 긴 문장 번역 시, 연쇄 법칙에 의해 최종 문장의 확률값은 0에 수렴할 가능성이 높음
	=> P(I think the vacation is too short) = P(I) * P(think|I) * P(the|I think) * P(vacation|I think the) * P(is|I think the vacation) * P(too|I think the semester is) * `P(short|I think the vacation is too)`

- **전통적인 RNN**
	- 입력과 출력의 크기가 같다고 가정 & 입력 문장 전체를 바라보고 해석하지 않고 단어 토큰을 하나하나 해석
	=> `How are you?` => `잘 지내?` : 입력 출력 크기 불일치
	=> `I eat lunch` => `나는 점심을 먹는다` :  어순 구조 불일치

## Sequence to Sequence
**특징: 고정된 크기의 `context vector`를 활용해 번역**
![img](https://miro.medium.com/max/1838/0*zvuIAdb3pBGENWMW.)

![img](https://www.programmersought.com/images/614/5f460595484caf7a27bfb0ab27ef1bee.png)
- `context vector = last hidden state`
- 디코더가 context vector를 기반으로 번역을 실행함
	 =>  긴 문장이어도 확률값이 0이 될 걱정 X
	 => 단어 하나하나마다 해석하는 것이 아닌 문장을 토대로 해석
- Valina RNN이 아닌 **LSTM 구조**를 활용해 context vector 추출
- **Encoder, Decoder 는 서로 다른 파라미터**를 가지며, 본 논문에서는 **4개의 layer**를 쌓았을 때 가장 성과가 좋았다고 함


![img](https://www.researchgate.net/profile/Anbang-Xu/publication/313204805/figure/fig1/AS:457086996881408@1485989440751/Sequence-to-sequence-learning-with-LSTM-neural-networks.png)

**- 입력 문장의 순서를 뒤바꿨을 때 더 높은 정확도가 나옴**
=> 출력 문장은 문장 앞 부분부터 나오는데, 만약 입력 문장의 순서를 바꾸지 않았다면 context vector에는 입력 문장의 앞부분은 이미 많이 희석된 상태가 됨
=> 따라서 입력 문장의 순서를 거꾸로 뒤집어서 문장 앞부분의 정보를 더 많이 담는 것이 모델이 학습하는 데는 더 수월하다고 함

## Experiments
- 영어/프랑스 문장 쌍들에 대해 실험(`WMT'14 English to French dataset`)
- source언어(번역할 언어): 160,000개 단어를 사용, target 언어(번역될 언어): 80,000개의 단어를 사용
- **batch 마다 비슷한 length를 가진 sentence**가 포함되도록 normalization을 수행한 뒤 실험 진행

![img](https://cpm0722.github.io/assets/images/2020-05-10-Sequence-to-Sequence-Learning-with-Neural-Networks/03.jpg)
![img](https://user-images.githubusercontent.com/25279765/35660574-29684b96-0750-11e8-9409-6502ff4f26d6.jpg)

## 결론
- 논문이 나온 2014 기준, SeqToSeq 방식의 두개의 LSTM과 context vector는  SMT 기법과 전통적인 RNN translation 모델의 한계를 넘어선다는 점에서 의의가 있음
- 그러나 **context vector의 크기가 고정**되어 있다는 점에서 vector의 크기보다 긴 문장이 들어오면 입력문장의 데이터를 온전히 번역할 수 없다는 단점이 존재함
- 이 논문 이후 등장한 Attention 방식과 거기서 확장된 **Transformer**를 통해 2021 현재는 **입력 sequence 전체에서 정보를 추출**하는 경향으로 흘러감