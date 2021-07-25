# NLP

자연어(natural language)란 우리가 일상 생활에서 사용하는 언어를 말한다. 자연어 처리(natural language processing)란 이러한 자연어의 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 일을 말한다.

## 목차

- NLP Pipeline
- 언어 모델 (Language Model)
  - 통계적 언어 모델
  - 인공 신경망 모델
    - RNN
    - LSTM
    - GRU
    - Seq2Seq
    - Attention Mechanism
    - Transformer
- 단어의 표현 방법 (Word Representation)
- 임베딩 기법
- 참고 자료

## NLP Pipeline

![img](https://miro.medium.com/max/1400/1*CbzCcP3XFtYVJmWowZLugQ.png)

_출처 [https://medium.com/mlearning-ai/basic-steps-in-natural-language-processing-pipeline](https://medium.com/mlearning-ai/basic-steps-in-natural-language-processing-pipeline-763cd299dd99)_

## 언어 모델 (Language Model)

언어 모델(Language Model, LM)은 언어라는 현상을 모델링하고자 단어 시퀀스(또는 문장)에 확률을 할당(assign)하는 모델이다.

언어 모델을 만드는 방법은 크게는 **통계를 이용한 방법**과 **인공 신경망을 이용한 방법**으로 구분할 수 있다. 통계에 기반한 전통적인 언어 모델(Statistical Languagel Model, SLM)은 우리가 실제 사용하는 자연어를 근사하기에는 많은 한계가 있었고, 요즘 들어 인공 신경망이 그러한 한계를 많이 해결해주면서 통계 기반 언어 모델은 사용 용도가 많이 줄었다. 

### 통계적 언어 모델

통계 기반 언어 모델의 n-gram은 자연어 처리 분야에서 활발하게 활용되고 있으며, 통계 기반 방법론에 대한 이해는 언어 모델에 대한 전체적인 시야를 갖는 일에 도움이 된다.

#### N-gram

n-gram은 n개의 연속적인 단어 나열을 의미한다. 갖고 있는 코퍼스에서 n개의 단어 뭉치 단위로 끊어서 이를 하나의 토큰으로 간주한다. 예를 들어서 문장 An adorable little boy is spreading smiles이 있을 때, 각 n에 대해서 n-gram을 전부 구해보면 다음과 같다.

**uni**grams : an, adorable, little, boy, is, spreading, smiles
**bi**grams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
**tri**grams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
**4**-grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

### 인공 신경망 모델

피드 포워드 신경망은 입력의 길이가 고정되어 있어 자연어 처리를 위한 신경망으로는 한계가 있었다. 결국 다양한 길이의 입력 `시퀀스`를 처리할 수 있는 인공 신경망이 필요하게 되었는데, 자연어 처리에 대표적으로 사용되는 인공 신경망으로는 RNN, LSTM 등이 있다.

#### 순환 신경망(Recurrent Neural Network, RNN)

RNN(Recurrent Neural Network)에 대한 기본적인 아이디어는 순차적인 정보를 처리한다는 데 있다. RNN은 입력과 출력을 시퀀스 단위로 처리하는 `시퀀스(Sequence) 모델`이다. RNN이 *recurrent* 하다고 불리는 이유는 동일한 태스크를 한 시퀀스의 모든 요소마다 적용하고, 출력 결과는 이전의 계산 결과에 영향을 받기 때문이다. 다른 방식으로 생각해 보자면, RNN은 현재지 계산된 결과에 대한 "메모리" 정보를 갖고 있다고 볼 수도 있다. 이론적으로 RNN은 임의의 길이의 시퀀스 정보를 처리할 수 있지만, 실제로는 비교적 짧은 시퀀스만 효과적으로 처리할 수 있다. 

RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고 있다.

일반적인 RNN 구조는 다음과 같이 생겼다.

![rnn_unfolding](http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg)

_출처 http://aikorea.org/blog/rnn-tutorial-1/_

위 그림에서 RNN의 recurrent한 연결이 펼쳐진 것을 볼 수 있다. RNN 네트워크를 "펼친다"는 말은 간단히 말해서 네트워크를 전체 시퀀스에 대해 그려놓았다고 보면 된다. 즉, 우리가 관심있는 시퀀스 정보가 5개의 단어로 이루어진 문장이라면, RNN 네트워크는 한 단어당 하나의 layer씩 (recurrent 연결이 없는, 또는 사이클이 없는) 5-layer 신경망 구조로 펼쳐질 것이다. 
_Cf. word based language model, character based language model_

RNN 구조에서 일어나는 계산에 대한 식은 아래와 같다.

- ![x_t](http://s0.wp.com/latex.php?latex=x_t&bg=ffffff&fg=000&s=0)는 시간 스텝(time step) ![t](http://s0.wp.com/latex.php?latex=t&bg=ffffff&fg=000&s=0)에서의 입력값이다. 

- ![s_t](http://s0.wp.com/latex.php?latex=s_t&bg=ffffff&fg=000&s=0)는 시간 스텝 ![t](http://s0.wp.com/latex.php?latex=t&bg=ffffff&fg=000&s=0)에서의 hidden state이다. 네트워크의 "메모리" 부분으로서, 이전 시간 스텝의 hiddent state 값과 현재 시간 스텝의 입력값에 의해 계산된다: 

  <img width="447" alt="RNN-계산식" src="https://user-images.githubusercontent.com/53266682/126878499-e2cc8cdf-1605-425c-8c62-abb2d6b2ca68.png">

  _출처 https://www.boostcourse.org/ai212/lecture/43749?isDesc=false_

  비선형 함수 ![f](http://s0.wp.com/latex.php?latex=f&bg=ffffff&fg=000&s=0)는 보통 [tanh](https://reference.wolfram.com/language/ref/Tanh.html)나 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))가 사용되고, 첫 hidden state를 계산하기 위한 ![s_-1](http://s0.wp.com/latex.php?latex=s_%7B-1%7D&bg=ffffff&fg=000&s=0)은 보통 0으로 초기화시킨다. (초기 은닉 층의 값을 0벡터로 초기화하여 첫 hidden state의 값을 계산할 수 있다.)

- ![o_t](http://s0.wp.com/latex.php?latex=o_t&bg=ffffff&fg=000&s=0)는 시간 스텝 ![t](http://s0.wp.com/latex.php?latex=t&bg=ffffff&fg=000&s=0)에서의 출력값이다. 예를 들어, 문장에서 다음 단어를 추측하고 싶다면 단어 수만큼의 차원의 확률 벡터가 될 것이다. ![o_t = softmax(Vs_t)](http://s0.wp.com/latex.php?latex=o_t+%3D+%5Cmathrm%7Bsoftmax%7D%28Vs_t%29&bg=ffffff&fg=000&s=0)

몇 가지 짚어두고 넘어갈 점이 있다.

- ![s_t](http://s0.wp.com/latex.php?latex=s_t&bg=ffffff&fg=000&s=0)는 과거의 시간 스텝들에서 일어난 일들에 대한 정보를 전부 담고 있고, 출력값 ![o_t](http://s0.wp.com/latex.php?latex=o_t&bg=ffffff&fg=000&s=0)는 오로지 현재 시간 스텝 ![t](http://s0.wp.com/latex.php?latex=t&bg=ffffff&fg=000&s=0)의 메모리에만 의존한다.
- 각 layer마다의 파라미터 값들이 전부 다 다른 기존의 deep한 신경망 구조와 달리, RNN은 모든 시간 스텝에 대해 파라미터 값을 전부 공유하고 있다 (위 그림의 U, V, W). 이는 RNN이 각 스텝마다 입력값만 다를 뿐 거의 똑같은 계산을 하고 있다는 것을 보여준다. 이는 학습해야 하는 파라미터 수를 많이 줄여준다.
- 위 다이어그램에서는 매 시간 스텝마다 출력값을 내지만, 문제에 따라 달라질 수도 있다. 예를 들어, 문장에서 긍정/부정적인 감정을 추측하고 싶다면 굳이 모든 단어 위치에 대해 추측값을 내지 않고 최종 추측값 하나만 내서 판단하는 것이 더 유용할 수도 있다. 마찬가지로, 입력값 역시 매 시간 스텝마다 꼭 다 필요한 것은 아니다. RNN에서의 핵심은 시퀀스 정보에 대해 어떠한 정보를 추출해 주는 hidden state이기 때문이다.

<img width="100%" alt="Character-level language model example" src="https://user-images.githubusercontent.com/53266682/126878508-eed5f7fc-14c7-43b1-8db1-7446fdfca716.png">


##### RNN Applications

- Language Modeling
- Speech Recognition
- Machine Translation
- Conversation modeling/Question Answering (Chatbot)
- Image/Video Captioning
- Image/Music/Dance Generation

![img](http://karpathy.github.io/assets/rnn/diags.jpeg)

_출처 http://karpathy.github.io/2015/05/21/rnn-effectiveness/_

- one to one

  - Vanilla Neural networks

    fixed-sized input → fixed-size input

    e.g. image classification

- one to many

  - Sequence output

    e.g. Image Captioning

    image → sequence of words

- many to one

  - Seqeunce input

    e.g. Sentiment Classification

    seqeunce of words → sentiment

- many to many

  - Sequence input and sequence output

    e.g. Machine Translation

    sequence of words → seqeunce of words

- many to many

  - Synced sequence input and sequence output

    e.g. Video classification on frame level

##### RNN의 학습 알고리즘

FFNN은 back-propagation 알고리즘을 통한 gradient descent 방법으로 모델 변수들을 학습한다. 하지만 back-propation 알고리즘은 모델에서 cycle이 존재하지 않다는 것을 가정하기 때문에 RNN에는 적용시킬 수 없다. RNN을 학습하는 방법 중 대표적인 방법은 `Back-Propagation Through Time (BPTT)` 알고리즘이다.

BPTT 알고리즘을 적용시킬 때 총 T 개의 time step 만큼을 고려한다면, 이는 T 개의 layer가 쌓여있는 구조를 학습시키는 것과 같다. 따라서 T가 커질수록 점점 더 깊은 구조를 학습시키는 것이기 때문에 RNN은 딥러닝의 일종이며, 딥러닝 구조를 학습할 때 발생하는 `vanishing gradient` 문제점들 역시 RNN을 학습할 때도 나타난다. 이런 문제점들을 해결하기 위해서, error propagation이 최대 r개의 time step까지만 이루어진다고 가정한 후 BPTT를 적용시키는 방법인 Truncated BPTT(TBPTT)가 있다.

RNN에서의 `vanishing gradient problem` 은 time step에 따라 hidden layer의 정보가 점점 사라지는 것으로 해석했다. 자세한 수식 및 설명은 이 [블로그](http://aikorea.org/blog/rnn-tutorial-3/)를 참고하자.

##### Vanilla RNN의 한계

위에서 설명한 RNN을 가장 단순한 형태의 RNN이라고 하여 바닐라 RNN(Vanilla RNN)이라고 한다. (케라스에서는 SimpleRNN) 

바닐라 RNN은 출력 결과가 이전의 계산 결과에 의존하는데, 바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있다. 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생한다. 이를 `장기 의존성 문제(the problem of Long-Term Dependencies)`라고 한다. 

바닐라 RNN 이후 바닐라 RNN의 한계를 극복하기 위한 다양한 RNN의 변형이 나왔다.

#### 장단기 메모리(Long Short-Term Memory, LSTM) 

_출처 http://colah.github.io/posts/2015-08-Understanding-LSTMs/_

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

` The repeating module in a standard (vanilla) RNN contains a single layer.`

![A LSTM neural network.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

`The repeating module in an LSTM contains four interacting layers.`

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)

위의 그림은 LSTM의 전체적인 내부의 모습을 보여준다. 전통적인 RNN의 단점을 보완한 RNN의 일종을 장단기 메모리(Long Short-Term Memory)라고 하며, 줄여서 LSTM이라고 한다. LSTM은 은닉층의 메모리 셀에 `입력 게이트, 망각 게이트, 출력 게이트`를 추가하여 `불필요한 기억을 지우고`, `기억해야할 것들을 정한다`. 

##### LSTM의 핵심 아이디어

LSTM의 핵심은 `cell state` 이다. 다이어그램 상단에 있는 수평선으로, 하나의 컨베이어 벨트와 같다. 이 구조는 전체 체인을 관통하여, 정보가 변하지 않고 쉽게 흘러갈 수 있다. 

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

LSTM은 신중하게 정제된 구조를 가진 `gate` 들을 이용해 cell state에 정보를 더하거나 없앨 수 있다. 게이트들은 sigmoid neural net layer와 점단위 곱하기 연산으로 이루어져있다.

<img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" alt="img" style="zoom:30%;" />

sigmoid layer는 0부터 1까지의 출력값을 가지며, 각 요소를 얼마나 통과시킬지 나타낸다. 0은 "아무것도 통과시키지 않음"을 의미하고, 1은 "모든 것을 통과시킴!"을 의미한다. (sigmoid layer를 통해 정보를 몇 % 통과시킬지 정한다.)

##### 첫 번째 단계: cell state에서 과거의 정보를 얼마나 지울지 정하기

`forget gate layer` 라고 불리는 sigmoid layer로 만들어진다.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

##### 두 번째 단계: cell state에서 새로운 정보를 얼마나 저장할지 정하기

1. `input gate layer` 라고 불리는 sigmoid layer로 어떤 값을 update 할지 정하고, `tanh layer`는 cell state에 더해질 수 있는 새로운 후보 값들의 벡터인 C̃t를 만든다. 이 두 값을 결합하여 cell state를 update한다.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

2. 이제 과거의 cell state(Ct-1)를 새로운 cell state인 (Ct)로 업데이트 해야한다. 이전 단계들에서 구해놓은 값을 이용하면 된다. 우선 잊어버리기로 한 데이터를 잊어버리기 위해 이전 state애 ft를 곱한다. 그리고 그 값에 각 state 값을 업데이트하기로 결정한 정도에 따라 후보값들을 스케일한 값인, it∗C̃t를 더한다. 

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)



##### 마지막 단계: 어떤 것을 출력할지 정하기

1. cell state의 어떤 부분을 출력할지 정하는 sigmoid layer를 실행한다.
2. cell state를 tanh에 넣어 -1과 1사이의 값으로 만든 후 `output of the sigmoid gate` 에 곱하여 원하는 부분만 출력한다.

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)



요약하면 LSTM은 은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며 셀 상태(cell state)라는 값을 추가하였다. 

![img](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png)

LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보인다.

#### 게이트 순환 유닛(Gated Recurrent Unit, GRU)

GRU(Gated Recurrent Unit)는 2014년 뉴욕대학교 조경현 교수님이 집필한 논문인 [Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)에서 제안되었다. GRU는 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였다. 다시 말해서, `GRU는 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 간단화` 시켰다.

![A gated recurrent unit neural network.](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했다. 반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재한다. GRU는 LSTM보다 학습 속도가 빠르다고 알려져있지만 여러 평가에서 GRU는 LSTM과 비슷한 성능을 보인다고 알려져있다.

#### Seqeunce-to-Seqeunce, seq2seq

![img](https://wikidocs.net/images/page/24996/seq2seq%EB%AA%A8%EB%8D%B811.PNG)

_출처 https://wikidocs.net/24996_

RNN을 이용한 인코더와 디코더로 이루어져있다. (물론, 성능 문제로 인해 실제로는 바닐라 RNN이 아니라 **LSTM 셀** 또는 **GRU 셀**들로 구성된다.) `인코더`는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데, 이를 `컨텍스트 벡터(context vector)`라고 한다. 입력 문장의 정보가 하나의 컨텍스트 벡터로 모두 압축되면 인코더는 컨텍스트 벡터를 디코더로 전송한다. `디코더`는 컨텍스트 벡터를 받아서 번역된 단어를 한 개씩 순차적으로 출력한다.

시퀀스-투-시퀀스(Sequence-to-Sequence)는 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용되는 모델이다. 예를 들어 `챗봇(Chatbot)`과 `기계 번역(Machine Translation)`이 그러한 대표적인 예인데, 입력 시퀀스와 출력 시퀀스를 각각 질문과 대답으로 구성하면 챗봇으로 만들 수 있고, 입력 시퀀스와 출력 시퀀스를 각각 입력 문장과 번역 문장으로 만들면 번역기로 만들 수 있다. 그 외에도 `내용 요약(Text Summarization)`, `STT(Speech to Text)` 등에서 쓰일 수 있다.

#### Attention Mechanism

RNN에 기반한 seq2seq 모델에는 크게 두 가지 문제가 있다.

1. 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생
2. RNN의 고질적인 문제인 기울기 소실(Vanishing Gradient) 문제가 존재
   (LSTM, GRU 한계)

결국 이는 기계 번역 분야에서 입력 문장이 길면 번역 품질이 떨어지는 현상으로 나타났다. 이를 위한 대안으로 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위해 어텐션(attention) 기법이 등장했다.

어텐션 매커니즘은 신경망들의 성능을 높이기 위한 메커니즘이자, 이제는 AI 분야에서 대세 모듈로서 사용되고 있는 트랜스포머의 기반이 된다.

#### Transformer

트랜스포머(Transformer)는 2017년 구글이 발표한 논문인 "[Attention is all you need](https://arxiv.org/abs/1706.03762)"에서 나온 `모델`로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 `어텐션(Attention)으로만`으로 구현한 모델이다. 이 모델은 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 성능도 RNN보다 우수하다는 특징을 갖고 있다. seq2seq처럼 인코더-디코더 구조를 유지하고 있는데, 다른 점은 인코더와 디코더라는 단위가 N개가 존재할 수 있다는 점이다.

<img width="311" alt="Transformer" src="https://user-images.githubusercontent.com/53266682/126884646-ed969f99-4460-4094-9a98-de16de465254.png">


## 단어의 표현 방법

자연어를 컴퓨터가 이해하고, 효율적으로 처리하게 하기 위해서는 컴퓨터가 이해할 수 있도록 자연어를 적절히 변환할 필요가 있다. 단어를 표현하는 방법에 따라서 자연어 처리의 성능이 크게 달라지기 때문에 이에 대한 많은 연구가 있었고, 여러가지 방법들이 알려져 있다.

![img](https://wikidocs.net/images/page/31767/wordrepresentation.PNG)

_출처 https://wikidocs.net/31767_

단어의 표현 방법은 크게 국소 표현(Local Representation) 방법과 분산 표현(Distributed Representation) 방법으로 나뉜다. `국소 표현 방법`은 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법이며, `분산 표현 방법`은 그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법이다.

_또한 비슷한 의미로 국소 표현 방법(Local Representation)을 이산 표현(Discrete Representation)이라고도 하며, 분산 표현(Distributed Representation)을 연속 표현(Continuous Representation)이라고도 한다._

### Local Representation

#### 원-핫 인코딩

원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다. 이렇게 표현된 벡터를 원-핫 벡터(One-Hot vector)라고 한다.

![img](https://blog.kakaocdn.net/dn/bQAqI1/btqYn4RUV8U/3gWF8KPkz2w0IsMzG6yduk/img.png)

_출처 https://deep-eye.tistory.com/67_

#### 카운트 기반의 단어 표현 (Count based word Representation)

머신러닝 등의 알고리즘이 적용된 본격적인 자연어 처리를 위해서는 **문자를 숫자로 수치화**할 필요가 있다.

##### Bag of Words(BoW)

단어의 빈도수를 카운트(Count)하여 단어를 수치화하는 단어 표현 방법이다. 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중한다.

##### 문서 단어 행렬(Document-Term Matrix, DTM)

BoW의 확장인 DTM(또는 TDM)은 각 문서에 대한 BoW 표현 방법을 그대로 갖고와서, 서로 다른 문서들의 BoW들을 결합한 표현 방법

##### TF-IDF(Term Frequency-Inverse Document Frequency)

TF-IDF는 빈도수 기반 단어 표현에 단어의 중요도에 따른 가중치를 줄 수 있다.

단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법이다. 사용 방법은 우선 DTM을 만든 후, TF-IDF 가중치를 부여한다.

### Continuous Representation

#### 워드 임베딩 (Word Embedding)

워드 임베딩(Word Embedding)은 단어를 벡터로 표현하는 방법으로, 단어를 `밀집 표현(dense representation)`으로 변환한다. 

앞서 원-핫 인코딩을 통해서 나온 원-핫 벡터들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 벡터 표현 방법이었다. 이렇게 벡터 또는 행렬(matrix)의 값이 대부분이 0으로 표현되는 방법을 `희소 표현(sparse representation)`이라고 한다. 그러니까 원-핫 벡터는 희소 벡터(sparse vector)다.

원-핫 벡터의 문제점

1. 공간적 낭비를 불러일으킨다.

   원-핫 벡터뿐 아니라 희소 표현의 일종인 DTM과 같은 경우에도 특정 문서에 여러 단어가 다수 등장하였으나, 다른 많은 문서에서는 해당 특정 문서에 등장했던 단어들이 전부 등장하지 않는다면 역시나 행렬의 많은 값이 0이 되면서 공간적 낭비를 일으킨다.

2. 단어의 의미를 담지 못한다.

   local representation : 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법

이러한 희소 표현과 반대되는 표현이 있으니, 이를 밀집 표현(dense representation)이라고 한다. 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않는다. 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춘다. 또한, 이 과정에서 더 이상 0과 1만 가진 값이 아니라 실수값을 가지게 된다.

단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 `워드 임베딩(word embedding)`이라고 한다. 그리고 이 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과라고 하여 `임베딩 벡터(embedding vector)`라고도 한다.

워드 임베딩 방법론으로는 LSA, Word2Vec, FastText, GloVe 등이 있다.

##### 잠재 의미 분석(Latent Semantic Analysis, LSA)

BoW에 기반한 DTM이나 TF-IDF는 기본적으로 단어의 빈도 수를 이용한 수치화 방법이기 때문에 단어의 의미를 고려하지 못한다는 단점이 있었다. (이를 토픽 모델링 관점에서는 단어의 토픽을 고려하지 못한다고도 한다.) 이를 위한 대안으로 DTM의 잠재된(Latent) 의미를 이끌어내는 잠재 의미 분석(Latent Semantic Analysis, LSA)이라는 방법이 있다. 잠재 의미 분석(Latent Semantic Indexing, LSI)이라고 부르기도 한다. 

##### Word2Vec

원-핫 벡터는 단어 간 유사도를 계산할 수 없다는 단점이 있어서 `단어 간 유사도를 반영`할 수 있도록 단어의 의미를 `벡터화` 할 수 있는 방법이 필요하다. 이를 위한 학습 방법으로는 NNLM, RNNLM 등이 있으나 요즘에는 해당 방법들의 속도를 대폭 개선시킨 Word2Vec가 많이 쓰이고 있다.

Word2Vec은 예측(prediction)을 기반으로 단어의 뉘앙스를 표현한다.

Word2Vec에는 `CBOW(Continuous Bag of Words)`와 `Skip-Gram` 두 가지 방식이 있다. CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법이다. 반대로, Skip-Gram은 중간에 있는 단어로 주변 단어들을 예측하는 방법이다.

##### FastText

Word2Vec의 확장이라고 볼 수 있다. Word2Vec와 FastText와의 가장 큰 차이점은 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주한다. 즉 내부 단어(subword)를 고려하여 학습한다.

##### GloVe

글로브(Global Vectors for Word Representation, GloVe)는 카운트 기반과 예측 기반을 모두 사용하는 방법론으로 2014년에 미국 스탠포드대학에서 개발한 단어 임베딩 방법론이다. 기존의 카운트 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 단점을 지적하며 이를 보완한다는 목적으로 나왔고, 실제로도 Word2Vec만큼 뛰어난 성능을 보여줍니다. 

LSA는 카운트 기반으로 코퍼스의 전체적인 통계 정보를 고려하기는 하지만, 왕:남자 = 여왕:? (정답은 여자)와 같은 단어 의미의 유추 작업(Analogy task)에는 성능이 떨어진다. Word2Vec는 예측 기반으로 단어 간 유추 작업에는 LSA보다 뛰어나지만, 임베딩 벡터가 윈도우 크기 내에서만 주변 단어를 고려하기 때문에 코퍼스의 전체적인 통계 정보를 반영하지 못한다. GloVe는 이러한 기존 방법론들의 각각의 한계를 지적하며, LSA의 메커니즘이었던 카운트 기반의 방법과 Word2Vec의 메커니즘이었던 예측 기반의 방법론 두 가지를 모두 사용한다.

## 임베딩 기법

워드 임베딩의 역사는 인공 망을 이용하여 주변 단어의 단어 등장 확률을 예측한 Neural Probabilistic Language Model(NPLM)이 발표된 이후부터 Word2Vec→FastText→ ELMO→BERT 기법으로 발전하고 있다. 

가장 최신의 언어분석 기법인 BERT는 다른 언어분석 기법들에 비해 임베딩 결과에서 우수한 성능을 보이고 있다. 이는 기존의 임베딩은 문장에서 단어를 순차적으로 입력받고 다음 단어를 예측하는 일방향(uni-directional)이 지만 BERT는 문장 전체를 입력받고 단어를 예측하고 양방향(bi- directional) 학습이 가능하기 때문이다.

### 단어 수준의 임베딩 기법

단어 수준의 임베딩은 신경망을 이용하여 텍스트를 변환하는 것이 가장 큰 특징으로 단어가 주어지면 그 단어와 주변 단어가 동시에 일어날 확률을 구하므로 단어의 의미를 수치화할 수 있다. 단어 수준의 벡터 표현은 텍스트를 수치화한 벡터 형태로 표현하는 것이다. 단어수준의 임베딩 기법에는 Word2Vec, GloVe, FastText 등이 있다.

### 문장 수준의 임베딩 기법 (Contextualized Word Representation)

문장 수준의 임베딩은 2018년 초에 ELMo(Embedding from Language Models)가 발표된 이후 주목받기 시작했다. 이는 개별 단어가 아닌 단어 Sequence 전체의 문맥적 의미를 함축하기 때문에 단어 임베딩 기법보다 Transfer Learning 효과가 좋은 것으로 알려져 있다. 또한, 단어 수준 임베딩의 단점인 동음이의어도 문장수준 임베딩 기법을 사용하면 분리해서 이해할 수 있다. 문장 수준의 임베딩 기법에는 BERT, GPT 등이 있다.

#### Embeddings from Language Model, ELMo

ELMO(Embeddings from Language Model)는 2018년에 제안된 새로운 워드 임베 딩 방법론으로 “언어 모델로 하는 임베딩”이라 해석된다. ELMO의 특징은 **사전 훈련된 언어 모델(Pre-trained Language Model)**을 사용한다는 점이다. Bank라는 단어를 생각해보자. Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 전혀 다른 의미를 가지는데, Word2Vec이나 GloVe 등으로 표현된 임베딩 벡터들은 두 가지 상황 모두에서 같은 벡터를 사용하여, 문맥을  제대로 반영하지 못한다는 단점이 있다. 이러한 한계점을 ELMO는 BiLM의 사전훈련으로 극복할 수 있다. 또한, 이 특징은 NLP에서 Transfer Learning이 확산되는 계기가 되어 지금의 BERT가 출현하게 되었다. 

#### BERT (Bidirectional Encoder Representations from Transformer)

이 모델은 최근까지 딥러닝 모델을 적용한 모든 자연어 처리 분야에서 좋은 성능을 보이고 있는 범용 `언어 모델`이다. BERT는 사전학습(pre-trained) 모델로서, 특정 과제(task)를 하기 전 사전훈련 임베딩을 실시하 므로 기존의 임베딩 기술보다 과제의 성능을 더욱 향상시킬 수 있는 모델로 관심받고 있다. BERT를 적용한 모델링 과정을 살펴보면 Pre-trained는 비지도 학습(Unsupervised Learning) 방식으로 진행되고 대량의 코퍼스를 Encoder가 임베딩하고, 이를 transfer하여 Fine-tuning을 통해 목적에 맞는 학습을 수행하여 과업을 수행하는 것이 특징이다. 또 다른 BERT의 특징은 양방향 모델을 적용하여 문장의 앞과 뒤의 문맥을 고려하는 것으로 이전보다 더 높은 정확도를 나타낸다. BERT의 활용은 대량의 텍스트 데이터와 다양한 언어를 적용할 수 있다는 장점 때문에, 연구자들 사이에서 가장 각광 받는 기술 중 하나이다.

#### GPT-3 (Generative Pre-trained Transformer 3)

GPT 3의 기반은 트랜스포머(transformers)라 불리는 딥러닝 체계다. 트랜스포머에 대한 개념은 2017년 구글 브레인(Google Brain)이 발간한 보고서 ‘필요한 것은 집중(Attention is all you need)’에서 처음 소개됐다. 트랜스포머는 방대한 크기의 데이터 세트를 학습할 수 있고 효율적으로 비교 가능한 다양한 모델의 밑거름이 됐다. 구글 보고서가 발간된 이후 다양한 언어 작업을 처리할 수 있는 슈퍼 모델을 구축하기 위한 경쟁이 시작된 걸 보면 잘 알 수 있다. 구글의 버트(BERT), 마이크로소프트의 [튜링NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)와 Open AI의 GPT 3 모두 트랜스포머를 기반으로 하는 최신 `모델`이다. GPT3가 나오기 전 가장 큰 언어 모델은 2020년 마이크로소프트가 선보인 튜링 NLG였다. 170억 개의 매개 변수 언어 모델인데, GPT3보다 10배나 규모가 작다. GPT3가 나오자 마이크로소프트는 경쟁을 포기하고 독점적 사용권을 얻었을 정도로 GPT3는 막강한 글쓰기 실력을 자랑한다.



## 참고자료

[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) ⭐️

[모두를 위한 딥러닝 시즌 2 - Tensorflow](https://www.boostcourse.org/ai212/lecture/43749?isDesc=false) ⭐️

http://karpathy.github.io/2015/05/21/rnn-effectiveness/ ⭐️

http://colah.github.io/posts/2015-08-Understanding-LSTMs/ ⭐️

http://aikorea.org/blog/rnn-tutorial-1/ ⭐️

https://www.cs.ubc.ca/labs/lci/mlrg/slides/rnn.pdf

https://www.koreascience.or.kr/article/JAKO201510350830298.pdf

[인공지능과 자연어 처리 기술 동향](https://www.itfind.or.kr/publication/regular/weeklytrend/weekly/view.do?boardParam1=8085&boardParam2=8085)

[GPT 모델의 발전 과정 그리고 한계](https://medium.com/ai-networkkr/gpt-%EB%AA%A8%EB%8D%B8%EC%9D%98-%EB%B0%9C%EC%A0%84-%EA%B3%BC%EC%A0%95-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%ED%95%9C%EA%B3%84-81cea353200c)



## 추가로 보면 좋을 자료

[인공 신경망 언어 모델 종류](https://medium.com/@datamonsters/artificial-neural-networks-for-natural-language-processing-part-1-64ca9ebfa3b2)

[NLP의 진화](https://www.hostcomm.co.uk/blog/2019/evolution-natural-language-infographic)

[LSTM 쉽게 이해하기](https://www.youtube.com/watch?v=bX6GLbpw-A4&t=309s)

