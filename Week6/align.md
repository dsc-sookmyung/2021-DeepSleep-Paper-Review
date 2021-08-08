# Neural Machine Translation by Jointly Learning to Align and Translate
- 링크: https://arxiv.org/pdf/1409.0473v7.pdf
- 주제: **Alignment model** 의 등장과 입력 문장 벡터의 연관 순위 책정에 따른 번역 효율 향상
- 배경: 기존처럼 입력 문장을 **고정된 길이의 context vector** 로 변환할 시, **길이가 긴 입력 문장에 대해서는 번역 성능이 급격히 저하**되는 문제점이 발생
- 내용
	1. decoder 에서 output 을 출력할 때, 입력 문장을 순차적으로 탐색해서 현재 생성하려는 decoder의 output 과 가장 관련있는 영역을 적용시킴
	2. 따라서 고정된 길이의 context vector 를 사용하지 않고, encoder 에서 생성한 여러 context vector 를 계속해서 참조하므로 문장의 길이가 길어도 성능 유지 가능


## RNN을 이용한 encoder-decoder 모델
- **Encoder**	
	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdSTFvD%2FbtqLIGDkLDC%2Fk6grHXK3TzZpkJiSQk7mBk%2Fimg.png)  
	`=> time t 에서의 hidden state `

	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbDo2qZ%2FbtqLNrrUVqn%2FoYHCbG5XfqMBxsyJFEV1hK%2Fimg.png)

- **Decoder**
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUT3Nv%2FbtqLSOsADAV%2FzxNjXC0Z9pImVudtYASQB1%2Fimg.png)
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbUxSEj%2FbtqLNsK7Kdh%2FIkG2J7O4V3kOgzsKEbexc0%2Fimg.png)
`=> 고정된 길이의 vector에 Input Sentence를 모두 담아내기는 어렵기에 Input Sentence가 길어질수록 Output Sentence 에 대한 성능을 기대하기 어렵다고 판단`

## Align and translate
![img](https://heiwais25.github.io/img/nlp/arxiv1409_img1.jpg)
- **Encoder**
	- 입력문장: **x=(x1,...,xTx)**
	- **bidirectional RNN(BiRNN)** = **forward RNN + backward RNN**
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAgvdh%2FbtqLRkZQzcZ%2FlQ7dRACU751ktqpB0KXm4K%2Fimg.png)
`=> time j 에서의 입력 xj 에 대해서 forward hidden state hj와 backward hidden state hj 를 연결하여 생성`
`=> hidden state hj 는 입력 단어 xj 의 가까운 위치에 있는 단어들의 정보를 더 많이 보유하게 됨`

- **Decoder**
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3x4Cz%2FbtqLLGbLB89%2Fr1n6AaPAg3RraLX1nrdC1k%2Fimg.png)
**a: alignment model(feedforward neural network)**
**si: time i 에서 decoder의 hidden state**
**hj: time j에서 encoder의 hidden state**
`=> decoder 의 time i 에서의 정보가 encoder의 time j 에서의 정보와 얼마나 연관성이 있는지 확인(decoder 의 바로 전 time의 hidden state와 encoder 의 time j 에서의 hidden state를 인자값으로 사용)`

	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbvBpvW%2FbtqLNsYBuXq%2FpdP5uyYL95S7jSlrKsvivK%2Fimg.png)
	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdvFLJC%2FbtqLOPlKAag%2FxKJknCLw03iKaxCKDsMCnk%2Fimg.png)
	**time i 에서의 context vector**: encoder 의 모든 hidden state의 weighted sum
	**weight αij**:  target word yi 가 source word xj 와 얼마나 연관이 있는지 나타냄. 따라서 decoder는 해당 weight 값을 기반으로 source sentence에서 어떤 위치의 단어에 더 attention을 줄 지 판단 가능
	
	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4CRFz%2FbtqLLF44ruR%2FfzcZZD3Xv9GaucfxVWMOzK%2Fimg.png)
`=> 최종적으로 구해지는 time i 에서의 decoder hidden state`


## Experiments
- 데이터셋: WMT'14 English-French parallel corpus 
- 모델: 두 모델을 각각 두 번씩 학습
	- RNN Encoder-Decoder
	- RNN Search
	=> 첫 학습: 최대 30개 단어들로 구성된 문장들로 학습
	=> 두번째 학습: 50개의 단어로 구성된 문장들로 학습
	 - RNNencdec 1000개의 hidden unit / RNNsearch encoder(forward/backward) 1000개의 hidden unit, decoder 1000개의 hidden unit
	 - minibatch SGD 사용, minibatch 80개 문장으로 5일간 학습
	![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUDOFU%2FbtqLRU859S8%2FgG3Ehzq2tQ0gshvYSfMme0%2Fimg.png)

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc2qsxc%2FbtqLRVtjaJO%2FvTnECjV3z6cpSqXc4YFJP0%2Fimg.png)

`=> RNNsearch가 기존 모델보다 좋은 성과를 보여주었으며, RNNsearch-50은 문장이 길어져도 성능 저하가 일어나지 않음`


## 참고
https://misconstructed.tistory.com/49