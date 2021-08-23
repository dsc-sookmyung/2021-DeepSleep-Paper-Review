# Improving Language Understanding by Generative Pre-Training

### Generative Pre-training of language model, GPT

**`Language Model`**

언어 모델(Language Model, LM)은 **단어 시퀀스에 확률을 할당**하는 일을 하는 모델로, **이전 단어들이 주어졌을 때 다음 단어를 예측**하도록 한다.

<img src="https://thegradient.pub/content/images/2019/10/lm-1.png" alt="img" width="70%" />

_출처 https://thegradient.pub/understanding-evaluation-metrics-for-language-models/_

**`Generative Model vs Discriminative Model`**

<img src="https://datawarrior.files.wordpress.com/2016/05/discriminative_vs_generative.png" alt="img" width="40%" />

_출처 https://datawarrior.wordpress.com/2016/05/08/generative-discriminative-pairs/_

- Discriminative model: 레이블 정보가 있어야 하기 때문에 지도학습(supervised learning) 범주에 속하며, 주어진 데이터의 레이블을 잘 구분하는 결정 경계(decision boundary)를 학습하는 것이 목표다.
- Generative model: 레이블 정보가 있어도 되고, 없어도 되며 범주의 분포(distribution)를 학습하는 것이 목표다.

_generative 모델과 discriminative 모델의 차이점은 [여기](https://ratsgo.github.io/generative%20model/2017/12/17/compare/)에서 자세히 알아볼 수 있다._

## 배경

대부분의 딥러닝 학습은 수동으로 라벨링된 데이터가 필요한데, `labeled data`가 부족하기 때문에, discriminatively trained models가 잘 작동하기 어렵다. 

따라서 `unlabeled data`를 이용하는 것이 시간 소모나 비용을 줄이는데 좋은 대안이 되었다. 

지난 몇 년동안, 연구자들은 `pre-trained word embedding`을 사용하는 것이 가장 강력하다고 증명해왔는데, 이는 주로 단어 수준의 정보를 transfer했다.

하지만 우리는 고수준의 의미(higher-level semantics)를 포착해야했는데, 라벨링되지 않은 텍스트에서 단어 수준을 넘어서는 정보를 뽑아내는 것은 크게 두 가지 이유 때문에 어렵다.

1.  어떤 최적화를 위한 목적 함수(optimization objectives)가 전이 학습에 유용한 text representation을 학습하기에 가장 효과적인지 불명확하다.
2. 학습된 representation을 target task에 전이하는 가장 효과적인 방법에 대한 합의가 없다.

이 불확실성은 언어 처리를 위한 효과적인 semi-supervised 학습 방법을 발전시키는 데 어려움을 주었다.

⇒ 이 논문에서는 unsupervised pre-training과 supervised fine-tuning의 조합을 이용한 semi-supervised approach를 연구했다.

**`목표`**   `약간의 fine-tuning만으로 다양한 task에 잘 전이하는, 범용적인(universal) representations 학습하기`

## Framework

### 2단계 학습 과정 (Two-stages training procedure)

**1단계**

- Generative pre-training language model

- Unlabeled large corpus로 학습

**2단계**  

- Discriminative Fine-tuning Language Model
- Labeled data 활용하여 specific task에 학습

### Model architecture

- 기존 Transformer의 decoder를 12개 쌓은 구조 (decoder에서 Multi-Head Attention 제외)

  → Transformer는 텍스트의 long-term dependencies를 처리하기 위한 보다 구조화된 메모리를 제공한다.

  → Masked Multi-Head Attention에서 `Masking`은 언어 모델이 현재 단어의 오른쪽에 있는 후속 단어에 접근할 수 없게 하는 언어 모델 목표를 달성하는데 도움을 준다.

  <img src="https://i.imgur.com/Q7IS78n.png" alt="img" width="40%" />
  
  _출처 https://ratsgo.github.io/nlpbook/docs/language_model/bert_gpt/_

- transfer를 하는 동안, 구조화된 텍스트 입력을 단일 연속 토큰 시퀀스로 처리하는 traversal-style 접근 방식에서 파생된 `task-specific input transformation`을 활용한다.

  → 사전 학습된 모델의 아키텍처를 최소한으로 변경하여 효과적으로 fine-tune을 할 수 있다.

### Step 1. Unsupervised pre-training

- Pre-training of  `Language Model`

  Unsupervised Learning with unlabeled text

  - use a standard language modeling objective (기존의 LM과 공식 같음)

    ![img](https://miro.medium.com/max/1168/1*Zrg8WFl_Zc7FDtSVgLC9eg.png)
  
    - Unsupervised corpus of token 𝑇에 대해 𝑘(window size)개 토큰이 주어졌을 때, 다음 토큰 예측
    - 다음 토큰이 등장할 likelihood L1(𝑇)를 최대화 하도록 학습
  
  - use a multi-layer _Transformer decoder_ for the language model

### Step 2. Supervised fine-tuning

- Fine tuning on each specific task

  Supervised Learning

  - 주어진 토큰(x1, ..., xn)을 이용하여 label y를 예측할 가능성을 최대화

    ![img](https://miro.medium.com/max/1044/1*5hDwpxGf2KGPlNOvmcBX6g.png)

  - equation (ii)를 바로 최대화하지 않고, 보조 목적 함수(auxiliary objective) 사용

    ![img](https://miro.medium.com/max/1088/1*pFWB54O7V8HtWu97H0wUIw.png)

    - supervised model의 일반화(generalization) 더 잘 되게 함
    - 수렴(convergence) 가속화

  - 기존 모델들은 fine-tuning 시에 layer를 추가해야했고, 적지 않은 시간과 비용이 소모된다.

    But, GPT-1은 **layer 추가 작업 없이** language model 학습 시에 사용한 Transformer decoder 모델을 그대로 fine-tuning에도 유지한다.

  - Fine-tuning 과정 동안 레이블링을 통해 모델이 특정 task에 최적화되게 한다.

![GPT1구조](https://user-images.githubusercontent.com/53266682/130374807-1c4457f4-61d6-45f7-bb75-f423109f4bac.png)

### Task-specific input transformation

Classification 같은 일부 task는 바로 fine-tune이 가능하다.

다른 특정 task는 구조화된 입력을 갖고 있다. 논문의 pre-trained 모델이 텍스트의 연속 시퀀스로 학습했기 때문에, 이러한 task들을 이 모델에 적용하기 위해서는 수정이 필요하다. 이 논문에서는 논문의 pre-trained 모델이 처리할 수 있게, 구조화된 입력을 정렬된 시퀀스로 변환해주는 traversal-style approach를 사용했다. 이 입력 변환(input transformation)을 통해 작업 전반에 걸쳐 아키텍처를 최소한으로 변경하여 효과적으로 fine-tune을 할 수 있다.

- 무작위로 초기화된 start와 end tokens를 입력 시퀀스에 추가한다.

- 두 개의 문장 사이에 special character(delimeter)를 집어넣고 하나의 문장으로 묶어서 모델의 입력 layer에 넣는다.


## Experiments

### 1. Unsupervised Training

**`Dataset`**

- BooksCorpus dataset을 이용하여 언어 모델 학습
  - 본 적 없는 데이터를 학습할 수 있는데 도움이 된 약 7000권의 미출간 책으로 구성
  - 말뭉치(corpus)에 연속적인 텍스트가 많이 포함되어 있어 모델이 장거리 종속성(long range dependencies)을 학습하는데 도움

**`BPE: Byte Pair Encoding`**

- 기존 딥러닝 모델의 embedding 방법보다 진화된 방법 사용

  - **기존**: word embedding 또는 character embedding

    - word embedding: 신조어, 오탈자에 약함

      → `OOV 문제` OOV(Out-Of-Vocabulary) 또는 UNK(Unknown Token)

    - character embedding: 단어 간 유사도가 word embedding에 비해 떨어짐

  - **GPT-1**: Byte Pair Encoding

    - Byte Pair Encoding

      대표적인 서브워드 분리(Subword segmentation) 알고리즘

      : 하나의 단어는 더 작은 단위의 의미있는 여러 서브워드들(Ex) birthplace = birth + place)의 조합으로 구성된 경우가 많기 때문에, 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩하겠다는 의도를 가진 전처리 작업

      → `OOV`나 희귀 단어, 신조어와 같은 문제 `완화`

      → 입력값의 의미 더 잘 전달

### 2. Supervised Fine-tuning

Unsupervised pre-training에서 사용한 하이퍼파라미터를 재사용했다.

**`Four types of language understanding tasks`**

- **Natural Language Inference (NLI)**

  - 텍스트 함의 인식(recognizing textual entailment)으로도 알려져있으며, 두 문장 간의 관계를 맞추는 task

  - Label은 3가지 종류 존재: Contradiction / Neutral / Entailment

    _datasets: SNLI, transcribed speech, popular fiction, MNLI, QNLI, SciTail, RTE_

  - Examples

    <img width="681" alt="img" src="https://user-images.githubusercontent.com/53266682/130374277-1feba1ca-e9d1-4a4f-b8a7-b083c76c161c.png">
    
    _출처 https://github.com/kakaobrain/KorNLUDatasets_

- **Question answering and commonsense reasoning**

  - 지문과 이에 관련된 질문이 주어졌을 때, 알맞은 답을 찾아내는 task

    _dataset: RACE_

  - Examples

    <img src="https://rajpurkar.github.io/mlx/qa-and-squad/example-squad.png" alt="img" width="70%" />
    
    _출처 https://rajpurkar.github.io/mlx/qa-and-squad/_

- **Semantic Similarity**

  - 주어진 두 문장 간의 유사한 정도를 점수로 계산

    _datasets: MRPC, QQP, STS-B_

  - Examples

    <img src="https://2.bp.blogspot.com/-9Qk1fubLpzg/Wv2QGgKVVmI/AAAAAAAACvs/Gm-XF3prXVIIvaIkrTmkcIcYz-4qSxLKwCLcBGAs/s1600/image2.png" alt="img" width="50%" />

    _출처 https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html_

- **Classification**

  - CoLA dataset: 문장이 문법적으로 맞았는지 틀렸는지를 분류하는 task
  - SST-2 dataset: 표준 이진 분류 task로, 문장의 sentiment를 분류

  - Examples

    <img src="https://paperswithcode.com/media/datasets/sst.jpg" alt="img" width="50%" />

    _출처 https://paperswithcode.com/dataset/sst_

**⇒ 12개 부문 중 9개 부문에서 SOTA를 달성했다.**

## Analysis

**`Impact of number of layers transferred`**

unsupervised pre-training에서 supervised target task로 다양한 수의 레이어를 전이(transfer)했을 때의 영향을 관찰한 결과

- Layer의 개수가 증가함에 따라 정확도가 향상되었다.
- Layer #12 이후부터는 수렴 양상을 보였다.  _(Cf. 모델 아키텍처: Transformer의 decoder를 12개 쌓은 구조)_

![GPT1-Figure2](https://user-images.githubusercontent.com/53266682/130374819-ef287ae8-495a-4240-910e-5ebcabd71468.png)

**`Zero-shot Behaviors of the pre-trained model`** 

Transformer로 사전 학습된 언어 모델은 downstream tasks에 유용한 언어 지식을 효과적으로 얻는다.

- LSTM이 zero-shot 성능에서 더 큰 variance를 보이는 것을 관찰하였고, Transformer 아키텍처가 LSTM보다 전이(transfer)를 더 효율적으로 한다는 것을 시사한다.

_Cf. Transfer learning과 downstream task_

_전이 학습(Transfer Learning)이란 특정 태스크를 학습한 모델을 다른 태스크 수행에 재사용하는 기법을 가리킨다. 모델이 새로운 태스크(`Task2`)를 배울 때, 이전에 태스크(`Task1`)를 수행해봤던 경험을 재사용한다고 했을 때, `Task1`은 **upstream task**, `Task2`는 **downstream task**라고 한다._

_Upstream task를 학습하는 과정을 사전 학습(**pretrain**)이라고 하고, downstream task를 학습하는 과정은 방식에 따라 여러 가지 용어로 불린다. **Fine tuning, zero-shot learning, one-shot learning, few-shot learning** 등이 있다. 이 글의 마지막 부분에 추가적인 설명을 달아놓았다._

_위 설명은 [ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/docs/introduction/transfer/)을 참고하였으며, Transfer learning에 대한 더 자세한 설명이 보고 싶다면 이 사이트에서 살펴보자_.

**`Alblation studies`**

_Cf. Ablation study란? 전체 시스템에 대한 구성 요소의 기여도를 이해하기 위해 특정 구성 요소를 제거하여 AI 시스템의 성능을 연구하는 것이다._

![GPT1-Table5-1](https://user-images.githubusercontent.com/53266682/130374825-e98fa781-0523-42a5-8438-e06ea09ae366.png)

_Classification tasks: CoLA, SST2  | Semantic Similarity tasks: MRPC, STSB, QQP | NLI tasks: NMLI, QNLI, RTE_

1. Pre-training 없이 directly trained on supervised target tasks 경우 성능 비교
   - 모든 task에 대해 pre-training이 없으면 성능이 저하된다. (14.8%)
2. Fine-tuning 시에 보조 목적 함수(auxiliary LM objective)를 사용하지 않았을 경우 성능 검사
   - auxiliary objective을 이용하는 것이 큰 데이터셋에는 효과적이지만, 작은 데이터셋에는 아니다.
3. Single layer 2048 unit LSTM과 비교하여 Transformer의 효과 분석
   - Transformer 대신 LSTM을 이용하면 average score가 5.6만큼 떨어진다.

⇒ 사전 학습이 성능 향상에 중요한 영향을 미치고, LSTM보다 Transformer를 사용했을 때 성능이 좋다. Auxiliary object 사용은 큰 데이터셋에는 효과적지만 작은 데이터셋에는 아니다.

## Conclusion

**`GPT-1`**

- Transformer의 decoder를 사용
- Unsupervised Learning with unlabeled text for pre-training
- Fine tuning without additional task specific model

GPT-1은 생성적 사전 학습(generative pre-training)의 힘을 보여주었고, 더 큰 데이터셋과 더 많은 매개변수로 이러한 잠재력을 더 잘 발휘할 수 있는 다른 모델에 대한 길을 열어주었다. 그리고 이듬해 나온 GPT-2가 바로 그러한 모델 중 하나다.

---

## 추가

### Fine-tuning, N-shot learning

전이 학습(Transfer Learning)에서 downstream task를 학습하는 방식

- **Fine-tuning**  downstream task에 해당하는 데이터 **전체**를 사용한다. 모델 전체를 downstream 데이터에 맞게 업데이트한다.
- **Zero-shot learning**  downstream task 데이터를 **전혀** 사용하지 않는다. 모델이 바로 downstream task를 수행한다.
- **One-shot learning**  downstream task 데이터를 **한 건만** 사용한다. 모델 전체를 한 건의 데이터에 맞게 업데이트한다. 업데이트 없이 수행하는 one-shot learning도 있다. 모델이 1건의 데이터가 어떻게 수행되는지 참고한 뒤 바로 downstream task를 수행한다.
- **Few-shot learning**  downstream task 데이터를 **몇 건만** 사용한다. 모델 전체를 몇 건의 데이터에 맞게 업데이트한다. 업데이트 없이 수행하는 few-shot learning도 있다. 모델이 몇 건의 데이터가 어떻게 수행되는지 참고한 뒤 바로 downstream task를 수행한다.

_출처 [ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/docs/introduction/transfer/)_

### OpenAI GPT models

1. **GPT-1** paper ([Improving Language Understanding by Generative Pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)).
   - Fine-tuning
   - 매개변수: 약 1억 1700만 개
   - 활용
     - 주어진 두 문장의 관계 유추
     - 주어진 두 문장 간의 유사도 계산
     - 하나의 정보가 주어졌을 때 답 찾기
     - 특정 그룹으로 분류

2. **GPT-2** paper ([Language Models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) 

   GPT-1에 대한 후속 개선사항

   - Zero-shot learning
   - 매개변수: 약 15억 개
   - 활용
     - 독해 (Reading Comprehension)
     - 글 짓기

3. **GPT-3** paper [(Language models are few shot learners](https://arxiv.org/pdf/2005.14165.pdf)) 

   오늘날까지 자연어 처리에서 가장 강력한 모델 중 하나

   - Few-shot learning
     - fine-tuning 없이도 엄청난 성능 발휘
   - 매개변수: 약 1750억 개
   - 활용
     - 기사 작성
     - 상식 Q&A
     - 검색 엔진
     - 대화
     - 텍스트 요약

_위 논문들은 [여기](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)에 순차적으로 잘 정리되어 있다._

_Cf. GPT-1, GPT-2와 비교한 GPT-3의 어마어마한 파라미터 개수_

<img src="https://research.aimultiple.com/wp-content/uploads/2021/01/number-of-model-parameters-from-Elmo-to-Turing-NLG-1536x917.png" alt="img" style="zoom:40%;" /><img src="https://research.aimultiple.com/wp-content/uploads/2021/01/number-of-model-parameters-until-gpt-3.png" alt="img" />

_출처 https://research.aimultiple.com/gpt/_
  

## 	참고 자료

**논문**  [Improving Language Understanding by Generative Pre-Training](https://arxiv.org/pdf/1706.03762.pdf)

**영상**  [GPT-1 (밑바닥부터 알아보는 GPT 1강) Minsuk Heo 허민석](https://www.youtube.com/watch?v=FeEmmylAF0o_) 🌟

**영상**  [[Paper Review] Improving Language Understanding by Generative Pre-Training](https://www.youtube.com/watch?v=4qv_ofZN5_U) 🌟

**문서**  [The Journey of Open AI GPT models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2) 🌟

**문서**  [딥 러닝을 이용한 자연어 처리 입문 / 바이트 페어 인코딩](https://wikidocs.net/22592)

**문서**  [ratsgo's NLPBOOK](https://ratsgo.github.io/nlpbook/docs/introduction/transfer/) 🌟

**문서**  [ratsgo's blog / discriminative vs generative](https://ratsgo.github.io/generative%20model/2017/12/17/compare/)

