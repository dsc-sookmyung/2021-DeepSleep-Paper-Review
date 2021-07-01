# [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

- 목차
  - [개요](#개요)
  - [구조](#구조)
    - [Residual Learning](#residual-learning)
    - [Identity Mapping by Shortcuts](#identity-mapping-by-shortcuts)
    - [Network Architectures](#network-architectures)
  - [구현](#구현)

## 개요
* 내용
  * 이전보다 훨씬 더 깊은 네트워크 학습을 용이하게 하기 위한 `Residual Learning framework`
* 배경
  * VGGNet, GoogLeNet 등을 통해, 네트워크의 깊이가 매우 중요하다는 사실을 깨달았다.
  * 문제 1 : `vanishing/exploding gradients`
    * 학습할 때, 레이어가 깊어질수록 전달되는 오차가 줄어들어 기울기가 소멸하여 학습이 되지 않는 현상
    * 해결 : 
      * `normalized initialization`
      * `batch normalization`
      * 이를 통해, 수십개의 레이어를 가진 네트워크를 수렴시킬 수 있다.
  * 문제 2 : `degradation problem`
    * 적절한 깊이의 모델에 레이어를 추가하였더니 training error가 증가하는 현상
    * 오버피팅 문제가 아님, 최적화가 힘든 것
    * 해결 : 
      * 더 깊은 모델이 더 얕은 모델보다 더 큰 에러를 만드는 것은 안 된다는 생각(비슷하거나 더 나아야 한다)
      * 따라서 모델에 `identity mapping` 레이어를 추가한다.
      * `deep residual learning framework` : 이 논문의 주제

## 구조
### `Residual Learning`
<img src="https://user-images.githubusercontent.com/35680202/123977910-5ff6e700-d9fa-11eb-8950-73971fa0baaa.png" width="450" height="300">

* original 에서 $H(x) = x$ 를 배우는 것보다, residual learning 에서 $F(x) = 0$ 을 배우는 것이 더 쉽다.

### `Identity Mapping by Shortcuts`
#### 수학적 표현
* $y = F(x, \{W_i\}) + x$
  * $x$ : input vector
  * $y$ : output vector
  * $F(x, \{W_i\})$ : residual mapping
  * $F + x$ : shortcut connection and element-wise addition
  * **$x$ 와 $F$ 의 차원(dimension)이 반드시 같아야 한다.**
* $y = F(x, \{W_i\}) + W_s x$
  * $W_s$ : 차원을 맞추기 위해서만 사용된다.
#### 특징
* shortcut connection 에서는 **추가 매개 변수나 계산 복잡성이 발생하지 않는다.**
* $F$가 하나의 레이어로 구성된다면 그냥 선형 레이어와 유사해지고, 이점이 없어진다.
* $F$ 는 FC layer 뿐만 아니라, 여러 개의 Conv layers 가 될 수 있다.

### `Network Architectures`
#### Plain Network
* 주로 VGGNet 의 철학에서 영감을 받아서 구성되었다.
  * **3x3 conv** 만 사용한다.
  * 동일한 출력 피쳐 맵 크기에 대해 레이어는 동일한 수의 필터를 갖는다.
  * **피쳐 맵 크기가 절반으로 줄어들면 필터 수가 두 배로 증가**하여 계층당 시간 복잡성을 줄인다.
* conv layer에서 downsampling 할 때만 stride=2
* global average pooling -> 1000-way fc layer with softmax
* weighted layer 개수는 총 34개

#### Residual Network
* 위의 Plain Network 를 기반으로, shortcut connections 를 넣어, 대응되는 residual 버전으로 만든다. (ResNet-34)
* shortcut
  * (A) identity mapping : dimension 이 같은 경우 그대로 더해주기
  * (B) projection shortcut : 1x1conv 이용해서 dimension 맞춰주기

## 구현
* 특징
  * conv -> **batchnorm** -> activation
  * dropout 사용안함
* 코드
  ```python
  from torch import nn
  from collections import OrderedDict

  # ========== Plain/Residual Block ====================
  class Block(nn.Module):
    def __init__(self, in_c=64, double=True, residual=False):
      super(Block, self).__init__()

      self.residual = residual
      self.double = double
      
      out_c = 2*in_c if double else in_c
      stride = 2 if double else 1

      self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(num_features=out_c),
        nn.ReLU()
      )
      self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(num_features=out_c)
      )
      self.downsample = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_c)
      )
      self.relu = nn.ReLU()

    def forward(self, x):
      out = self.conv1(x)
      out = self.conv2(out)
      # === The only difference between Plain and Residual ===
      if self.residual:
        if self.double: # match dimension
          x = self.downsample(x) # option (B) projection
        out = out + x
      # ======================================================
      out = self.relu(out)
      return out

  # ========== Plain/Residual Stage ====================
  class Stage(nn.Module):
    def __init__(self, num_blocks=3, in_c=64, double=True, residual=False):
      super(Stage, self).__init__()
    
      out_c = 2*in_c if double else in_c
      doubles = [double] + [False]*(num_blocks-1)
      channels = [in_c] + [out_c]*(num_blocks-1)

      self.stage = nn.Sequential(OrderedDict([]))
      for i in range(num_blocks):
        self.stage.add_module(f'block{i}', Block(in_c=channels[i], double=doubles[i], residual=residual))
    
    def forward(self, x):
      return self.stage(x)

  # ========== Plain/Residual Network ====================
  class Network(nn.Module):
    def __init__(self, residual=False):
      super(Network, self).__init__()
      # input : (3, 224, 224) (ignore batch size here)
      self.stem = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=2), # out : (64, 112, 112)
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # out : (64, 56, 56)
        nn.BatchNorm2d(num_features=64),
        nn.ReLU()
      )
      self.stage1 = Stage(num_blocks=3, in_c=64, double=False, residual=residual) # out : (64,  56, 56)
      self.stage2 = Stage(num_blocks=4, in_c=64,  double=True, residual=residual) # out : (128, 28, 28)
      self.stage3 = Stage(num_blocks=6, in_c=128, double=True, residual=residual) # out : (256, 14, 14)
      self.stage4 = Stage(num_blocks=3, in_c=256, double=True, residual=residual) # out : (512,  7,  7)
      self.avgpool = nn.AvgPool2d(kernel_size=7) # out : (512, 1, 1)
      self.fc = nn.Linear(in_features=512, out_features=1000) # out : (1000)
    
    def forward(self, x):
      N = x.shape[0] # batch size
      x = self.stem(x)
      x = self.stage1(x)
      x = self.stage2(x)
      x = self.stage3(x)
      x = self.stage4(x)
      x = self.avgpool(x)
      x = self.fc(x.reshape(N, -1))
      return x
  ```