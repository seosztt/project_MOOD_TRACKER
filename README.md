# 주제: 감정 분석을 통한 음악 선곡과 무드 트래커 제공 서비스

> 무드트래커(MOOD TRACKER)란? 기분(MOOD)과 추적(TRACKER)의 합성어로 자신의 감정을 인식하고 추적하여 스스로 기분을 점검할 수 있도록 하는 활동입니다.

- 당신의 [오늘/어제/한달 전]은 어떤 기분이었나요? 당신의 기분을 노래로 표현해드립니다.
- 사용자의 사진을 통해 감정을 분류하고 누적하여 해당 감정과 관련된 노래를 선곡해주는 서비스
- 개인 일기장 같은 감성적인 GUI 구성, 감정 분석과 음악 매칭 기능이 접목된 사이트
- 로그인/회원가입/사진 업로드/감정 분석 결과&음악 매칭/무드 트래커 등이 합쳐진 하나의 다이어리 컨셉



# 얼굴 사진을 입력 받아 감정 상태를 출력하는 모델 학습

## 전처리

### 분류 클래스

AI허브에서 제공한 데이터에는 총 7개의 클래스(기쁨, 분노, 슬픔, 불안, 중립, 상처, 당황)로 분류되어있었으나 상처를 제외하기로 하였다. 왜냐하면

- 몇 번의 학습을 시도해본 결과 상처의 정확도가 다른 클래스에 비해 현저히 낮게 나왔다.

- 상처가 다른 감정들과 구분되는 뚜렷한 특징이 없는 듯하다. 상처입으면 기쁘진 않겠지만, 분노할 수도, 슬플 수도, 불안할 수도 있을 것 같다. 상처 클래스로 라벨링 된 사진을 봐도 분노에서도, 슬픔에서도, 당황에서도 봤던 사진 같다는 의아한 느낌이 든다.

### 이미지에서 얼굴 crop

- openCV 사용 -> 부정확하게 crop되는 경우 발생
- AI허브에서 제공하는 json파일의 좌표값을 사용

### input size

- crop한 이미지 데이터의 size의 height와 width의 평균을 계산해보니 각각 1100, 820이었다. 원본 이미지는 고화질이어서 학습 시간이 많이 소요될 것으로 여겼다. 그래서 height와 width를 1/4 하여 input size를 (275, 205)로 정하였다.
- keras의 ImageDataGenerator를 사용하여 이미지를 input size로 resize했다.

### 정성평가의 어려움

![sad_neu](C:\Users\USER\Final_Project\image\sad_neu.png)![sad_neu](C:\Users\USER\Final_Project\image\sad_new_result.png)

- AI허브에서 제공 받았을 때 이 사진은 sad로 labeling되어있었다. 그러나 이 사진의 표정이 어떤 감정을 나타내는지는 사람에게 물어도 뚜렷하게 대답하기 어렵다. 이 표정이 어떤 감정을 나타내는지 모든 사람이 동의하는 객관적 답은 없을 것이다. 학습시킨 모델이 출력하는 결과값을 보아도 happy를 제외한 모든 감정에 5%이상의 확률을 나타낸다.

- 일단은 AI허브에서 제공한 labeling에 따른 정확도를 높히는 것을 1차 목표로 해야하겠다. 그러나 사람마다 답이 다른 이런 문제에서 AI허브에서 제시한 답이 꼭 보편적으로 설득력 있는 답이라고 말하기는 어렵기에 예측값이 제시한 각 감정별 확률이 그럴싸하게 느껴지면 충분히 설득력 있는 모델이라고 생각할 수도 있겠다. 그러나 이것은 느낌에 의존하므로 정성평가이다.



## 모델링

### regulation

- image augmentation
- dropout layer

 ### 전이학습

SOTA ImageNet 분류 문제 순위, keras에서 지원하는지 여부, 가용 하드웨어가 감당할만한 모델인지를 검토하여 선정하였다.

- VGG16
- Xception
- resnet50
- alexnet

### fine tuning



### optimizer

- rmsprop
- adam

### batch_size

작게 할수록 학습 시간은 길어지지만 정확도는 개선됐다.

### overfitting

과적합이 될수록 예측값을 극단적으로 산출했다. 각 감정별 확률을 설득력있게 산출하는 모델을 만들고 싶었기에 validation loss를 monitor하여 best 모델을 저장하는 callback을 설정하여 정확도가 높은 모델보다는 validation loss가 낮은 모델을 우선 선택했다.

### ensemble



## 결과

