# 주제

`감정 분석을 통한 음악 선곡과 무드 트래커 제공 서비스`

## 기획 배경

- 증가하는 우울증, 자살, 고독사. 우리의 감정을 기록하고 되돌아 보자.

- 마스크로 잃어버린 우리의 표정을 되찾자.

## 프로젝트 목표

`무드 트레커 서비스 제공`

> 무드트래커(MOOD TRACKER)란? 기분(MOOD)과 추적(TRACKER)의 합성어로 자신의 감정을 인식하고 추적하여 스스로 기분을 점검할 수 있도록 하는 활동입니다.

- 당신의 [오늘/어제/한달 전]은 어떤 기분이었나요? 당신의 기분을 노래로 표현해드립니다.
- 사용자의 사진을 통해 감정을 분류하고 누적하여 해당 감정과 관련된 노래를 선곡해주는 서비스
- 개인 일기장 같은 감성적인 GUI 구성, 감정 분석과 음악 매칭 기능이 접목된 사이트
- 로그인/회원가입/사진 업로드/감정 분석 결과&음악 매칭/무드 트래커 등이 합쳐진 하나의 다이어리 컨셉



# 얼굴 사진을 입력 받아 감정 상태를 출력하는 모델 학습

## 전처리

### 분류 클래스

AI허브에서 제공한 데이터에는 총 7개의 클래스(기쁨, 분노, 슬픔, 불안, 중립, 상처, 당황)로 분류되어있었으나 상처를 제외하기로 하였다. 왜냐하면

- 몇 번의 학습을 시도해본 결과 상처의 정확도가 다른 클래스에 비해 낮게 나왔다.
- 상처가 다른 감정들과 구분되는 뚜렷한 특징이 없는 듯하다. 상처입으면 기쁘진 않겠지만, 분노할 수도, 슬플 수도, 불안할 수도 있을 것 같다. 상처 클래스로 라벨링 된 사진을 봐도 분노에서도, 슬픔에서도, 당황에서도 봤던 사진 같다는 의아한 느낌이 든다.

따라서 6개의 클래스로 분류하는 모델을 만들 것이다.

### 이미지에서 얼굴 crop

- openCV 사용 -> 부정확하게 crop되는 경우 발생
- AI허브에서 제공하는 json파일의 좌표값을 사용

### input size

- crop한 이미지 데이터의 size의 height와 width의 평균을 계산해보니 각각 1100, 820이었다. 원본 이미지는 고화질이어서 학습 시간이 많이 소요될 것으로 여겼다. 그래서 height와 width를 1/4 하여 input size를 (275, 205)로 정하였다.
- keras의 ImageDataGenerator를 사용하여 이미지를 input size로 resize했다.

### 정성평가의 어려움

![sad_neu](https://github.com/seosztt/project_MOOD_TRACKER/blob/master/image/sad_neu.png?raw=true)![sad_neu](https://github.com/seosztt/project_MOOD_TRACKER/blob/master/image/sad_new_result.png?raw=true)

- AI허브에서 제공 받았을 때 이 사진은 sad로 labeling되어있었다. 그러나 이 사진의 표정이 어떤 감정을 나타내는지는 사람에게 물어도 뚜렷하게 대답하기 어렵다. 이 표정이 어떤 감정을 나타내는지 모든 사람이 동의하는 객관적 답은 없을 것이다. 학습시킨 모델이 출력하는 결과값을 보아도 happy를 제외한 모든 감정에 5%이상의 확률을 나타낸다.

- 일단은 AI허브에서 제공한 labeling에 따른 정확도를 높히는 것을 1차 목표로 해야하겠다. 그러나 사람마다 답이 다른 이런 문제에서 AI허브에서 제시한 답이 꼭 보편적으로 설득력 있는 답이라고 말하기는 어렵기에 예측값이 제시한 각 감정별 확률이 그럴싸하게 느껴지면 충분히 설득력 있는 모델이라고 생각할 수도 있겠다. 그러나 이것은 느낌에 의존하므로 정성평가이다.

## 모델링

### regulation

- image augmentation
- dropout layer

### 전이학습

> 다음을 검토하여 선정하였다.
>
> - SOTA ImageNet 분류 문제 순위
> - keras에서 지원하는지 여부
> - 가용 하드웨어가 감당할만한 크기의 모델인지

- VGG16
- Xception
- resnet50
- alexnet

### fine tuning

처음에는 import한 모델이 직접 학습시킨 모델보다 훨씬 강력할 것이라 여겨 import한 모델의 파라미터를 훼손하지 않고자 결과 출력층 전의 1~2개 층만 동결을 해제 했었다. 그러나 실험 결과 합성곱층의 절반 가까이 동결을 해제했더니 학습시간이 증가하긴 했지만 정확도가 개선되었다.

### optimizer

- rmsprop
- adam

### batch_size

작게 할수록 학습 시간은 길어지지만 정확도는 개선됐다.

### overfitting

과적합이 될수록 예측값을 극단적으로 산출했다. 각 감정별 확률을 설득력있게 산출하는 모델을 만들고 싶었기에 validation loss를 monitor하여 best 모델을 저장하는 callback을 설정하여 정확도가 높은 모델보다는 validation loss가 낮은 모델을 우선 선택했다.

### ensemble

- resnet50은 정확도가 제일 높기 때문에 가중치를 높게 설정했다.
- alexnet이 unrest클래스에 대한 recal이 상대적으로 높아서 가중치를 높게 설정했다.

## 결과

### validation: loss / accuracy 

VGG16: 0.8344 / 0.7004

Xception: 0.9072 / 0.6842

resnet50: 0.9213 / 0.7287

alexnet: 0.7719 / 0.6982

ensemble(25:20:30:25): 0.7219 / 0.7533

### Confusion Matrix

![confusion_matrix](https://github.com/seosztt/project_MOOD_TRACKER/blob/master/image/confusion_matrix.png?raw=true)

- Confusion Martix를 통해 클래스별 precision과 recall을 확인하였다. unrest가 다른 클래스에 비해 f1-score가 낮게 나왔다.
- unrest의 f1-score를 0.5 이상으로 높이는 것을 목표로 하이퍼파라미터 튜닝과 ensemble을 시도하였다. 그 결과 unrest의 f1-score 0.51을 달성하였다.

### test

최종적으로 ensemble 모델을 사용하여 서비스를 구현하였다.

test Loss: 0.69735

test Accuracy: 0.7568



# 음악 선곡

- 감성 상태에 부합하는 키워드에 해당하는 음악 선곡

- 저작권 문제로 클래식을 주로 선곡

- 유튜브 API를 이용한 동영상 재생



# 후기

