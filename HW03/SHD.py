import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 상수
BATCH_SIZE = 64
IMG_HEIGHT = 251
IMG_WIDTH = 251
NUM_CLASSES = 4

# 데이터셋 경로
data_dir = 'Multi-class Weather Dataset'

# path, label list 초기화
train_filepaths = []
train_labels = []
test_filepaths = []
test_labels = []

classlist= os.listdir(data_dir) # data directory 안의 class directory list
for klass in classlist: # 각 class 폴더에 대해
    # class별로 list 초기화
    X = []
    y = []

    classpath=os.path.join(data_dir, klass) # class dir 경로
    file_list=os.listdir(classpath) # class dir에 들어있는 file list
    
    for f in file_list: # 각 file에 대해
        fpath=os.path.join(classpath, f)
        X.append(fpath) # filepath 추가
        y.append(klass) # label 추가(class 폴더명)
    
    # 8:2로 train 데이터와 test 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # train path, label과 test path, label
    train_filepaths.extend(X_train)
    train_labels.extend(y_train)
    test_filepaths.extend(X_test)
    test_labels.extend(y_test)

# train path, label로 dataframe 생성
Fseries=pd.Series(train_filepaths, name='filename')
Lseries=pd.Series(train_labels, name='class')
train_df=pd.concat([Fseries, Lseries], axis=1)
train_df = train_df.sample(frac=1).reset_index(drop=True)

# test path, label로 dataframe 생성
Fseries=pd.Series(test_filepaths, name='filename')
Lseries=pd.Series(test_labels, name='class')
test_df=pd.concat([Fseries, Lseries], axis=1)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print(train_df.head(), len(train_df))
print(test_df.head(), len(test_df))

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# 이미지 데이터 설정
train_datagen = ImageDataGenerator(
    rescale=1/255, # 0~255의 rgb를 0~1 사이의 값으로 일반화
    rotation_range=40, # 무작위 회전각도 범위
    #shear_range=0.2, # 층밀리기 강도
    zoom_range=0.2, # 무작위 줌 범위
    horizontal_flip=True, # input을 무작위로 가로로 뒤집음
    fill_mode='nearest', # 경계 바깥공간 채우기 모드
    validation_split=0.2) # 8:2로 training, validation 데이터 분할
test_datagen = ImageDataGenerator(rescale=1/255) # 테스트 데이터는 기본 설정 

train_ds = train_datagen.flow_from_dataframe(
    train_df, # dataframe
    class_mode='categorical', # classification이 목적이므로 categorical
    subset="training", # training dataset
    target_size=(IMG_HEIGHT,IMG_WIDTH), # resize될 크기
    batch_size=BATCH_SIZE # batch 크기
)
val_ds = train_datagen.flow_from_dataframe(
    train_df,
    class_mode='categorical',
    subset="validation", # validation dataset
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)
test_ds = test_datagen.flow_from_dataframe(
    test_df,
    class_mode='categorical',
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)

## 모델 정의
import tensorflow as tf
model = tf.keras.models.Sequential([
  # the input shape is the size of the image 251x251 with 3 bytes color
  # first convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  tf.keras.layers.MaxPooling2D(3, 3), # first pooling
  # The second convolution
  tf.keras.layers.Conv2D(64, (3,3), padding='same', strides=2, activation='relu'),
  # The third convolution
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2), # second pooling
  # The fourth convolution
  tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=2, activation='relu'),
  # The fifth convolution
  tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2), # third pooling
  # The sixth convolution
  tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=2, activation='relu'),
  tf.keras.layers.MaxPooling2D(3, 3), # fourth pooling
  # Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  # 2048 neuron hidden layer
  tf.keras.layers.Dense(2048, activation='relu'),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # 4 weather class
])
model.summary()

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy', # 손실함수
    optimizer='rmsprop', # 최적화함수
    metrics=['accuracy'] # 척도
)

# 모델 학습
history = model.fit(
    train_ds, # 학습할 training dataset
    validation_data=val_ds, # validation dataset
    batch_size=BATCH_SIZE,
    steps_per_epoch=len(train_df)*0.8//BATCH_SIZE,
    epochs=100,
    verbose=1,
    validation_steps=len(train_df)*0.2//BATCH_SIZE
)

# 학습 결과 및 Plot 출력
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

# test dataset으로 모델 정확도 평가
loss, accuracy = model.evaluate(
    test_ds,
    batch_size=BATCH_SIZE,
    steps=len(test_df)//BATCH_SIZE
)
print(loss, accuracy)