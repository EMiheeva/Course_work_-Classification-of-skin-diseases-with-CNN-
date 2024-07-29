"""
Датасет для обучения:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

В него включены 7 классов кожных заболеваний:
Меланоцитарные невусы - Melanocytic nevi (nv)
Меланома - Melanoma (mel)
Доброкачественные кератозоподобные поражения - Benign keratosis-like lesions (bkl)
Базальноклеточная карцинома - Basal cell carcinoma (bcc) 
Актинический кератоз - Actinic keratoses (akiec)
Сосудистые поражения - Vascular lesions (vas)
Дерматофиброма  - Dermatofibroma (df)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils import to_categorical # Используется для преобразования меток в однократное кодирование(one-hot-encoding)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

'''
Однократное кодирование (one-hot-encoding) - это метод представления категориальных данных 
в виде бинарных векторов. Этот метод широко используется в машинном обучении 
для обработки категориальных признаков, таких как цвет, тип, метка класса и так далее.

Идея заключается в том, чтобы каждая категория была представлена в виде бинарного вектора, 
где все элементы равны 0, за исключением одного, соответствующего индексу этой категории, который равен 1. 
Таким образом, каждая категория будет представлена уникальным бинарным вектором.

Например, если у нас есть три категории: "красный", "зеленый" и "синий", 
то их можно представить в виде бинарных векторов следующим образом:
- "красный" - [1, 0, 0]
- "зеленый" - [0, 1, 0]
- "синий" - [0, 0, 1]

Этот метод позволяет алгоритмам машинного обучения работать с категориальными данными, 
такими как цвета или метки классов, которые не могут быть обработаны напрямую числовыми методами.
'''

# Метаданные, хранящие столбцы lesion_id,image_id, dx, dx_type, age, sex, localization
skin_df = pd.read_csv('data_kaggle/archive/HAM10000_metadata.csv')

SIZE=32 # Размер изображений для масштабирования

# Кодирование меток в числовые значения из текста
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
print(list(le.classes_))
 
skin_df['label'] = le.transform(skin_df["dx"]) 
print(skin_df.sample(10)) #Выводим столбцы lesion_id,image_id,dx,dx_type,age,sex,localization


# Визуализация распределения данных
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)

ax1.set_ylabel('Количество')
ax1.set_title('График с 7 классами заболеваний');

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)

ax2.set_ylabel('Количество', size=15)
ax2.set_title('График пола людей');

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')

ax3.set_ylabel('Количество',size=12)
ax3.set_title('График расположения заболеваний')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]

sns.distplot(sample_age['age'], fit=stats.norm, color='blue');
ax4.set_ylabel('Количество', size=15)
ax4.set_title('График возрастов людей')


plt.tight_layout()
plt.show()


# Распределение данных по различным классам
from sklearn.utils import resample
print(skin_df['label'].value_counts())

# Сбалансировка данных.
# Разделяем каждый класс, повторяем выборку и объединяем обратно в один дата-фрейм
df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

# Объединяем обратно в один дата-фрейм
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])

# Проверяем распределение данных. Теперь все классы должны быть сбалансированы.
print(skin_df_balanced['label'].value_counts())


# Прочитаем изображения на основе идентификатора изображения из CSV-файла
# Это самый безопасный способ чтения изображений, поскольку он гарантирует, 
# что нужное изображение будет готово для правильного идентификатора
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('data_kaggle/archive/', '*', '*.jpg'))}

# Определяем путь и добавляем в новый столбец
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
# Используем путь для чтения изображений. 
# lambda-функция, которая: принимает путь к изображению x, открывает изображение с помощью Image.open(x),
# изменяет размер изображения до указанного размера (SIZE, SIZE) с помощью resize, 
# а затем преобразует его в массив numpy с помощью np.asarray.
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


n_samples = 5  # количество выборок для построения графика
# Построение графика
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
plt.show()

# Преобразование столбцы данных с изображениями в массив numpy
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.  # Масштабирование значения до 0-1. 
Y = skin_df_balanced['label']  # Присвоение метки к значению
Y_cat = to_categorical(Y, num_classes=7) #Преобразовать в категориальный, поскольку это проблема многоклассовой классификации
# Разделение на данные для обучения(train) и тестирования(test) с помощью функции train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

#Определение модели.
#Мы использовали automakers чтобы найти наилучшую модель для решения поставленной задачи классификации.

num_classes = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])


# Тренируем модель.
batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print()
print('Результат [Test accuracy]:', score[1])
print()


# Построим график точности обучения и валидации и потерь в каждую эпоху.
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Потери обучения')
plt.plot(epochs, val_loss, 'g', label='Потери валидации')
plt.title('Потери обучения и валидации')
plt.xlabel('Эпохи') # Полная итерация алгоритма над обучающим набором данных.
plt.ylabel('Потери') 
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'b', label='Точность обучения')
plt.plot(epochs, val_acc, 'g', label='Точность валидации')

plt.title('Точность обучения и валидации') 
plt.xlabel('Эпохи') # Полная итерация алгоритма над обучающим набором данных

plt.ylabel('Точность') # Показатель, насколько точно модель способна выдавать правильные ответы
plt.legend()
plt.show()


# Прогнозируем на основе тестовых данных
y_pred = model.predict(x_test)
# Преобразовываем классы предсказаний в виде бинарных векторов (one-hot-vector)
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Преобразовываем тестовые данные в виде бинарных векторов (one-hot-vector)
y_true = np.argmax(y_test, axis = 1) 

# Выводим матрицу неточностей
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
plt.show()


# Построим график, показывающие диаграмму неверных предсказаний
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('Истинные прогнозы')
plt.ylabel('Ложные прогнозы')
plt.show()
