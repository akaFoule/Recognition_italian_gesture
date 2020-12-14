from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# Creazione del file con le etichette
def make_label(text):
    with open("label.txt", "w") as f:
        f.write(text)
    f.close()

def rm_zero_sequences_endlist(txtlist):
    numbers = txtlist
    if (numbers[-1] == 0 and numbers[-2] == 0 and numbers[-3] == 0 and numbers[-4] == 0 and numbers[-5] == 0 and
            numbers[-6] == 0
            and numbers[-7] == 0 and numbers[-8] == 0 and numbers[-9] == 0 and numbers[-10] == 0 and numbers[
                -11] == 0 and numbers[-12] == 0
            and numbers[-13] == 0 and numbers[-14] == 0 and numbers[-15] == 0 and numbers[-16] == 0 and numbers[
                -17] == 0 and numbers[-18] == 0
            and numbers[-19] == 0 and numbers[-20] == 0 and numbers[-21] == 0 and numbers[-22] == 0 and numbers[
                -23] == 0 and numbers[-24] == 0
            and numbers[-25] == 0 and numbers[-26] == 0 and numbers[-27] == 0 and numbers[-28] == 0 and numbers[
                -29] == 0 and numbers[-30] == 0
            and numbers[-31] == 0 and numbers[-32] == 0 and numbers[-33] == 0 and numbers[-34] == 0 and numbers[
                -35] == 0 and numbers[-36] == 0
            and numbers[-37] == 0 and numbers[-38] == 0 and numbers[-39] == 0 and numbers[-40] == 0 and numbers[
                -41] == 0 and numbers[-42] == 0):
        numbers = numbers[:-42]
    return numbers

# Per stampare la matrice di confusione
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Questa funzione stampa a grafico la matrice di confusione
    La normalizzazione può essere applicata settando `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=13)

    plt.tight_layout()
    plt.ylabel('Etichette originali', fontsize=15)
    plt.xlabel('Etichette predette', fontsize=15)

"""
Funzione per la fase di test
"""
def load_testdata(dirname):
    listfile = os.listdir(dirname)
    XT = []
    YT = []
    for file in listfile:
        if "_" in file:
            continue
        wordname = file
        textlist = os.listdir(dirname + wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname = dirname + wordname + "/" + text

            with open(textname, mode='r') as t:
                numbers = [float(num) for num in t.read().split()]
                while numbers[0] == 0:
                    numbers = numbers[1:]
                for i in range(len(numbers), 4200):
                    numbers.extend([0.000])
            landmark_frame = []
            row = 0
            for i in range(0, 35):
                landmark_frame.extend(numbers[row:row + 84])
                row += 84
            landmark_frame = np.array(landmark_frame)
            landmark_frame = landmark_frame.reshape(-1, 84)
            XT.append(np.array(landmark_frame))
            YT.append(wordname)
    XT = np.array(XT)
    YT = np.array(YT)

    tmp1 = [[xt, yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)

    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]

    k = set(YT)
    ks = sorted(k)
    text = ""
    for i in ks:
        text = text + i + " "

    s = Tokenizer()
    s.fit_on_texts([text])
    encoded1 = s.texts_to_sequences([YT])[0]
    one_hot2 = to_categorical(encoded1)

    (x_test, y_test) = XT, one_hot2
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test

"""
Estrazione delle features, e creazione dei dati di test.
Vengono create 2 categorie, dati di training e dati di test.
"""
def load_data(dirname):
    if dirname[-1] != '/':
        dirname = dirname + '/'
    listfile = os.listdir(dirname)  # lista contenente i file
    X = []
    Y = []
    XT = []
    YT = []

    for file in listfile:  # Sfoglia i wordname
        if ".DS_Store" in file:
            continue
        wordname = file
        textlist = os.listdir(dirname + wordname)
        k = 0
        for text in textlist:
            if ".DS_Store" in text:
                continue
            textname = dirname + wordname + "/" + text
            numbers = []
            # print(textname)
            with open(textname, mode='r') as t:
                numbers = [float(num) for num in t.read().split()]
                # print(len(numbers))
                while numbers[0] == 0:
                    numbers = numbers[1:]
                for i in range(len(numbers), 4200):  # 50frame * 84 = 4200
                    numbers.extend([0.000])
            row = 0
            landmark_frame = []
            for i in range(0, 35):
                landmark_frame.extend(numbers[row:row + 84])
                row += 84
            landmark_frame = np.array(landmark_frame)  # (5880,) 1dim
            landmark_frame = landmark_frame.reshape(-1, 84)  # (70,84) 2dim
            if (k%5 == 4):
                XT.append(np.array(landmark_frame))
                YT.append(wordname)
            else:
                X.append(np.array(landmark_frame))
                Y.append(wordname)
            k += 1

    X = np.array(X)
    Y = np.array(Y)
    XT = np.array(XT)
    YT = np.array(YT)

    tmp = [[x, y] for x, y in zip(X, Y)]
    random.shuffle(tmp)

    tmp1 = [[xt, yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)

    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]

    k = set(Y)
    ks = sorted(k)
    text = ""
    for i in ks:
        text = text + i + " "
    make_label(text)

    s = Tokenizer()
    s.fit_on_texts([text])
    encoded = s.texts_to_sequences([Y])[0]
    encoded1 = s.texts_to_sequences([YT])[0]
    one_hot = to_categorical(encoded)
    one_hot2 = to_categorical(encoded1)

    (x_train, y_train) = X, one_hot
    (x_test, y_test) = XT, one_hot2
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

"""
Costruzione della rete neurale LSTM.
"""
def build_model(label):
    model = Sequential()
    model.add(layers.LSTM(256, return_sequences=True,
                          input_shape=(35, 84)))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(16))
    model.add(layers.Dense(label, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])
    return model


def main(dirname):
    print('Caricamento dei dati:')
    print('========================================================')
    x_train, y_train, x_test, y_test = load_data(dirname) #Creazione dei vettori e split delle cartelle
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print('Creazione del modello:')
    print('========================================================')
    model = build_model(y_train.shape[1])
    model.summary()

    print('Addestramento della rete:')
    print('========================================================')

    checkpoint = ModelCheckpoint('modello_rete.h5', monitor='val_acc', verbose=1, mode='max', save_best_only=True,
                                 save_weights_only=False, period=1)
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test),
                        callbacks=[checkpoint])

    epoche = 50
    # Stampa di tutti parametri delle history
    print(history.history.keys())

    # Valutazione del modello
    score, acc = model.evaluate(x_train, y_train, batch_size=32, verbose=0)
    print('\nTest performance: accuracy={0}, loss={1}'.format(acc, score))

    print("    x        ||      y       ")
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    model.save('modello_rete.h5')

    return epoche, history, model

"""
Il codice dopo aver creato il modello, lo testa e lo allena.
"""
output_data_path = '/Users/drissouissiakavaleriofoule/Desktop/outputvideo/Relative/'
epochs, history, model = main(output_data_path)

# summarize history for accuracy
plt.plot(range(epochs),history.history['acc'])
plt.plot(range(epochs),history.history['val_acc'], 'o--')
plt.title('Accuratezza del modello')
plt.ylabel('Accuracy')
plt.xlabel('Epoche')
plt.legend(['Training', 'Validazione'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(range(epochs),history.history['loss'])
plt.plot(range(epochs),history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoche')
plt.legend(['Training', 'Validità'], loc='upper left')
plt.show()

# Creazione dei dati di test, utilizzo la cartella Relative
dirname = '/Users/drissouissiakavaleriofoule/Desktop/outputvideo/Relative/'
x_test, y_test = load_testdata(dirname)
new_model = tf.keras.models.load_model('modello_rete.h5')
x = x_test
yhat = new_model.predict(x)
print("Accuratezza", accuracy_score(np.argmax(y_test, axis=1), np.argmax(yhat, axis=1)))
print("Precisione", precision_score(np.argmax(y_test, axis=1), np.argmax(yhat, axis=1), average='macro'))
print("Recall", recall_score(np.argmax(y_test, axis=1), np.argmax(yhat, axis=1), average='micro'))

# Costruzione della matrice di confusione
cfm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(yhat, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(15, 6))
class_names = ['Bombazza', 'Bacio', 'Buono', 'Omg', 'Pazzo']
class_names = sorted(class_names)
plot_confusion_matrix(cfm, classes=class_names, title='Matrice di confusione\n', normalize=False)
plt.show()
