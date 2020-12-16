from __future__ import absolute_import, division, print_function, unicode_literals

import os

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from build_model import confusion_matrix, plot_confusion_matrix, plt, load_testdata
import numpy as np
import tensorflow as tf
import argparse

def load_data(dirname):
    listfile=os.listdir(dirname)
    X = []
    Y = []

    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)

        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text

            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                while numbers[0] == 0:
                    numbers = numbers[1:]
                for i in range(len(numbers),4200):
                    numbers.extend([0.000])

            landmark_frame=[]
            row=0

            for i in range(0,35):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y

# Per ottenere ogni etichetta nel file label.txt
def load_label():
    with open("label.txt", mode='r') as l:
        listfile = [i for i in l.read().split()]
        label = {}
        count = 1
        for l in listfile:
            if "_" in l:
                continue
            label[l] = count
            count += 1
        return label
    
def main(output_data_path):
    output_dir = output_data_path

    print("Caricamento dati")
    print("=========================================================")
    #x_test,Y =load_data(output_dir)
    #print("x_test:",x_test, "Y", Y)
    print("Caricamento completato!\n")

    print("New Model")
    print("=========================================================")
    #new_model = tf.keras.models.load_model('modello_rete.h5')
    print("Caricamento completato!\n")

    print("Etichette")
    print("=========================================================")
    labels = load_label()
    print(labels)
    print("Caricamento completato!\n")

    print("Predizione")
    print("=========================================================")
    #xhat = x_test
    #yhat = new_model.predict(xhat)
    #predictions = np.array([np.argmax(pred) for pred in yhat])
   # print(predictions)

    print("Rev Labels")
    print("=========================================================")
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    print(rev_labels)

    print("Scrittura file")
    print("=========================================================")
    s = 0
    count = 0
    txtpath = output_data_path + "result.txt"

   # for i in predictions:
    #    print("true_label: ",Y[s]," === ","predict_label: ",rev_labels[i])
     #   print("\n")
     #   if rev_labels[i] == Y[s]:
      #      count+=1
    #    s+=1
   # print("Numero di entrate dalla media: " + str(count))

    x_test, y_test = load_testdata(output_dir)
    new_model = tf.keras.models.load_model('modello.h5')
    x = x_test
    yhat = new_model.predict(x)

    print('Costruzione della matrice di confusione')
    cfm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(yhat, axis=1))
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10,10))

    class_names = ['Bombazza', 'Bacio', 'Buono', 'OMG', 'Pazzo', 'Ti prego']
    class_names = sorted(class_names)
    plot_confusion_matrix(cfm, classes=class_names, title='Matrice di confusione', normalize=False)
    plt.savefig('/Users/drissouissiakavaleriofoule/Desktop/TESI/matrix2.png')
    print('Salvataggio OK')
   # print("Accuratezza", accuracy_score(Y, rev_labels))
    #print("Precisione", precision_score(np.argmax(Y), np.argmax(yhat, axis=1), average='macro'))
    #print("Recall", recall_score(np.argmax(Y), np.argmax(yhat, axis=1), average='micro'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--output_data_path",help=" ")
    args=parser.parse_args()
    output_data_path = '/Users/drissouissiakavaleriofoule/Desktop/TESI/PROGETTO/CartellaVideo/outputvideo/Relative/'
    main(output_data_path)
