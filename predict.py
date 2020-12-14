from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import numpy as np
import tensorflow as tf

def load_data(dirname):
    if dirname[-1]!='/':
        dirname=dirname+'/'
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
                #print(len(numbers[0]))
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
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y


#Per ottenere ogni etichetta nel file label.txt
def load_label():
    with open("label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label

def main(input_data_path,output_data_path,data_path):
    comp='bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu'
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt'
    listfile=os.listdir(input_data_path)
    if not(os.path.isdir(output_data_path+"Relative/")):
        os.mkdir(output_data_path+"Relative/")
    if not(os.path.isdir(output_data_path+"Absolute/")):
        os.mkdir(output_data_path+"Absolute/")
    for file in listfile:
        if not(os.path.isdir(input_data_path+file)): #ignore .DS_Store
            continue
        word = file+"/"
        fullfilename=os.listdir(input_data_path+word)
        if not(os.path.isdir(output_data_path+"_"+word)):
            os.mkdir(output_data_path+"_"+word)
        if not(os.path.isdir(output_data_path+"Relative/"+word)):
            os.mkdir(output_data_path+"Relative/"+word)
        if not(os.path.isdir(output_data_path+"Absolute/"+word)):
            os.mkdir(output_data_path+"Absolute/"+word)
        os.system(comp)
        #outputfilelist=os.listdir(output_data_path+'_'+word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue
            inputfilen='   --input_video_path='+input_data_path+word+mp4list
            outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
            cmdret=cmd+inputfilen+outputfilen
            os.system(cmdret)

    #output_dir=output_data_path
    x_test,Y=load_data(data_path)
    new_model = tf.keras.models.load_model('model.h5')
    #new_model.summary()

    labels=load_label()
    print(labels)

    xhat = x_test
    yhat = new_model.predict(xhat)
    print(yhat[0])
    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    s=0
    txtpath=output_data_path + "result.txt"

    with open(txtpath, "w") as f:
        for i in predictions:
            f.write("true_label: ")
            f.write(Y[s])
            f.write(" === ")
            f.write("predict_label: ")
            f.write(rev_labels[i])
            f.write("\n")
            s+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    parser.add_argument("--output_data_path",help=" ")
    parser.add_argument("--data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    output_data_path=args.output_data_path
    data_path=args.data_path
    main(input_data_path,output_data_path,data_path)
