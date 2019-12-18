import keras
from keras.models import model_from_json, load_model
import tensorflow as tf
import numpy as np
from keras.models import load_model
import os
import cv2
import datetime
from tensorflow.python.keras import backend as k
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from time import time
from keras.optimizers import SGD
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from Recortar import recortar

#extraindo os dados de teste e de treino da base de dados
arquivo = os.listdir("/Users/gustavozagocanal/PycharmProjects/IAII2/caracteres")
x_treino=[]
y_treino=[]
c = 0
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
# STAMP = 'simple_lstm_glove_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
# bst_model_path = STAMP + '.h5'
# model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

caracteres = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def convert_index(index):
    for i in range(len(caracteres)):
        if caracteres[i] == index:
            return i

for arq in arquivo:
    if arq:
        x_treino.append(cv2.imread("caracteres/"+arq))

    c = convert_index(arq[0])

    y_treino.append(c)

totaltreino = len(x_treino)
norma = 255

for i in range (totaltreino):
    x_treino[i] = cv2.cvtColor(x_treino[i], cv2.COLOR_BGR2GRAY)
    trh, x_treino[i] = cv2.threshold(x_treino[i],128,255, cv2.THRESH_BINARY)
    x_treino[i] = cv2.resize(x_treino[i], (30,50))
    x_treino[i] = cv2.resize(x_treino[i], (1500,1))
    x_treino[i] = x_treino[i]/norma
    x_treino[i] = x_treino[i][0]
imagesize = 784
ncls = max(y_treino)+1
x_treino = np.asfarray(x_treino)
y_treino = keras.utils.to_categorical(y_treino, num_classes=ncls)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu, input_shape = (1500,)))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(ncls, activation = tf.nn.softmax))

# sgd = SGD(lr=1e-3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])

# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def save_weights(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Pesos salvos com sucesso!!!")

def load_weights(model):
    model.load_weights("best_79.h5")
    print("Pesos carregados com sucesso!!!")

def treinar_rede():
    # filepath="~/PycharmProjects/IAII/IAII/weights.best.h5"
    # filepathAtual="~/PycharmProjects/IAII/IAII/weights.current.hdf5"
    checkpoint = ModelCheckpoint('best.h5', verbose=1, monitor='acc', save_best_only=True, mode='auto')
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit(x_treino, y_treino, epochs = 1000, verbose = 2, callbacks=[checkpoint, tensorboard_callback], shuffle=False)

def Verificar_Placa(placa, img):
    y_predict_x_treino = model.predict(x_treino)
    y_predict_placa = model.predict(placa)
    y_predict_placa = tf.keras.backend.get_value(tf.keras.backend.argmax(y_predict_placa, axis=-1))
    y_predict_x_treino = tf.keras.backend.get_value(tf.keras.backend.argmax(y_predict_x_treino, axis=-1))
    y_treino2 = tf.keras.backend.get_value(tf.keras.backend.argmax(y_treino, axis=-1))
    for j in range(len(x_treino)):
        print("Predicted=%s, padrao = %s" %(caracteres[y_predict_x_treino[j]], caracteres[y_treino2[j]]))

    for i in range(len(placa)):
        print("Predicted=%s " % caracteres[y_predict_placa[i]])
    cv2.imshow("Placa", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mostrar_pesos():
    print(model.layers[-1].get_weights())
    print(model.history['acc'])

def main():
  x = '1'
  while (x!='10'):
    print('Digite a opção que você deseja escolher: \n')
    print('1 - Treinar a rede')
    print('2 - Verificar a placa')
    print('3 - Salvar pesos sinapticos')
    print('4 - Carregar pesos sinapticos')
    print('5 - Mostrar pesos sinapticos')
    print('10 - Sair do sistema ')

    x = input()
    if x == '1':
        treinar_rede()
    elif x == '2':
        root = Tk()
        root.filename = filedialog.askopenfilename(initialdir="/Users/gustavozagocanal/PycharmProjects/IAII2/Imagens", title="Select A File", filetypes=(( "png files", ".png"),("all files", "*.*")))
        path = root.filename
        root.destroy()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        placa = recortar(img)
        Verificar_Placa(placa, img)
    elif x == '3':
        save_weights(model)
    elif x == '4':
        load_weights(model)
    elif x == '5':
        mostrar_pesos()
    elif x == '10':
        x == '10'
    else:
        print('Opção selecionada não existe, por favor digite novamente')
main()
