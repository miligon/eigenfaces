#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:37:16 2022

@author: miguel
"""

from PIL import Image
import numpy as np
import random
import os

#SHAPE=(5,5)
SHAPE=(26,26)
#SHAPE=(130,180)
#SHAPE=(260,360)

def open_and_convert(file):
    # Abrir imagen
    imagen = Image.open(file)
    #print(type(imagen))
    #print("loading: ", file)
    
    # Imprimir atributos
    #print(imagen.mode)
    #print(imagen.size)
    W, H = imagen.size
    
    #imagen = Image.open(file).resize((int(W/5),int(H/5)))
    #imagen = Image.open(file).resize((130,180))
    imagen = Image.open(file).resize(SHAPE)
    imagen = imagen.convert('L')
    #imagen.show()
    
    # Convertir a array
    img_np = np.array(imagen)
    img_h, img_w = img_np.shape
    #print("image shape:", img_h, img_w)
    
    # Reshape a 1-D
    tam = img_h*img_w
    linear = img_np.reshape(1,tam)
    #linear.dtype = np.float32
    #print("Shape final: ", linear.shape)
    #print(linear[0].tolist())
    
    # Normalizar
    #n = linear/np.linalg.norm(linear)
    #print(np.float_(linear[0]))
    return np.float_(linear[0])#.tolist()

def add_to_array(attrib,data):
    if (len(attrib) == 0):
        for i in data:
            attrib.append([float(i)])
    else:
        for i in range(len(data)):
            attrib[i].append(float(data[i]))

def load_data(dataset_path, n_muestras_train = 25):
    attrib_e = []
    labels_e = []
    
    attrib_p = []
    labels_p = []
    
    file_names = os.listdir(dataset_path)
    
    #Barajeo
    random.shuffle(file_names)
        
    for i in range(len(file_names)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            attrib_e.append(open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_e.append((i,label))
        else:
            #print("pruebas", i, buffer[i])
            attrib_p.append(open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_p.append((i-n_muestras_train,label))
            
            
    return np.array(attrib_e), np.array(labels_e), np.array(attrib_p), np.array(labels_p)

def load_data_2(dataset_path, n_muestras_train = 25):
    attrib_e = []
    labels_e = []
    
    attrib_p = []
    labels_p = []
    
    file_names = os.listdir(dataset_path)
    
    #Barajeo
    random.shuffle(file_names)
        
    for i in range(len(file_names)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            add_to_array(attrib_e, open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_e.append((i,label))
        else:
            #print("pruebas", i, buffer[i])
            add_to_array(attrib_p, open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_p.append((i-n_muestras_train,label))
            
    return np.array(attrib_e), np.array(labels_e), np.array(attrib_p), np.array(labels_p)

def load_data_real_label(dataset_path, n_muestras_train = 25):
    attrib_e = []
    labels_e = []
    
    attrib_p = []
    labels_p = []
    file_p = []
    
    file_names = os.listdir(dataset_path)
    
    #Barajeo
    random.shuffle(file_names)
        
    for i in range(len(file_names)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            label = file_names[i].split('-')[1].split('.')[0]
            add_to_array(attrib_e, open_and_convert(dataset_path + file_names[i]))
            labels_e.append((i,float(label)))
        else:
            #print("pruebas", i, buffer[i])
            label = file_names[i].split('-')[1].split('.')[0]
            add_to_array(attrib_p, open_and_convert(dataset_path + file_names[i]))
            labels_p.append((i-n_muestras_train,float(label)))
            file_p.append(dataset_path + file_names[i])
            
    return np.array(attrib_e), np.array(labels_e), np.array(attrib_p), np.array(labels_p), file_p
        



