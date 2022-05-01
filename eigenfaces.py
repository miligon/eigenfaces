#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:33:10 2022

@author: miguel
"""

import numpy as np
from PIL import Image
import math
import statistics
from matplotlib import pyplot as plt
import imagenes



class EIGENFACES:
    def __init__(self):
        self.data = []
        self.k = 10

    def setK(self, k):
        self.k = k

    def copy_data(self, i, dst):
        if len(dst) == len(self.data):
            for j in range(len(self.data)):
                dst[j].append(self.data[j][i])
        else:
            for j in range(len(self.data)):
                dst.append([self.data[j][i]])

    def load_data(self, attrib, labels, knn=False, representacion=0.98):
        self.data = attrib.tolist()
        self.labels = labels.tolist()
        self.categorias = []

        if (knn == False):
            self.labels.sort(key=lambda x: x[1])
            # Separacion de datos en categorias
            cat = self.labels[0][1]
            cat_i = 0
            data = [[]]
            labels = [[]]
            self.categorias.append(cat)
            for i in range(len(self.labels)):
                if self.labels[i][1] == cat:
                    index = int(self.labels[i][0])
                    data[cat_i].append(self.data[index])
                    #self.show_face(self.data[index])
                    labels[cat_i].append([index, self.labels[i][1]])
                    
                else:
                    cat = self.labels[i][1]
                    cat_i += 1
                    data.append([])
                    labels.append([])
                    index = int(self.labels[i][0])
                    data[cat_i].append(self.data[index])
                    labels[cat_i].append([index, self.labels[i][1]])
                    self.categorias.append(cat)
            
            self.data = data.copy()
            self.labels = labels.copy()
        else:
            self.data = [self.data]
            self.labels = [self.labels]
            # print(np.array(self.data[0]).T.shape)
            # exit()
            self.categorias.append("unica")
            
            
        # Calculo de las caras medias
        self.mean_face = []
        for i in self.data:
            m_f = self.calc_mean_face(i)
            #self.show_face(m_f)
            self.mean_face.append(m_f)
        
        # Calculo de las matrices de covarianza
        self.covariance = []
        self.normalized = []
        for i in range(len(self.categorias)):
            C, A = self.calc_covariance(self.data[i], self.mean_face[i])
            self.covariance.append(C)
            self.normalized.append(A)
            
        # Calculo de eigenvalores y eigen vectores a partir de matriz de covarianza
        self.eigenvalues = []
        self.eigenvectors = []
        for i in range(len(self.categorias)):
            values, vectors = self.calc_eigen(self.covariance[i], i, representacion)
            self.eigenvalues.append(values)
            self.eigenvectors.append(vectors)
            
       
        # Calculo de las eigen faces y los w
        self.e_faces = []
        for i in range(len(self.categorias)):
            u_t = np.array(self.eigenvectors[i])
            f = np.array(self.normalized[i])
            tmp = np.matmul(f, u_t.T)
            tmp = self.normalize(tmp)
            self.e_faces.append(tmp)
            
            #----------------------------------------------------
            # print(u_t.shape, f.shape)
            # tmp = tmp.T
            # print(tmp.shape)
            # for x in range(len(tmp)):
            #     self.show_face(self.data[i][x])
            #     self.show_face(tmp[x])
            #     b = input("Presione tecla para continuar . . .")
            #---------------------------------------------------
        #print(np.array(self.e_faces).shape)
        
        # Saca la matriz de pesos del conjunto de entrenamiento para knn
        if (knn):
            tmp = []
            for i in range(len(self.categorias)):
                for j in self.data[i]:
                    normalizado = np.array([j]) - self.mean_face[i]
                    w = np.matmul(self.e_faces[i].T, normalizado.T)
                    if len(tmp) == 0:
                        tmp = w
                    else:
                        tmp = np.hstack((tmp,w))
                
            self.w = tmp.tolist()
            self.labels = self.labels[0]
            # print(np.array(self.w).shape, np.array(self.labels).shape)
            # exit()
            
        # for i in self.mean_face:
        #     self.show_face(i)

    def calc_mean_face(self, data):
        mean_face = np.zeros((1,np.array(data).shape[1]))
        total = len(data)
        for i in data:
            mean_face = np.add(mean_face, i)
        mean_face = mean_face/total
        return mean_face

    def calc_covariance(self, data, mean_face):
        # A
        A = []
        A_t = []
        for i in range(0, len(data)):
            tmp = np.array(data[i])
            result = tmp - mean_face
            #self.show_face(result)
            A.append(result[0])
        A_t = np.array(A)
        #print(A_t.shape)
        #exit()
        A = np.array(A).transpose()
        
        C = np.matmul(A_t, A)
        # print(C.shape)
        return C, A

    def calc_eigen(self, cov, index, representacion):
        values, vectors = np.linalg.eig(np.array(cov))
        values, vectors = values.tolist(), vectors.tolist()
        valores = values.copy()
        valores.sort(reverse=True)
        
        # Calculo del porcentaje de varianza de cada componente
        len_orig = len(valores)
        print("representacion:", representacion)
        # Proporcion de la varianza respecto a otras componentes
        var_proporcion = np.cumsum(valores)/sum(valores)
        num_componentes = 0  # Valor usado para seleccion de componenetes
        for j in var_proporcion:
            if (j>representacion): # Porcentaje de representacion
                num_componentes = var_proporcion.tolist().index(j)+1
                break
            
        num_comp = range(1,len(valores)+1)
        #plt.scatter(num_comp, var_proporcion)
        #plt.show()
        
        print("original: ", len_orig," componentes: ", num_componentes)
            
        vectores = []
        etiquetas = []
        #labels = self.labels[index]
        for i in range(0, num_componentes):
            indice = values.index(valores[i])
            vectores.append(vectors[indice])
            # Reordenamiento de etiquetas
        return valores, vectores

    def show_face(self, data):
        data = np.array(data)
        data = np.reshape(data, (imagenes.SHAPE[1],imagenes.SHAPE[0]))
        #data = data / np.linalg.norm(data)
        data = self.normalize2(data,255)
        #print(data)
        img = Image.fromarray(data.astype(np.uint8))
        img.show()
    
    def normalize2(self, data, max_range = 1):
        data_n = ((data - np.min(data)) / (((np.max(data)) - np.min(data))+0.001))* max_range
        return data_n
    
    def normalize(self, data, max_range = 1):
        #data_n = ((data - np.min(data)) / (((np.max(data)) - np.min(data))+0.001))* max_range
        norm = np.linalg.norm(data)
        data_n = data / norm
        return data_n
    
    def calc_pesos_knn(self, face):
        face = np.array([face])
        #self.show_face(face)
        mag_min = []
        for i in range(len(self.categorias)):
            normalizada = face-self.mean_face[i]
            #self.show_face(normalizada)
            #print(self.e_faces[i].shape)
            #print(self.e_faces[i].T.shape, np.array(normalizada.T).shape)
            w = np.matmul(self.e_faces[i].T, normalizada.T)
            #w = self.normalize(w.T)
            w = w.T
            #exit()
            #reconstruccion = np.matmul(self.e_faces[i].transpose(), w).transpose()
            #self.show_face(reconstruccion)
            #exit()
        # print(w.transpose().tolist()[0], len(w.transpose().tolist()[0]))
        # exit()
        return w.tolist()[0]

    def recognize(self, face):
        face = np.array([face])
        #self.show_face(face)
        mag_min = []
        for i in range(len(self.categorias)):
            normalizada = face-self.mean_face[i]
            w = np.matmul(self.e_faces[i].T, normalizada.transpose())
            #w = self.normalize(w)
            #print(self.e_faces[i].shape, w.shape)
            reconstruccion = np.matmul(self.e_faces[i], w).transpose() + self.mean_face[i]
            #print(reconstruccion.shape)
            #self.show_face(reconstruccion)
            
            diferencia = self.normalize(reconstruccion)-face
            #print(reconstruccion.shape, face.shape)
            magnitud = np.linalg.norm(diferencia)
            #print(magnitud, self.categorias[i])
            mag_min.append(magnitud)
            #exit()
        #print(min(mag_min))
        #print(self.categorias[mag_min.index(min(mag_min))])
        #exit()
        return [self.categorias[mag_min.index(min(mag_min))],min(mag_min)]
        