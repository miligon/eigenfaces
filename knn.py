#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:01:50 2022

@author: miguel
"""
 # 
 #  KNN 
 #
 #  EJEMPLO DE USO:
 #    
 #    clasificador = KNN()
 #    clasificador.load_data(train_attrib, train_labels)
 #    res = clasificador.clasificar(atrib, k)
 
 #  PARAMETROS:
                
 #  Arreglo de atributos ([[float]])
             
 #  train_attrib = [[Atributo(1 ,1);   Atributo(1 ,2);   Atributo(1 ,..);   Atributo(1 ,M)],
 #                  [Atributo(2 ,1);   Atributo(2 ,2);   Atributo(2 ,..);   Atributo(2 ,M)],
 #                  [Atributo(... ,1); Atributo(... ,2); Atributo(... ,..); Atributo(... ,M)],
 #                  [Atributo(N,1);    Atributo(N,2);    Atributo(N ,..);   Atributo(N,M)]]
 #  Donde los indices del arreglo corresponden a (N -> # de Atributo, M -> indice de ejemplo)
 
 #  Arreglo de etiquetas
 #                   cadena de texto 
 #                   con numero entero        cadena de texto
 #                         ||                       ||
 #                         \/                       \/
 #
 #  train_labels = [[indice de ejemplo1,   etiqueta del ejemplo1]
 #                  [indice de ejemplo2,   etiqueta del ejemplo2],
 #                  [indice de ejemplo..., etiqueta del ejemplo...]
 #                  [indice de ejemploN,   etiqueta del ejemploN]]
 #
 #  Vector del ejemplo a clasificar
 
 #  atrib = [A1,A2,..,AN]
 #  k = numero entero mayor a 0
 
 #  SALIDA:
     
 #  [ETIQUETA, WEIGHT]
     
import numpy as np
import math

class KNN:
    def __init__(self):
        self.data = []
        
    def load_data(self, attrib,labels):
        self.data = attrib
        self.labels = labels
        
    def get_distance(self,origen,destino):
        a = np.array(origen)
        b = np.array(destino)
        c = b-a
        c = np.power(c, 2)
        total = np.sum(c)
        return math.sqrt(total)
    
    def extract_attributes(self,index):
        res = []
        for a in self.data:
            res.append(a[index])
        return res
    
    def calc_peso(self, distance):
        # En caso de que el test y el punto a comparar sean identicos
        return math.exp((-1*math.pow(distance,2))/(2*math.pow(self.sigma,2)))
            
    def get_vecinos(self,t_data, num):
        test = np.array(t_data)
        distancias = []
        for i in range(len(self.data[0])):
            label = self.labels[i][1]
            distancia = self.get_distance(test, self.extract_attributes(i))
            if distancia == 0:
                return [[distancia, label]]
            distancias.append([distancia, label])
        # orden ascendente
        distancias.sort(reverse=False, key=lambda dist: dist[1])
        self.dmin = distancias[0][0]
        self.sigma = self.dmin
        selected = [[]]
        lbl = distancias[0][1]
        cont = 0
        for x in distancias:
             if (lbl == x[1]):
                 selected[cont].append(x)
             else:
                 cont += 1
                 selected.append([])
                 selected[cont].append(x)
                 lbl = x[1]
        
        selected[0].sort(reverse=False, key=lambda val: val[0])
        selected[1].sort(reverse=False, key=lambda val: val[0])
        #print(selected[0][:num])
        #print(selected[1][:num])
        dist = selected[0][:num]
        for x in selected[1][:num]:
            dist.append(x)
        dist.sort(reverse=False, key=lambda val: val[0])
        #print(dist)
        #exit()
        return dist
    
    def simple_knn(self, vecinos):
        conteo_vecinos = []
        for valores in vecinos:
            dist = valores[0]
            conteo_vecinos.append(valores[1])
        voto = max(conteo_vecinos, key=conteo_vecinos.count)
        return voto
    
    def weighted_regresion(self, vecinos):
        vecinos.sort(key=lambda label: label[1])
        votos = np.array([])
        labels = np.array([])
        weights = np.array([])
        for vecino in vecinos:
            labels = np.append(labels, float(vecino[1]))
            distancia = vecino[0]
            weights = np.append(weights, float(self.calc_peso(distancia)))

        s_weights = weights.sum()
        WxV = labels * weights
        
        p = WxV/s_weights
        
        Z = p.sum()
        
        return Z
    
    def weighted_predict(self, vecinos):
        vecinos.sort(key=lambda label: label[1])
        votos = []
        labels = []
        for vecino in vecinos:
            label = vecino[1]
            distancia = vecino[0]
            weight = self.calc_peso(distancia)
            if label not in labels:
                votos.append(weight)
                labels.append(label)
            else:
                index = labels.index(label)
                votos[index] = votos[index] + weight       
        #print(labels, votos)
        v_mayor = 0.0
        l_mayor = ""
        for i in range(len(labels)):
            if (votos[i] > v_mayor):
                v_mayor = votos[i]
                l_mayor = labels[i]
        
        return [l_mayor, v_mayor]
            
    
    def clasificar(self,atrib, k):
        vecinos = self.get_vecinos(atrib, k)
        if (len(vecinos) == 1):
            print("(muestra==test, d=0): ", vecinos[0][1])
            return [vecinos[0][1],100]
        #print(vecinos)
        #exit()
        res = self.weighted_predict(vecinos)
        #print("Predicción: ", res)
        #exit()
        return res
    
    def regresion(self,atrib, k):
        vecinos = self.get_vecinos(atrib, k)
        if (len(vecinos) == 1):
            print("(muestra==test, d=0): ", vecinos[0][1])
            return [vecinos[0][1],100]
        #print(vecinos)
        #res = self.weighted_regresion(vecinos)
        res = self.simple_knn(vecinos)
        #print("Predicción: ", res)
        return res
        
        