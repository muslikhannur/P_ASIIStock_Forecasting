# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:56:49 2019
@author: muslikhannur
"""

# ===== 1. LOAD ALL LIBRARIES NEEDED =====

from keras.models import Sequential
from keras.layers import LSTM #Create LSTM
from keras.layers import GRU #Create Scenario
from keras.layers import Dense #Create Neural
from keras.layers import Dropout #Create Regularization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from numpy import hstack
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ===== 2. LOAD DATASETS =====

dataset_Kurs = pd.read_csv('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Historical Data USD_IDR.csv', thousands = ",")
dataset_ASII = pd.read_csv('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Historical Data Stock ASII.csv', thousands = ",")
dataset_KAEF = pd.read_csv('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Historical Data Stock KAEF.csv', thousands = ",")
dataset_SMGR = pd.read_csv('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Historical Data Stock SMGR.csv', thousands = ",")

dataset_Kurs.dtypes
dataset_ASII.dtypes
dataset_KAEF.dtypes
dataset_SMGR.dtypes
 
# ===== 3. MERGE DATASETS =====

dataset_merge = [dataset_Kurs, dataset_ASII, dataset_KAEF, dataset_SMGR]
dataset_merge = reduce(lambda left,right: pd.merge(left,right,on='Date',how='left'), dataset_merge)
dataset_merge = dataset_merge.iloc[::-1] #Reverse
dataset_merge = dataset_merge.reset_index()
dataset_merge = dataset_merge.drop(['index', 'Change %_x', 'Change %_y', 'Vol.', 'Vol._x', 'Vol._y'], axis = 1) #Userless Features
dataset_merge.columns = ['Date', 
                         'Price_Kurs', 'Open_Kurs', 'High_Kurs', 'Low_Kurs',
                         'Price_ASII', 'Open_ASII', 'High_ASII', 'Low_ASII',
                         'Price_KAEF', 'Open_KAEF', 'High_KAEF', 'Low_KAEF',
                         'Price_SMGR', 'Open_SMGR', 'High_SMGR', 'Low_SMGR']

# ===== 4. FEATURE SELECTION =====

cor_mat = dataset_merge[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

dataset_merge = dataset_merge.drop(['Open_Kurs', 'High_Kurs', 'Low_Kurs', 'Open_ASII', 'High_ASII', 'Low_ASII', 'Open_KAEF', 'High_KAEF', 'Low_KAEF', 'Open_SMGR', 'High_SMGR', 'Low_SMGR'], axis = 1)
#dataset_merge.to_excel(r'C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Pre-Process.xlsx', index=False) #Posisi Masih Missing

# ===== 5. FILL MISSING VALUES =====

dataset_merge = dataset_merge.fillna(method='ffill')
dataset_merge.isnull().values.any()
dataset_merge.isnull().sum()
#dataset_merge.to_excel(r'C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Pre-Process DONE.xlsx', index=False) #Posisi Masih Missing

# ===== 6. SUMMARY FINAL MERGED DATASET =====

plt.plot(dataset_merge['Price_Kurs'], color = 'red', label = 'USD/IDR')
plt.plot(dataset_merge['Price_ASII'], color = 'green', label = 'ASII')
plt.plot(dataset_merge['Price_KAEF'], color = 'blue', label = 'KAEF')
plt.plot(dataset_merge['Price_SMGR'], color = 'purple', label = 'SMGR')
plt.title('Perbandingan Grafik Kurs, ASII, KAEF, SMGR')
plt.xlabel('Time')
plt.ylabel('Rupiah')
plt.legend()
plt.show()

# ===== 7. SPLIT TRAIN TEST =====

n_testProp = 0.3 #1st Scenario: TRAIN TEST

# Split Data Row Univariate - SKENARIO
data_Univariate = dataset_merge.loc[:,['Price_ASII']] #Last SKENARIO: Price_ASII, Price_KAEF, Price_SMGR
data_trainUni, data_testUni = train_test_split(data_Univariate, test_size=n_testProp, shuffle=False) #Train 912, Test 392
data_testUni = data_testUni.reset_index().iloc[:,1:2] #Pasti muncul Index, Select
data_testActual = data_testUni

# Split Data Row Multivariate - SKENARIO
data_Multivariate = dataset_merge.loc[:,['Price_Kurs', 'Price_ASII']] #Last SKENARIO: Price_ASII, Price_KAEF, Price_SMGR
data_trainMulti, data_testMulti = train_test_split(data_Multivariate, test_size=n_testProp, shuffle=False) #Train 912, Test 392
data_testMulti = data_testMulti.reset_index().iloc[:,1:3] #Pasti muncul Index, Select
# Split Data Column Train karena beda Scaler
data_trainMultiKurs, data_trainMultiSaham = data_trainMulti.iloc[:,0:1], data_trainMulti.iloc[:,1:2]

# ===== 8. FEATURE SCALING - TRAIN =====

# Scaling Train Univariate
sc = MinMaxScaler(feature_range = (0, 1)) #Di Scaling Range 0-1
data_trainUni = sc.fit_transform(data_trainUni)

# Scaling Train Multivariate using Different Scaler
scI = MinMaxScaler(feature_range = (0, 1))
scD = MinMaxScaler(feature_range = (0, 1))
data_trainMultiKurs = scI.fit_transform(data_trainMultiKurs)
data_trainMultiSaham = scD.fit_transform(data_trainMultiSaham)

# ===== 9. DATA STRUCTURE TRAIN =====

n_timestep = 8 #Coba coba biar gak numpuk

# Struktur Data Training Univariate
x_trainUni = []
y_trainUni = []
for i in range(n_timestep, len(data_trainUni)): #60, 912
    x_trainUni.append(data_trainUni[i-n_timestep:i, 0]) #Input: 0-59 Stock
    y_trainUni.append(data_trainUni [i, 0]) #Output: 60 Stock
x_trainUni, y_trainUni = np.array(x_trainUni), np.array(y_trainUni)
x_trainUni = np.reshape(x_trainUni, (x_trainUni.shape[0], x_trainUni.shape[1], 1))  

# Struktur Data Training Multivariate
data_trainMultiHstack = hstack((data_trainMultiKurs, data_trainMultiSaham)) #Himpunan Array, 1 Kolom per baris isi 2 Elemen
x_trainMulti = []
y_trainMulti = []
for i in range(n_timestep, len(data_trainMultiHstack)): #60, 912
    x_trainMulti.append(data_trainMultiHstack[(i-n_timestep):i]) #Input: 0-59 Kurs Stock
    y_trainMulti.append(data_trainMultiSaham[i,0]) #Output: 60 Stock
x_trainMulti, y_trainMulti = np.array(x_trainMulti), np.array(y_trainMulti) #Auto 3D
for i in range(len(x_trainMulti)): #Merge
	print(x_trainMulti[i], y_trainMulti[i]) #Cek Struktur Data

# ===== 10. DATA STRUCTURE TEST =====

# Struktur Data Test Univariate untuk diPrediksi
inputs_uni = data_Univariate[len(data_Univariate) - len(data_testUni) - n_timestep:].values 
inputs_uni = sc.transform(inputs_uni)
x_testUni = []
for i in range(n_timestep, len(inputs_uni)): #60, 452 
    x_testUni.append(inputs_uni[i-n_timestep:i, 0])
x_testUni = np.array(x_testUni)
x_testUni = np.reshape(x_testUni, (x_testUni.shape[0], x_testUni.shape[1], 1)) #392, 60, 1

# Struktur Data Test Multivariate untuk diPrediksi
inputs_multi = data_Multivariate[len(data_Multivariate) - len(data_testMulti) - n_timestep:].values
inputs_multi = pd.DataFrame(inputs_multi) #ke DF biar bisa di iloc
inputs_multi.columns = ['Price_Kurs', 'Price_Saham']
data_testMultiKurs, data_testMultiSaham = inputs_multi.iloc[:,0:1], inputs_multi.iloc[:,1:2] # Split Data Column Test karena beda Scaler
data_testMultiKurs = scI.transform(data_testMultiKurs)
data_testMultiSaham = scD.transform(data_testMultiSaham)
data_testMultiHstack = hstack((data_testMultiKurs, data_testMultiSaham))
x_testMulti = []
for i in range(n_timestep, len(inputs_multi)): #60, 452 
    x_testMulti.append(data_testMultiHstack[i-n_timestep:i, : ])
x_testMulti = np.array(x_testMulti)
x_testMulti = x_testMulti.reshape((x_testMulti.shape[0], x_testMulti.shape[1], 2)) #392, 60, 2

# ===== 11. TRAINING MODEL =====

n_DenseUnits = 1 #Not Adjustable
n_loss = 'mean_squared_error' #Not Adjustable

n_LSTMunits = [10, 30, 50] #Adjustable : 10, 30, 50
n_DropoutRate = [0.1, 0.3, 0.5] #Adjustable : 0.1, 0.3, 0.5
n_optimizer = ['Adam', 'RMSprop', 'SGD'] #Adjustable : 'Adam',', RMSprop','SGD'
n_epoch = [50, 100, 150, 200] #Adjustable : 50, 100, 150, 200
n_batchsize = [32, 64, 96] #Adjustable : 32, 64, 96

for i in range(len(n_LSTMunits)):
    for j in range(len(n_DropoutRate)):
        for k in range(len(n_optimizer)):
            for l in range(len(n_epoch)):
                for m in range(len(n_batchsize)):
                    # Build Univariate LSTM
                    modelUni = Sequential()
                    modelUni.add(LSTM(units = n_LSTMunits[i], input_shape = (x_trainUni.shape[1], x_trainUni.shape[2]))) #1 is Timestep
                    modelUni.add(Dropout(rate = n_DropoutRate[j]))
                    modelUni.add(Dense(units = n_DenseUnits))
                    modelUni.compile(optimizer = n_optimizer[k], loss = n_loss)
                    startTrain = time.perf_counter()
                    historyUni = modelUni.fit(x = x_trainUni, y = y_trainUni, epochs = n_epoch[l], batch_size = n_batchsize[m])
                    elapsedTrainUni = time.perf_counter() - startTrain
                    TTUni = '%.4f' % elapsedTrainUni
                    LastLossUni = '%.4f' % historyUni.history['loss'][-1]
                    
                    # Build Multivariate LSTM
                    modelMulti = Sequential()
                    modelMulti.add(LSTM(units = n_LSTMunits[i], input_shape = (x_trainMulti.shape[1] , x_trainMulti.shape[2]))) #1 is Timestep
                    modelMulti.add(Dropout(rate = n_DropoutRate[j]))
                    modelMulti.add(Dense(units = n_DenseUnits))
                    modelMulti.compile(optimizer = n_optimizer[k], loss = n_loss)
                    startTrain = time.perf_counter()
                    historyMulti = modelMulti.fit(x = x_trainMulti, y = y_trainMulti, epochs = n_epoch[l], batch_size = n_batchsize[m])
                    elapsedTrainMulti = time.perf_counter() - startTrain
                    TTMulti = '%.4f' % elapsedTrainMulti
                    LastLossMulti = '%.4f' % historyMulti.history['loss'][-1]
                    
                    # ===== 12. PLOT TRAINING LOSS =====
                    
                    figLoss = plt.gcf()
                    plt.plot(historyUni.history['loss'])
                    plt.plot(historyMulti.history['loss'])
                    plt.title(str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate,\n'
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size.')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.figtext(0.60, 0.70, "TT Uni: " +str(TTUni)+ "\nTT Multi: " +str(TTMulti)+ "\nLastLoss Uni: " +str(LastLossUni)+ "\nLastLoss Multi: " +str(LastLossMulti))
                    plt.legend(['Train Loss Univariate', 'Train Loss Multivariate'], loc='upper left')
                    plt.show()
                    figLoss.savefig('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Hasil/' +str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate, '
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size - Training Loss.png', dpi = 100)
                    
                    # ===== 13. TESTING MODEL =====
                    
                    startTestUni = time.perf_counter()
                    y_testUni = modelUni.predict(x_testUni)
                    elapsedTest = time.perf_counter() - startTestUni
                    data_testPredUni = sc.inverse_transform(y_testUni) #Inverse to Original Form
                    
                    startTestMulti = time.perf_counter()
                    y_testMulti = modelMulti.predict(x_testMulti) 
                    elapsedTest = time.perf_counter() - startTestMulti
                    data_testPredMulti = scD.inverse_transform(y_testMulti) # denormalization forecast result
                    
                    # Combine and Save Result
                    A_PredictionResult = np.hstack([data_testPredUni,data_testPredMulti])
                    A_PredictionResult = np.around(A_PredictionResult, decimals=2, out=None)
                    A_PredictionResult = A_PredictionResult.astype(str) #Array String - Agar ntar bisa Comma
                    
                    # ===== 14. EVALUATE THE RESULT =====
                    
                    def mean_absolute_percentage_error(y_true, y_pred): 
                        y_true, y_pred = np.array(y_true), np.array(y_pred)
                        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    MAPEUni = '%.4f' % mean_absolute_percentage_error(data_testUni, data_testPredUni) #Univariate
                    MAPEMulti = '%.4f' % mean_absolute_percentage_error(data_testActual, data_testPredMulti) #Multivariate
                    
                    # ===== 15. VISUALIZE FORECASTING RESULT =====
                    
                    figPredict = plt.gcf()
                    plt.plot(data_testActual, color = 'red', label = 'Real Stock Price', linewidth=0.75)
                    plt.plot(data_testPredUni, color = 'blue', label = 'Univariate - Predicted Stock Price', linewidth=0.75)
                    plt.plot(data_testPredMulti, color = 'orange', label = 'Multivariate - Predicted Stock Price', linewidth=0.75)
                    plt.title(str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate,\n'
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size.')
                    plt.xlabel('Period (Day)')
                    plt.ylabel('Stock Price (Rupiah)')
                    plt.figtext(0.63, 0.77, "MAPE Uni: " +str(MAPEUni)+ "\nMAPE Multi: " +str(MAPEMulti))
                    plt.legend()
                    plt.show()
                    figPredict.savefig('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Hasil/' +str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate, '
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size - Testing Result.png', dpi = 100)
                    
                    # ===== 16. RECAP RESULT =====
                    
                    # Untuk diGabungkan
                    A_EvaluationResult = [[MAPEUni, MAPEMulti], [TTUni, TTMulti], [LastLossUni, LastLossMulti]]
                    A_EvaluationResult = np.asarray(A_EvaluationResult) #Array String - Agar ntar bisa Comma
                     
                    # Print Result
                    print("\n=================================\nRECAP RESULT (MAPE, TT, LastLoss)\n=================================\n")
                    print(str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate,\n'
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size\n')
                    print("Univariate Model:")
                    print(MAPEUni+ "\n" +TTUni+ "\n" +LastLossUni+ "\n")
                    print("Multivariate Model:")
                    print(MAPEMulti+ "\n" +TTMulti + "\n" +LastLossMulti+ "\n")
                    
                    # Sesuaikan dengan Template dengan Naruh Evaluasi diatas agar tinggal Copas
                    A_AnExcelResult = np.concatenate((A_EvaluationResult, A_PredictionResult), axis=0)
                    A_AnExcelResult = pd.DataFrame(A_AnExcelResult)
                    A_AnExcelResult.columns = ['Univariate', 'Multivariate']
                    A_AnExcelResult.Univariate = A_AnExcelResult.Univariate.str.replace('.', ',') #Agar sama Formatnya kayak Excel
                    A_AnExcelResult.Multivariate = A_AnExcelResult.Multivariate.str.replace('.', ',') #Agar sama Formatnya kayak Excel
                    A_AnExcelResult.to_excel('C:/Users/muslikhannur/Documents/8 - Tugas Akhir/Code and Dataset/Hasil/' +str(n_testProp)+ ' Test Size, ' +str(n_timestep)+ ' Timestep, ' +str(n_LSTMunits[i])+ ' LSTM Units, ' +str(n_DropoutRate[j])+ ' Dropout Rate, '
                              +str(n_optimizer[k])+ ' Optimizer, ' +str(n_epoch[l])+ ' Epoch, ' +str(n_batchsize[m])+ ' Batch Size - Eval & Predict Excel.xlsx', index = False, header = False) #Header False mulai TimeStep 4