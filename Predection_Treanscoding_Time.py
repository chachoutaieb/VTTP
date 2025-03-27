#creat by Chachou Taieb (^_^)
import os
import argparse
import pandas as pd
import xgboost as xgb
import lightgbm as ltb
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tqdm import tqdm
#model for tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta


class machine_learning:

    def __init__(self):
        self.dataset = 'dataset'
 
 


#********************************************************************************************************************
#-------------------------------------------------- All ALgorithme --------------------------------------------------
#********************************************************************************************************************
    def all_method(self, path_project,  ds):
        # 'Resolution', 'Height', 'Frame_rate','Codec',
        sh = 0
        if "SVD_NVD" in ds:
            csv_dataset = path_project+'/dataset/TTP2021_SVD_NVD_CPU_GPU.csv'
            
            FastTTPS = ['TR_Video', 'Video', 'Segment', 'TR_Bitrate', 'Duration', 'TR_Width', 
            'TR_Height', 'TR_Preset',  'Framerate', 'size_full144p', 'rtime_full144p', 'TR_Codec', 'VM_Type', 'TR_Rtime']
        
            ComplexCTTP = ['TR_Video','Video',  'Segment', 'TR_Codec', 'TR_Bitrate', 'TR_Resolution', 'Duration',   'TR_Preset',
            'Framerate', 'Complexity_class', 'VM_Type', 'TR_Rtime']
        
            TTP2015 = ['TR_Video', 'Duration', 'TR_Codec', 'Height',  'Width', 'Framerate', 'Nb_frame', 'Bitrate', 
            'Codec',  'TR_Bitrate', 'TR_Height', 'TR_Width', 'TR_Framerate', 'Size_B', 'Size_P', 
            'Size_I','Size_kb',  'Numbre_B', 'Numbre_P', 'Numbre_I', 'VM_Type', 'TR_Rtime']
        
            path_nbf = path_project+'/dataset/FeatresImportance_SVD_NVD.csv'
            nbf = pd.read_csv(path_nbf, error_bad_lines=False, sep = ',')
            TTP2021 = nbf["key"].to_list()
            TTP2021 = ['TR_Video',  'Video', 'Segment']+TTP2021[0:22]+['TR_Rtime']

            ti  = 'TR_Rtime'

            dataset = pd.read_csv(csv_dataset, error_bad_lines=False, sep = ',')
            dataset = shuffle(dataset)
            dataset = shuffle(dataset)
            X = dataset
            idy =  dataset.columns.get_loc(ti)
            y = dataset.iloc[:,idy]     
            X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
            
            print('Start evalution on SVD_NVD dataset')
            y_preds_list = []
            
            y_preds_list.append(self.TTP2015(path_project, TTP2015, X_train_all, X_test_all, y_train, y_test, "SVD_NVD" ))
            print('TTP2015 100%')
            y_preds_list.append(self.ComplexCTTP(path_project, ComplexCTTP, X_train_all, X_test_all, y_train, y_test, "SVD_NVD"))
            print('ComplexCTTP 100%')
            y_preds_list.append(self.FastTTPS(path_project, FastTTPS, X_train_all, X_test_all, y_train, y_test, "SVD_NVD" ))
            print('FastTTPS 100%')
            y_preds_list.append(self.TTP2021(path_project, TTP2021, X_train_all, X_test_all, y_train, y_test, "SVD_NVD", "SVD_NVD"))
            print('Our method 100%')

            l = ['Tewodros et al.',  'ComplexCTTP', 'FastTTPS', 'Our method' ]
            if ds == 'Zabrovskiy et al.' :
                self.figure_ypred(path_project, y_test, y_preds_list, l, "SVD_NVD", 100)

#-------------------------------------------------- DS_Zabrovskiy_al-----
        if 'Zabrovskiy et al.' in ds:
            
            csv_dataset = path_project+'/dataset/DS_Zabrovskiy_al.csv'
                
            FastTTPS = ['TR_Video','Video',  'Segment', 'TR_Bitrate', 'Duration', 'TR_Width',
        'TR_Height', 'TR_Preset',  'Framerate', 'Full_144pixel_Rtime', 'Full_144pixel_size', 'TR_Codec','VM_Type', 'encoding_time']
        
            ComplexCTTP = ['TR_Video','Video',  'Segment', 'TR_Codec', 'TR_Bitrate', 'TR_Resolution', 'Duration',   'TR_Preset',
        'Framerate', 'Complexity_class', 'VM_Type', 'encoding_time']
        
            TTP2015 = ['TR_Video', 'Duration', 'TR_Codec',  'TR_Width', 'TR_Height', 'TR_Bitrate',  'TR_Framerate', 'Numbre_I', 'Numbre_P', 'Numbre_B',  'Nb_frame',  
        'Size_I', 'Size_P', 'Size_B', 'Size_kb',   'Codec',  'Bitrate', 'Framerate', 'Width',  'Height', 
        'VM_Type', 'encoding_time']
        
            path_nbf = path_project+"/dataset/FeatresImportance_SVD_NVD.csv"
            nbf = pd.read_csv(path_nbf, error_bad_lines=False, sep = ',')
            TTP2021 = nbf["key"].to_list()
            TTP2021 = ['TR_Video',  'Video', 'Segment']+TTP2021[0:22]+['encoding_time']
            
            ti  = 'encoding_time'

            dataset = pd.read_csv(csv_dataset, error_bad_lines=False, sep = ',')
            dataset = shuffle(dataset)
            dataset = shuffle(dataset)
            X = dataset
            idy =  dataset.columns.get_loc(ti)
            y = dataset.iloc[:,idy]     
            X_train_all, X_test_allZabr, y_train, y_testZabr = train_test_split(X, y, test_size=0.2, random_state=123) 
            
            print('Start evalution on Zabrovskiy et al. dataset')
            y_preds_list1 = []
            
            y_preds_list1.append(self.TTP2015(path_project, TTP2015, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy_DS.'))
            print('TTP2015 100%')
            y_preds_list1.append(self.ComplexCTTP(path_project, ComplexCTTP, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy_DS' ))
            print('ComplexCTTP 100%')
            y_preds_list1.append(self.FastTTPS(path_project, FastTTPS, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy_DS.' ))
            print('FastTTPS 100%')
            y_preds_list1.append(self.TTP2021(path_project, TTP2021, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy et al.', 'Zabrovskiy_DS'))
            print('Our method 100%')
            l = ['Tewodros et al.', 'ComplexCTTP', 'FastTTPS', 'Our method' ]
        
        
        if "SVD_NVD" in ds:     
            sh = sh + 1
            self.showResult(y_preds_list, X_test_all, y_test, "SVD_NVD", sh)
        if 'Zabrovskiy et al.' in ds:
            sh = sh + 1
            self.showResult(y_preds_list1, X_test_allZabr, y_testZabr, 'Zabrovskiy et al.', sh)
        
            
#********************************************************************************************************************
#-------------------------------------------------- Figures Ypred---------------------------------------------------------S
#********************************************************************************************************************

    def figure_ypred(self, path_project, y_test, y_preds, method, ds, yt):
        y_test = list(y_test)
        zise = int(len(y_test)/5)
        plt.figure()
        fig, ax = plt.subplots(figsize=(9,5))
        line = ['-', '-', '-', '-', '--']
        c = [(242/255, 29/255, 232/255), (19/255, 177/255, 189/255), (230/255, 94/255, 76/255), (16/255, 115/255, 158/255), (242/255, 147/255, 30/255)]
        ax.set_title('True transcoding time vs Predicted transcoding time')
        y_test1 = []
        for i, y_pred in enumerate(y_preds):
            y_test1 = []
            y_pred1 = []
            for j, y in enumerate(y_test):
                if y <= yt:
                    y_test1.append(y_test[j])
                    y_pred1.append(y_pred[j])

            x_ax = range(len(y_test1[0:50]))  
            y_pred1 = y_pred1[0:50]
            ax.plot(x_ax, y_pred1,linestyle= line[i], color=c[i], linewidth=1.3,  label= "[R² = %.3f] " % r2_score(y_test, y_preds[i])+ method[i])
        
        ax.plot(x_ax, y_test1[0:50], linestyle=line[-1], color=c[-1], linewidth=1.3, label="True transcoding time")
        ax.set_ylabel('Transcoding time (s)', fontsize=11)
        ax.set_xlabel('ID video segment', fontsize=11)
        labels = np.arange(0, 51, 5)
        labels = np.arange(0, 15, 5)
        print(labels)
        ax.set_xticks(labels)
        ax.legend()
        fig.savefig(path_project+'/figures/PTTvsOTT_XGBoost_'+ds+'.pdf',  bbox_inches='tight')
        

#********************************************************************************************************************
#-------------------------------------------------- show results-----------------------------------------------------
#********************************************************************************************************************

    def showResult(self, y_preds_list, X_test_all, y_test, ds,sh):
        
        header1 = "|{:^33}|{:^83}|".format("Dataset", ds)
        header2 = "|{:^33}|{:^27}|{:^27}|{:^27}|".format("Metric", "R2", "RMSE", "MAE")
        header3 = "|{:^10}|{:^22}|{:^9}{:^9}{:^9}|{:^9}{:^9}{:^9}|{:^9}{:^9}{:^9}|".format("Platform", "Algorithm", "ALL", "H.264", "H.265", "ALL", "H.264", "H.265", "ALL", "H.264", "H.265")
        line = "+"
        for i in range(len(header3)-2):
            line = line+"-"
        line = line+"+"
        line0 = list(line)
        line0[34] = "+"
        line1 = list(line)
        for i in [34, 62, 90]:
            line1[i] = "+"
        line2 = list(line)
        for i in [11, 34, 62, 90]:
            line2[i] = "+"
        
        line2midl = list(line)
        for i in range(1,11):
            line2midl[i] = " "
        for i in [11, 34, 62, 90]:
            line2midl[i] = "+"
        if sh == 1:
            print("".join(line0))
            print(header1)
            print("".join(line1))
            print(header2)
            print("".join(line2))
            print(header3)
            print("".join(line2))
        elif sh == 2:
            print(header1)
            print("".join(line1))

        l = ['Tewodros et al.', 'ComplexCTTP', 'FastTTPS',  'Our method' ]
        
        if ds == 'Zabrovskiy et al.':  
            for i, y_preds in enumerate(y_preds_list):
                codec = list(X_test_all['TR_Codec'])
                y_tesyh64 = []
                y_tesyh65 = []
                y_predsh64 = []
                y_predsh65 = []
                for k, c in enumerate(codec):
                    if c == 0:
                        y_tesyh64.append(y_test.iloc[k])
                        y_predsh64.append(y_preds[k])
                    elif c == 1:
                        y_tesyh65.append(y_test.iloc[k])
                        y_predsh65.append(y_preds[k])
        
                mse = mean_squared_error(y_test, y_preds)
                rmse = np.sqrt(mse) #evaluate the aucuracy of modele by compute the mean squared error
                mae = mean_absolute_error(y_test, y_preds)
                R2 = r2_score(y_test, y_preds)
                
                mseh4 = mean_squared_error(y_tesyh64, y_predsh64)
                rmseh4 = np.sqrt(mseh4) #evaluate the aucuracy of modele by compute the mean squared error
                maeh4 = mean_absolute_error(y_tesyh64, y_predsh64)
                R2h4 = r2_score(y_tesyh64, y_predsh64)

                mseh5 = mean_squared_error(y_tesyh65, y_predsh65)
                rmseh5 = np.sqrt(mseh5) #evaluate the aucuracy of modele by compute the mean squared error
                maeh5 = mean_absolute_error(y_tesyh65, y_predsh65)
                R2h5 = r2_score(y_tesyh65, y_predsh65)
                core = "|{:^10}|{:^22}|{:^9.4f}{:^9.4}{:^9.4f}|{:^9.4f}{:^9.4f}{:^9.4f}|{:^9.4f}{:^9.4f}{:^9.4f}|"
                if i == 1:
                    core = core.format("CPU", l[i], R2, R2h4, R2h5, rmse, rmseh4, rmseh5, mae, maeh4, maeh5)
                else :
                    core = core.format("", l[i], R2, R2h4, R2h5, rmse, rmseh4, rmseh5, mae, maeh4, maeh5)
                print(core)
                if  i == 3:
                    print("".join(line2))
                else:
                    print("".join(line2midl))

        elif ds == 'SVD_NVD' :
            allinfo = []
            cpuinfo = []
            gpuinfo = []
            for i, y_preds in enumerate(y_preds_list):
                cle = ["all", "H.264", "H.265", "CPU", "CPU_H.264", "CPU_H.265", "GPU", "GPU_H.264", "GPU_H.265"]
                codec = list(X_test_all['TR_Codec'])
                y_test = list(y_test)
                #print(codec[1], y_test[1], y_preds[1], y_preds[1][0])
                info = []
                for k, c in enumerate(codec):
                    if i == 3:
                        info.append([codec[k], y_test[k], y_preds[k]])
                    else:
                        info.append([codec[k], y_test[k], y_preds[k][0]])
                info  = np.array(info)
                infoh64 = []
                infoh65 = []
                for k, c in enumerate(codec):
                    if c == 0:
                        if i == 3:
                            infoh64.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infoh64.append([codec[k], y_test[k], y_preds[k][0]])
                    elif c == 1:
                        if i == 3:
                            infoh65.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infoh65.append([codec[k], y_test[k], y_preds[k][0]])
                infoh64  = np.array(infoh64)
                infoh65  = np.array(infoh65)
                infocpu = []
                infogpu = []
                vmtype = list(X_test_all['VM_Type'])
                #print("vmtype : ", vmtype)
                for k ,v in enumerate(vmtype):
                    if v == 1:
                        if i == 3:
                            infocpu.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infocpu.append([codec[k], y_test[k], y_preds[k][0]])
                    else:
                        if i == 3:
                            infogpu.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infogpu.append([codec[k], y_test[k], y_preds[k][0]])
                infocpu = np.array(infocpu)
                codecpu = infocpu[:,0]
                info64cpu = []
                info65cpu = []
                for k, c in enumerate(codecpu):
                    if c == 0:
                        info64cpu.append(list(infocpu[k,:]))
                    elif c == 1:
                        info65cpu.append(list(infocpu[k,:]))
                infogpu = np.array(infogpu)
                codegpu = infogpu[:, 0]
                info64gpu = []
                info65gpu = []
                for k, c in enumerate(codegpu):
                    if c == 0:
                        info64gpu.append(list(infogpu[k,:]))
                    elif c == 1:
                        info65gpu.append(list(infogpu[k,:]))
                pas = 4

                allinfo.append([info, infoh64, infoh65])
                cpuinfo.append([infocpu, np.array(info64cpu), np.array(info65cpu)])
                gpuinfo.append([infogpu, np.array(info64gpu), np.array(info65gpu)])
            allcpugpu = [cpuinfo, gpuinfo, allinfo]
            vml = ["CPU", "GPU", "CPU+GPU"]
            for ip, ivm in enumerate(allcpugpu):
                vmt = vml[ip]
                for il, infoplat in enumerate(ivm):
                    rmse = []
                    mae = []
                    R2 = []
                    for ic, codeinfo in enumerate(infoplat):
                        #print(cle[k], ":", len(inf[1]))
                        mse = mean_squared_error(codeinfo[:, 1], codeinfo[:, 2])
                        rmse.append(np.sqrt(mse)) #evaluate the aucuracy of modele by compute the mean squared error
                        mae.append(mean_absolute_error(codeinfo[:, 1], codeinfo[:, 2]))
                        R2.append(r2_score(codeinfo[:, 1], codeinfo[:, 2]))
                    core = "|{:^10}|{:^22}|{:^9.4f}{:^9.4}{:^9.4f}|{:^9.4f}{:^9.4f}{:^9.4f}|{:^9.4f}{:^9.4f}{:^9.4f}|"
                    if il == 1:
                        core = core.format(vmt, l[il], R2[0], R2[1], R2[2], rmse[0], rmse[1], rmse[2], mae[0], mae[1], mae[2])
                    else :
                        core = core.format("", l[il],  R2[0], R2[1], R2[2], rmse[0], rmse[1], rmse[2], mae[0], mae[1], mae[2])
                    print(core)
                    if  il == 3:
                        print("".join(line2))
                    else:
                        print("".join(line2midl))

#********************************************************************************************************************
#-------------------------------------------------- FastTTPS---------------------------------------------------------S
#********************************************************************************************************************

    
    def FastTTPS(self, path_project, FastTTPS, X_train_all, X_test_all, y_train, y_test, tds):
        
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()

        for i, key in enumerate(FastTTPS):
            X_train.insert(i, key, list(X_train_all[key]), True)
            X_test.insert(i, key, list(X_test_all[key]), True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dim = X_train.shape
        x = pd.concat([X_train, X_test])  
        x = scaler.fit_transform(x.iloc[:,3:-1])
        X_train_scaled = x[: dim[0], :-2]
        X_test_scaled = x[dim[0] :, :-2]
        X_testcsv = x[dim[0] :, :]
        #make able to run code in GPU and identifier the memory
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        model = tf.keras.models.Sequential([
                Dense(8, activation = 'relu', input_dim = 8, kernel_initializer='he_uniform'),
                Dense(units = 64, activation = 'relu'),
                Dense(units = 32, activation = 'relu'),
                Dense(units = 64, activation = 'relu'),
                Dense(1, activation='linear')])
        model.compile(loss='mean_absolute_error', optimizer= Adadelta(learning_rate=0.1), metrics=['mae', 'mse'])     
        model.fit(X_train_scaled, y_train, epochs=500, batch_size=2000, verbose=0, validation_data = (X_test_scaled, y_test))
        y_preds = model.predict(X_test_scaled) # testing the modele
        columns = ['TR_Bitrate', 'Duration', 'TR_Width',
  'TR_Height', 'TR_Preset',  'Framerate', 'Full_144pixel_Rtime', 'Full_144pixel_size', 'TR_Codec','VM_Type']
        df = pd.DataFrame(data= X_testcsv, columns=columns)
        idf = df.columns.get_loc('VM_Type')+1
        df.insert(idf, 'Y_trus' , list(y_test), True) 
        df.insert(idf, 'Y_predit' , y_preds, True) 
        df.to_excel(path_project+'/results/FastTTPS_'+tds+'_ypred.xlsx', index = False)
        return  y_preds


#********************************************************************************************************************
#---------------------------------------------- ComplexCTTP ---------------------------------------------------------
#********************************************************************************************************************            
    def ComplexCTTP(self, path_project, ComplexCTTP, X_train_all, X_test_all, y_train, y_test, tds):
        
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()

        for i, key in enumerate(ComplexCTTP):
            X_train.insert(i, key, list(X_train_all[key]), True)
            X_test.insert(i, key, list(X_test_all[key]), True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dim = X_train.shape
        x = pd.concat([X_train, X_test])  
        x = scaler.fit_transform(x.iloc[:,3:-1])
        X_train_scaled = x[: dim[0], :-1]
        X_test_scaled = x[dim[0] :, :-1]
        X_testcsv = x[dim[0] :, :]
        
        #make able to run code in GPU and identifier the memory
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        model = tf.keras.models.Sequential([
                Dense(7, activation = 'relu', input_dim = 7, kernel_initializer='he_uniform'),
                Dense(units = 64, activation = 'relu'),
                Dense(units = 32, activation = 'relu'),
                Dense(units = 64, activation = 'relu'),
                Dense(1, activation='linear')])
        model.compile(loss='mean_absolute_error', optimizer= Adadelta(learning_rate=0.1), metrics=['mae', 'mse'])     
        model.fit(X_train_scaled, y_train, epochs=500, batch_size=2000, verbose=0, validation_data = (X_test_scaled, y_test))

        y_preds = model.predict(X_test_scaled) # testing the modele
        columns = ComplexCTTP[3:-1]
        df = pd.DataFrame(data= X_testcsv, columns=columns)
        idf = df.columns.get_loc('VM_Type')+1
        df.insert(idf, 'Y_trus' , list(y_test), True) 
        df.insert(idf, 'Y_predit' , y_preds, True) 
        df.to_excel(path_project+'/results/ComplexCTTP_'+tds+'_ypred.xlsx', index = False)
        return  y_preds

                
#********************************************************************************************************************
#---------------------------------------------- TTP2015 ---------------------------------------------------------
#********************************************************************************************************************            

    def TTP2015(self, path_project, TTP2015, X_train_all, X_test_all, y_train, y_test, tds):
        
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()

        for i, key in enumerate(TTP2015):
            X_train.insert(i, key, list(X_train_all[key]), True)
            X_test.insert(i, key, list(X_test_all[key]), True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dim = X_train.shape
        x = pd.concat([X_train, X_test])  
        x = scaler.fit_transform(x.iloc[:,1:-1])
        X_train_scaled = x[: dim[0], :-1]
        X_test_scaled = x[dim[0] :, :-1]
        X_testcsv = x[dim[0] :, :]
        
        #make able to run code in GPU and identifier the memory
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True) 
        model = tf.keras.models.Sequential([
                Dense(19, activation = 'relu', input_dim = 19, kernel_initializer='he_uniform'),
                Dense(units = 64, activation = 'relu'),
                Dense(units = 32, activation = 'relu'),
                Dense(units = 64, activation = 'relu'),
                Dense(1, activation='linear')])
        model.compile(loss='mean_absolute_error', optimizer= Adadelta(learning_rate=0.1), metrics=['mae', 'mse'])     
        model.fit(X_train_scaled, y_train, epochs=500, batch_size=2000, verbose=0, validation_data = (X_test_scaled, y_test))

        y_preds = model.predict(X_test_scaled) # testing the modele
        columns = TTP2015[1:-1]
        df = pd.DataFrame(data= X_testcsv, columns=columns)
        idf = df.columns.get_loc('VM_Type')+1
        df.insert(idf, 'Y_trus' , list(y_test), True) 
        df.insert(idf, 'Y_predit' , y_preds, True) 
        df.to_excel(path_project+'/results/TTP2015_'+tds+'_ypred.xlsx', index = False)
        return  y_preds           
                

#********************************************************************************************************************
#---------------------------------------------- TTP2021 ---------------------------------------------------------
#********************************************************************************************************************            

    def TTP2021(self, path_project, TTP2021, X_train_all, X_test_all, y_train, y_test, ds, tds):
        X_trainfinal, X_testfinal, y_trainfinal, y_testfinal, tri  = self.train_test_split1(TTP2021, X_train_all, X_test_all, y_train, y_test, ds)
        y_predsfinal = []
        df = []
        for k in range(len(X_trainfinal)):
            scaler = MinMaxScaler(feature_range=(0, 1))
            dim = X_trainfinal[k].shape
            x = X_trainfinal[k].append(X_testfinal[k], ignore_index=True)
            x = scaler.fit_transform(x.iloc[:, 3:-1])
            X_train_scaled = x[: dim[0], :]
            X_test_scaled = x[dim[0] :, :]
            X_testcsv = x[dim[0] :, :]
            #make able to run code in GPU and identifier the memory
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True) 
            xg_reg = xgb.XGBRegressor(objective ='reg:tweedie', colsample_bytree = 0.97, learning_rate = 0.07, 
                                    subsample = 1, max_depth = 5, reg_alpha = 0.1, n_estimators = 1000, tree_method='gpu_hist')
            eval_set1 = [(X_train_scaled, y_trainfinal[k]), (X_test_scaled, y_testfinal[k])]
            xg_reg.fit(X_train_scaled, y_trainfinal[k], eval_metric=["rmse"], eval_set=eval_set1, verbose=False)
            y_preds = xg_reg.predict(X_test_scaled) # testing the modele
            y_predsfinal.append(y_preds)
            columns = TTP2021[3:-1]
            df.append(pd.DataFrame(data= X_testcsv, columns=columns))
        trifinal = []
        y_predsfi = []
        y_testfin = []
        y_predsfin = []
        if len(df) == 2:
            bigdata = df[0].append(df[1], ignore_index=True)
        else:
            bigdata = df[0]
        for i in range(len(tri)):
            idf = bigdata.columns.get_loc('VM_Type')+1
            y_testfin = y_testfin + list(y_testfinal[i])
            y_predsfin = y_predsfin + list(y_predsfinal[i])
            trifinal = trifinal+list(tri[i])
            y_predsfi = y_predsfi+list(y_predsfinal[i])
        bigdata.insert(idf, 'Y_trus' , y_testfin, True) 
        bigdata.insert(idf, 'Y_predit' , y_predsfin, True) 
        bigdata.to_excel(path_project+'/results/TTP2021_'+tds+'_ypred.xlsx', index = False)
        zpred = zip(y_predsfi, trifinal)
        zpred = sorted(list(zpred), key = lambda x: x[1], reverse=False)
        ypredf, tri1 = zip(*zpred)
        return  list(ypredf)
       

#***************************************************************************************************************************************************
   
    
    def train_test_split1(self, TTP2021, X_train_all, X_test_all, y_train, y_test, ds):
        if ds == 'Zabrovskiy et al.':
            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            for i, key in enumerate(TTP2021):
                X_train.insert(i, key, list(X_train_all[key]), True)
                X_test.insert(i, key, list(X_test_all[key]), True)
            return [[X_train], [X_test], [y_train], [y_test], [list(range(len(y_test)))]]
        else:
            l = range(len(y_train))
            y_test1 = list(zip(list(y_test), l))
            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            
            for i, key in enumerate(TTP2021):
                X_train.insert(i, key, list(X_train_all[key]), True)
                X_test.insert(i, key, list(X_test_all[key]), True)
            gpuTR = []
            cpuTR = []
            gpuYTR = []
            cpuYTR = []
            tri = []
            X_trainfinal = []
            X_testfinal = []
            y_trainfinal = []
            y_testfinal = []
            vmtypeTrain = list(X_train['VM_Type'])
            arrX_train = np.array(X_train)
            for i, vm in enumerate(vmtypeTrain):
                if vm == 0:
                    gpuTR.append(list(arrX_train[i,:]))
                    gpuYTR.append(y_train.iloc[i])
                elif vm == 1:
                    cpuTR.append(list(arrX_train[i,:]))
                    cpuYTR.append(y_train.iloc[i])
            arrgpuTR= np.array(gpuTR)
            X_trainfinal.append(pd.DataFrame(arrgpuTR, columns=list(X_train.keys())))
            arrcpuTR = np.array(cpuTR)
            X_trainfinal.append(pd.DataFrame(arrcpuTR, columns=list(X_train.keys())))
            y_trainfinal.append(gpuYTR)
            y_trainfinal.append(cpuYTR)
            gpuTS = []
            cpuTS = []
            gpuYTS = []
            cpuYTS = []
            vmtypeTest = list(X_test['VM_Type'])
            arrX_test = np.array(X_test)
            for i, vm in enumerate(vmtypeTest):
                if vm == 0:
                    gpuTS.append(list(arrX_test[i,:]))
                    gpuYTS.append(y_test1[i])
                elif vm == 1:
                    cpuTS.append(list(arrX_test[i,:]))
                    cpuYTS.append(y_test1[i])
            arrgpuTS= np.array(gpuTS)
            X_testfinal.append(pd.DataFrame(arrgpuTS, columns=list(X_train.keys())))
            arrcpuTS = np.array(cpuTS)
            X_testfinal.append(pd.DataFrame(arrcpuTS, columns=list(X_train.keys())))
            gYTS, t = zip(*gpuYTS)
            y_testfinal.append(gYTS)
            cYTS, t1 = zip(*cpuYTS)
            y_testfinal.append(cYTS)
            tri.append(t)
            tri.append(t1)

            return [X_trainfinal, X_testfinal, y_trainfinal, y_testfinal, tri]

   
            


#********************************************************************************************************************
#----------------------------------------------------- FIN ----------------------------------------------------------
#********************************************************************************************************************
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--operation", required=False, help="Path to the output directory")
ap.add_argument("-p", "--pathProject", required=False, help="Path to the output directory")
ap.add_argument("-ds", "--ds",  nargs='+', default=[])
args = vars(ap.parse_args())

p1 = machine_learning()
if args['operation'] == 'all_method':
    p1.all_method(args['pathProject'], args['ds'])

"""
python3 '/content/gdrive/My Drive/Chachou_Transcodage/project2/githubTTP/Predection_Treanscoding_Time.py' \
-p '/content/gdrive/My Drive/Chachou_Transcodage/project2/githubTTP' \
-o 'all_method' -ds 'SVD_NVD' 'Zabrovskiy et al.'
"""
