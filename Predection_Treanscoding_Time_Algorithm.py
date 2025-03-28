# Created by Chachou Taieb (^_^)
import os
import argparse
import pandas as pd
import xgboost as xgb
import lightgbm as ltb
from scikeras.wrappers import KerasRegressor  # Updated import
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from tqdm import tqdm

# Model for TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, Adadelta

# Ensure TensorFlow and NumPy compatibility
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")

class machine_learning:

    def __init__(self):
        self.dataset = 'dataset'
    
    

#********************************************************************************************************************
#-------------------------------------------------- All ALgorithme --------------------------------------------------
#********************************************************************************************************************
    def all_method(self,  path_project,  ds):
        # 'Resolution', 'Height', 'Frame_rate','Codec',
        sh = 0
        if "SVD_NVD" in ds:    
            csv_dataset = path_project+'/dataset/VTTP2025_SVD_NVD_CPU_GPU_FS.csv'
            path_nbf = path_project+'/dataset/FeatresImportance_SVD_NVD.csv'
            nbf = pd.read_csv(path_nbf, on_bad_lines='skip', sep = ',')
            VTTP2025 = nbf["key"].to_list()
            VTTP2025 = ['TR_Video',  'Video', 'Segment']+VTTP2025[0:22]+['TR_Rtime']

            ti  = 'TR_Rtime'
            dataset = pd.read_csv(csv_dataset, on_bad_lines='skip', sep = ',')
            dataset = shuffle(dataset)
            dataset = shuffle(dataset)
            X = dataset
            idy =  dataset.columns.get_loc(ti)
            y = dataset.iloc[:,idy]     
            X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
            
            print('Start evalution')
            y_preds_list = []
            
            
            y_preds_list.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_all, y_train, y_test, "SVD_NVD", 'XGB', "SVD_NVD"))
            print('XGB 100%', y_preds_list[-1][1])
            y_preds_list.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_all, y_train, y_test, "SVD_NVD", 'LGB', "SVD_NVD"))
            print('LGB 100%', y_preds_list[-1][1])
            y_preds_list.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_all, y_train, y_test, "SVD_NVD", 'ANN', "SVD_NVD"))
            print('ANN 100%', y_preds_list[-1][1])
            y_preds_list.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_all, y_train, y_test, "SVD_NVD", 'KNN', "SVD_NVD"))
            print('KNN 100%', y_preds_list[-1][1])
           

        l = ['XGBoost', 'LightGBM', 'ANN',  'KNN']

        #-------------------------------------------------- DS_Zabrovskiy_al-----
        if 'Zabrovskiy et al.' in ds:
            csv_dataset = path_project+'/dataset/DS_Zabrovskiy_al_VTTP2025.csv' 
            path_nbf = path_project+'/dataset/FeatresImportance_SVD_NVD.csv'
            nbf = pd.read_csv(path_nbf, on_bad_lines='skip', sep = ',')
            VTTP2025 = nbf["key"].to_list()
            VTTP2025 = ['TR_Video',  'Video', 'Segment']+VTTP2025[0:22]+['encoding_time']
            ti  = 'encoding_time'

            dataset = pd.read_csv(csv_dataset, on_bad_lines='skip', sep = ',')
            dataset = shuffle(dataset)
            dataset = shuffle(dataset)
            X = dataset
            idy =  dataset.columns.get_loc(ti)
            y = dataset.iloc[:,idy]     
            X_train_all, X_test_allZabr, y_train, y_testZabr = train_test_split(X, y, test_size=0.2, random_state=123) 
            
            print('Start evalution')
            y_preds_list1 = []
            
            
            y_preds_list1.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy et al.', 'XGB', 'Zabrovskiy_DS'))
            print('XGB 100%')
            y_preds_list1.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy et al.', 'LGB', 'Zabrovskiy_DS'))
            print('LGB 100%')
            y_preds_list1.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy et al.', 'ANN', 'Zabrovskiy_DS'))
            print('ANN 100%')
            y_preds_list1.append(self.VTTP2025(path_project, VTTP2025, X_train_all, X_test_allZabr, y_train, y_testZabr, 'Zabrovskiy et al.', 'KNN', 'Zabrovskiy_DS'))
            print('KNN 100%')
           

        if "SVD_NVD" in ds:     
            sh = sh + 1
            self.showResult(y_preds_list, X_test_all, y_test, "SVD_NVD", sh)
        if 'Zabrovskiy et al.' in ds:
            sh = sh + 1
            self.showResult(y_preds_list1, X_test_allZabr, y_testZabr, 'Zabrovskiy et al.', sh)
        
            
        

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

        l = ['XGBoost', 'LightGBM', 'ANN',  'KNN']
        
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
                codec = list(X_test_all['TR_Codec'])
                y_test = list(y_test)
                #print(codec[1], y_test[1], y_preds[1], y_preds[1][0])
                info = []
                for k, c in enumerate(codec):
                    if i in [0, 1]:
                        info.append([codec[k], y_test[k], y_preds[k]])
                    else:
                        info.append([codec[k], y_test[k], y_preds[k][0]])
                info  = np.array(info)
                infoh64 = []
                infoh65 = []
                for k, c in enumerate(codec):
                    if c == 0:
                        if i in [0, 1]:
                            infoh64.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infoh64.append([codec[k], y_test[k], y_preds[k][0]])
                    elif c == 1:
                        if i in [0, 1]:
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
                        if i in [0, 1]:
                            infocpu.append([codec[k], y_test[k], y_preds[k]])
                        else:
                            infocpu.append([codec[k], y_test[k], y_preds[k][0]])
                    else:
                        if i in [0, 1]:
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
#---------------------------------------------- VTTP2025 ---------------------------------------------------------
#********************************************************************************************************************            

    def VTTP2025(self, path_project,  VTTP2025, X_train_all, X_test_all, y_train, y_test, ds, ml, tds):
        X_trainfinal, X_testfinal, y_trainfinal, y_testfinal, tri  = self.train_test_split1(VTTP2025, X_train_all, X_test_all, y_train, y_test, ds)
        y_predsfinal = []
        df = []
        for k in range(len(X_trainfinal)):
            scaler = MinMaxScaler(feature_range=(0, 1))
            dim = X_trainfinal[k].shape # Get the dimensions of the training dataset.
            x = pd.concat([X_trainfinal[k], X_testfinal[k]], ignore_index=True) # Merging the training and test datasets to apply the scaler transformation.
            x = scaler.fit_transform(x.iloc[:, 3:-1]) 
            X_train_scaled = x[: dim[0], :] # Extract the training dataset from the full dataset.
            X_test_scaled = x[dim[0] :, :] # Extract the testing dataset from the full dataset.
            X_testcsv = x[dim[0] :, :]
            eval_set1 = [(X_train_scaled, y_trainfinal[k]), (X_test_scaled, y_testfinal[k])]
            #make able to run code in GPU and identifier the memory
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True) 
           
            if ml == 'XGB':                
                model = xgb.XGBRegressor(objective ='reg:tweedie', colsample_bytree = 0.97, learning_rate = 0.07, 
                                     subsample = 1, max_depth = 5, reg_alpha = 0.1, n_estimators = 1000, tree_method='gpu_hist')
                params = {
                    "eval_metric": "rmse"  # Move eval_metric inside params
                }
                model.set_params(**params)  # Update model parameters
                model.fit(X_train_scaled, y_trainfinal[k], eval_set=eval_set1, verbose=False)
                

            elif ml == 'LGB':
                model = ltb.LGBMRegressor(colsample_bytree = 1, learning_rate = 0.1, num_leaves= 15,
                                    subsample = 1, max_depth = 8, reg_alpha =0.7, n_estimators=1000 )
                params = {
                    "verbose": -1, 
                    "eval_metric": "rmse" 
                }
                model.set_params(**params)
                model.fit(X_train_scaled, y_trainfinal[k], eval_set=eval_set1)
            elif ml == 'KNN':
                model = KNeighborsRegressor(n_neighbors = 21, weights = 'distance') #21
                model.fit(X_train_scaled, y_trainfinal[k])
            elif ml == 'ANN':
                #make able to run code in GPU and identifier the memory
                physical_devices = tf.config.experimental.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)     
                model = tf.keras.models.Sequential([
                    Dense(22, activation = 'relu', input_dim = 22, kernel_initializer='he_uniform'),
                    Dense(units = 30, activation = 'relu'),
                    Dense(units = 20, activation = 'relu'),
                    Dense(units = 10, activation = 'relu'),
                    Dense(1, activation='linear')])
                model.compile(loss='mean_absolute_error', optimizer= Adadelta(learning_rate=0.1), metrics=['mae', 'mse'])        
                model.fit(X_train_scaled, y_trainfinal[k], epochs=500, batch_size=2000, verbose=0, validation_data = (X_test_scaled, y_testfinal[k]))
            
            y_preds = model.predict(X_test_scaled) # testing the modele
            y_predsfinal.append(y_preds)
            columns = VTTP2025[3:-1]
            df.append(pd.DataFrame(data= X_testcsv, columns=columns))
        trifinal = []
        y_predsfi = []
        y_testfin = []
        y_predsfin = []
        if len(df) == 2:
            bigdata = pd.concat([df[0], df[1]], ignore_index=True)
        else:
            bigdata = df[0]
        for i in range(len(tri)):
            idf = bigdata.columns.get_loc('VM_Type')+1
            y_testfin = y_testfin + list(y_testfinal[i].iloc[:,0])
            y_predsfin = y_predsfin + list(y_predsfinal[i])
            trifinal = trifinal+list(tri[i])
            y_predsfi = y_predsfi+list(y_predsfinal[i])
       
        bigdata.insert(idf, 'Y_trus' , y_testfin, True) 
        bigdata.insert(idf, 'Y_predit' , y_predsfin, True) 
        bigdata.to_excel(path_project+'/results/VTTP2025_'+tds+'_'+ml+'_ypred.xlsx', index = False)
        zpred = zip(y_predsfi, trifinal)
        zpred = sorted(list(zpred), key = lambda x: x[1], reverse=False)
        ypredf, tri1 = zip(*zpred)
        return  list(ypredf)
       

#***************************************************************************************************************************************************
   
    
    def train_test_split1(self, VTTP2025, X_train_all, X_test_all, y_train, y_test, ds):
        if ds == 'Zabrovskiy et al.':
            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            for i, key in enumerate(VTTP2025):
                X_train.insert(i, key, list(X_train_all[key]), True)
                X_test.insert(i, key, list(X_test_all[key]), True)
            return [[X_train], [X_test], [y_train], [pd.DataFrame(np.array(y_test), columns=["Ytest"])], [list(range(len(y_test)))]]
        else:
            l = range(len(y_train))
            y_test1 = list(zip(list(y_test), l))
            X_train = pd.DataFrame()
            X_test = pd.DataFrame()
            
            for i, key in enumerate(VTTP2025):
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
            y_train = list(y_train)
            for i, vm in enumerate(vmtypeTrain):
                if vm == 0:
                    gpuTR.append(list(arrX_train[i,:]))
                    gpuYTR.append(y_train[i])
                elif vm == 1:
                    cpuTR.append(list(arrX_train[i,:]))
                    cpuYTR.append(y_train[i])
            arrgpuTR= np.array(gpuTR)
            X_trainfinal.append(pd.DataFrame(arrgpuTR, columns=list(X_train.keys())))
            arrcpuTR = np.array(cpuTR)
            X_trainfinal.append(pd.DataFrame(arrcpuTR, columns=list(X_train.keys())))
            y_trainfinal.append(pd.DataFrame(np.array(gpuYTR), columns=["YtGPU"]))
            y_trainfinal.append(pd.DataFrame(np.array(cpuYTR), columns=["YtCPU"]) )
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
            y_testfinal.append(pd.DataFrame(np.array(gYTS), columns=["YGPU"]))
            cYTS, t1 = zip(*cpuYTS)
            y_testfinal.append(pd.DataFrame(np.array(cYTS), columns=["YCPU"]))
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
python3 '/content/gdrive/My Drive/Chachou_Transcodage/project2/githubTTP/Predection_Treanscoding_Time_Algorithm.py' \
-p '/content/gdrive/My Drive/Chachou_Transcodage/project2/githubTTP' \
-o 'all_method' -ds 'SVD_NVD' 'Zabrovskiy et al.'
"""
