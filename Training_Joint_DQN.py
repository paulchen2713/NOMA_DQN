# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:19:07 2020

"""

from collections import deque
import scipy
import scipy.special
import math # PackagesNotFoundError
import time # PackagesNotFoundError
import random # PackagesNotFoundError
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # recommended import according to the docs
import pandas as pd
import heapq # PackagesNotFoundError
from keras.utils import np_utils
from random import choice
from itertools import combinations, product # PackagesNotFoundError
from collections import deque # PackagesNotFoundError
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


fd = 10
Ts = 20e-3
x_border = 2
y_border = 2
L = 2
C = 16
max_UE = 10   # user number in one BS
min_distance = 0.035#0.01 #km
max_distance = 0.25#1 #km
max_power = 38 #24. #dBm
min_power = 5.  #dBm
#n_power = -174  #-114. #dBm
n_power = -114.

power_num = 10  #action_num
Ns = 11
dtype = np.float32

class Env_cellular():
    def __init__(self, fd, Ts, x_border, y_border, max_UE, L, max_distance, min_distance, max_power, min_power, n_power, Ns):
        self.fd = fd  #doppler(Hz)
        self.Ts = Ts  #time interval between adjacent instants
        self.x_border = x_border #km
        self.y_border = y_border #km
        
        self.num_cell = self.x_border * self.x_border
        self.cell_BS  = 3
        self.num_BS = self.num_cell * 3#self.x_border * self.y_border #number of TRP 
        self.max_UE = max_UE #number of UE in one cell
        self.total_UE = self.num_cell * self.max_UE #self.max_UE * self.num_BS #Total number of UE in network M
        self.L = L #L=2?!
        self.adjascent_BS = self.num_BS #3*self.L*(self.L+1) + 1 # adjascent BS
        self.adjascent_UE = self.max_UE * self.adjascent_BS   #K
        ## maximum adjascent users, including itself 
        ## the number of adjascent BS * the number of each cell UE
        self.max_distance = max_distance #km
        self.min_distance = min_distance #km
        self.max_power = max_power #max transmit power(dBm)
        self.min_power = min_power #min transmit power(dBm)
        self.max_power_W =  1e-3 *pow(10., self.max_power/10.) #dBm >> mW
        self.min_power_W =  1e-3 *pow(10., self.min_power/10.) #dBm >> mW
        self.n_power   = n_power                           #noise power
        self.n_power_W =  1e-3* pow(10., self.n_power/10.)   #noise power (dBm to mW)
        self.W = np.ones((self.total_UE), dtype=dtype)  #bandwidth
        self.Ns = Ns #?!
        
        self.BS_antenna = 2 #{2,4,8,16}
        

    def train(self):
        
        max_episode = 5000
        INITIAL_EPSILON = 0.2
        FINAL_EPSILON = 0.0001
        batch_size = 128
        PA_state_size    = 65
        Clustering_state_size = 266
        
        PA_action_num = 10
        Clustering_action_num = 5
        Candidate_action_num = 5
        
        PA_agent         = DQNAgent(PA_state_size,PA_action_num)
        Clustering_agent = DQNAgent(Clustering_state_size,Clustering_action_num)
        
        PA_UA_agent = DQNAgent(Clustering_state_size,Candidate_action_num * PA_action_num)
        
        PA_UA_agent.load("Model_512_256_256_128_64_UAPA_DQN_"+str(self.num_cell)+"BS_"+str(self.max_UE)+"UE.h5")

        Clustering_agent.load("Model_512_256_256_128_64_OnlyUA_DQN_"+str(self.num_cell)+"BS_"+str(self.max_UE)+"UE.h5")
        PA_agent.load("Model_512_256_256_128_64_OnlyPA_DQN_"+str(self.num_cell)+"BS_"+str(self.max_UE)+"UE.h5")
        
        
        st = time.time()
        
        Separate_DQN_Average_sum_data_rate_list = []
        Joint_DQN_Average_sum_data_rate_list = []
        GA_Average_sum_data_rate_list = []
        Greedy_Average_sum_data_rate_list = []
        Clustering_Average_sum_data_rate_list = []
        
        Separate_DQN_SDR = []
        Joint_DQN_SDR = []
        Clustering_SDR = []
        GA_SDR = []
        Greedy_SDR = []        
        
        Separate_DQN_Average_rate_list = []
        Joint_DQN_Average_rate_list = []
        Clustering_Average_rate_list = []
        DQN_Average_rate_list  = []
        Greedy_Average_rate_list   = []
        GA_Average_rate_list  = []
        
        
        Separate_DQN = []
        Joint_DQN = []
        DQN    = []
        Greedy = []
        GA     = []
        
        Separate_DQN_PE = []
        Joint_DQN_PE    = []
        DQN_PE = []
        Greedy_PE  = []
        GA_PE  = []
        
        Separate_DQN_SE = []
        Joint_DQN_SE    = []
        DQN_SE = []
        Greedy_SE  = []
        GA_SE  = []
        
        
        Separate_DQN_cdf =[]
        Joint_DQN_cdf = []
        Greedy_cdf = []
        GA_cdf     = []
        
        
        Separate_DQN_rate =[]
        Joint_DQN_rate    =[]
        Greedy_rate =[]
        DQN_rate =[]
        GA_rate  =[]
        
        
        Separate_DQN_Average_PE_list = []
        Joint_DQN_Average_PE_list = []
        GA_Average_PE_list  = []
        DQN_Average_PE_list = []
        Greedy_Average_PE_list  = []
        
        Separate_DQN_Average_SE_list = []
        Joint_DQN_Average_SE_list = []
        GA_Average_SE_list  = []
        DQN_Average_SE_list = []
        Greedy_Average_SE_list  = []
        
        Time = []
        	
        Scheduling_DQN_total_time  = 0
        PA_DQN_total_time = 0
        mmse_total_time = 0 
        GA_total_time = 0
        
        Total_JT = 0
        DQN_JT = 0
        Greedy_JT = 0
        GA_JT = 0
        
        
        Separate_DQN_user = []
        Joint_DQN_user = []
        Greedy_user = []
        ga_user     = []
        
        power_set  = 1e-3 * pow(10., np.linspace(0,self.max_power, PA_action_num)/10.)
        #power_set = np.zeros((PA_action_num,))
        #power_set[1:]  = 1e-3 * pow(10., np.linspace(self.min_power,self.max_power, PA_action_num-1)/10.)
        
        mean_power = np.mean(power_set)
        std_power  = np.std(power_set)
        
        PA_highest_num =10
        scheduling_highest_num = 5
        C_max = 0
        for episode in range(1, max_episode+1):
            '--------------------------------------------- Build environment matrix ---------------------------------------------'
            
            Separate_DQN_sum_data_rate_list = []
            Joint_DQN_sum_data_rate_list = []
            Clustering_sum_data_rate_list = []
            Greedy_sum_data_rate_list = []
            GA_sum_data_rate_list = []
            mmse_sum_data_rate_list = []
            
            Separate_DQN_sum_rate_list = []
            Joint_DQN_sum_rate_list = []
            PA_sum_rate_list         = []
            Clustering_sum_rate_list = []
            Greedy_sum_rate_list    = []
            mmse_sum_rate_list  = []
            GA_sum_rate_list    = []
            
            
            Separate_DQN_PE_list = []
            Joint_DQN_PE_list = []
            GA_PE_list  = []
            DQN_PE_list = []
            Greedy_PE_list  = []
            mmse_PE_list = []
            
            Separate_DQN_SE_list = []
            Joint_DQN_SE_list    = []
            GA_SE_list  = []
            DQN_SE_list = []
            Greedy_SE_list  = []
            mmse_SE_list = []
            
            
            count = 0
            '''
            Jakes model
            '''
            H_set = np.zeros([self.Ns,self.total_UE,self.adjascent_BS,self.BS_antenna], dtype= complex)
            rho = np.float32(scipy.special.j0(2*np.pi*self.fd*self.Ts))   #rho=j0(2pi*fd*Ts), j0 = first kind zero-order Bessel function
            H_set[0,:,:,:] = np.sqrt(0.5*(np.random.randn(self.total_UE, self.adjascent_BS, self.BS_antenna) + 1j * np.random.randn(self.total_UE, self.adjascent_BS, self.BS_antenna)))
              
            for i in range(1,self.Ns):
                H_set[i,:,:,:]  = H_set[i-1,:,:,:]*rho + np.sqrt((1.-rho**2)*0.5*(np.random.randn(self.total_UE, self.adjascent_BS, self.BS_antenna) + 1j * np.random.randn(self.total_UE, self.adjascent_BS, self.BS_antenna)))          
            
            x_center = []
            y_center = []
            position_tx = []
            position_ty = []            
            small_length = 1/6 #0.03 / np.sqrt(3.)#1 / (4 * np.sqrt(3.))
            
            x_gap =  3 * small_length #0.5 #(1/2)* length *np.sqrt(3.) * 2
            y_gap =  ((3*np.sqrt(3.)) /2) * small_length   #3 / (4 * np.sqrt(3.))#np.array(length) #3 * length
            x_y_gap = (3/2) * small_length  #np.array( length / np.sqrt(3.) ) #np.sqrt(3.) * (1/2) * length
            
            for y in range(self.y_border):
                for x in range(self.x_border):
                    
                    
                    x_axis = x * x_gap - int(y%2) * x_y_gap
                    x_center.append(x_axis)                    
                    y_axis = y *  y_gap #+ x_y_axis
                    y_center.append(y_axis)
                    
                    cell_number = y*4 + x
                    for i in range(self.cell_BS):
                        position_tx.append(x_axis)
                        position_ty.append(y_axis)
            
            distance_rx = np.random.uniform(0, 0.25, size = (self.num_BS, self.max_UE))  #distance between BS and UE
            angle_rx    = np.random.uniform(-np.pi, np.pi, size = (self.num_BS, self.max_UE)) #The angle of one circle to UE
            position_rx = np.zeros((self.num_cell, self.max_UE))
            position_ry = np.zeros((self.num_cell, self.max_UE))
            x_round = []
            y_round = []
            x = []
            y = []
            
            for i in range(self.num_BS):
                x_round.append(position_tx[i])
                y_round.append(position_ty[i])
            for i in range(self.num_cell):
                for j in range(self.max_UE):
                    position_rx[i,j] = x_center[i] + distance_rx[i,j] * np.cos(angle_rx[i,j])
                    position_ry[i,j] = y_center[i] + distance_rx[i,j] * np.sin(angle_rx[i,j])
                    x.append(position_rx[i,j])
                    y.append(position_ry[i,j])
            distance = 1e10 * np.ones((self.total_UE, self.num_BS), dtype=dtype)
            for k in range(self.num_BS):
                for i in range(self.num_cell):
                    for j in range(self.max_UE):
                        dx = np.square((position_tx[k] - position_rx[i,j]))
                        dy = np.square((position_ty[k] - position_ry[i,j]))
                        dis = np.sqrt(dx + dy)
                        if dis < 0.01:
                            dis = 0.01
                        else:
                            dis = dis
                        distance[i*self.max_UE+j,k] = dis
            
            Antenna_pattern = 12* np.square(np.degrees(angle_rx)/70)
            Antenna_pattern[Antenna_pattern>20] = 20

            Att_pattern_expand = np.zeros((self.num_BS,self.total_UE))
            for i in range(self.num_cell):
                temp = np.array( Antenna_pattern[i*self.cell_BS:(i+1)*self.cell_BS , :])
                for j in range(self.num_cell):
                    Att_pattern_expand[j*self.cell_BS:(j+1)*self.cell_BS , i * self.max_UE:(i+1)*self.max_UE] = temp
            
            Att_gain = 14 #(dBi)  #anttena gain
            freq = 2000 #(M)      #carrier freq
            height_bs = 32 #(m)   #BS_ant_height
            height_ue = 1 #(m)    #UE_ant_height
            
            Att_pattern_expand = Att_pattern_expand.transpose()
            
            path_loss = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(height_bs) - ((1.1* np.log10(freq)-0.7)*height_ue-(1.56*np.log10(freq)-0.8)) + (44.9-6.55*np.log10(height_bs))*np.log10(distance) - Att_gain + Att_pattern_expand + (8 * np.random.rand(self.total_UE, self.num_BS))
            path_loss = pow(10.,-path_loss/10.)
            
            channel = np.zeros([self.Ns,self.total_UE,self.adjascent_BS,self.BS_antenna], dtype=complex)
            for i in range(self.BS_antenna):
                channel[:,:,:,i] = H_set[:,:,:,i] *path_loss# ch
                       
            MRT_precoding = np.zeros([self.Ns,self.total_UE,self.adjascent_BS,self.BS_antenna], dtype=complex)
            for i in range(self.BS_antenna):
                MRT_precoding[:,:,:,i] = H_set[:,:,:,i] / np.linalg.norm(H_set, axis=-1)
            
            channel_gain = (np.linalg.norm(H_set * np.conj(MRT_precoding), axis=-1)**2) * path_loss#ch
            #channel_gain = (np.linalg.norm(H_set, axis=-1)**2) * path_loss
            
            H2 = np.array(H_set[count,:,:,:])
            h_gain = np.array(channel_gain[count,:,:])
            MR_precoder = np.array(MRT_precoding[count,:,:,:])
            
            '------------------------------------------- The end of Build environment matrix -----------------------------------------'

            '--- (Start) Randomly Clustering ----'
            UE_candidate = list()
            idx_array = np.zeros((self.num_BS,Clustering_action_num), dtype = np.int32)          
            for i in range(self.num_BS):
                cell_number = int( i / 3 )  # 1cell 3BS
                heap = list(h_gain[cell_number * self.max_UE : ((cell_number + 1) * self.max_UE),i])
                UE_candidate.append(list(map(heap.index , heapq.nlargest(Clustering_action_num, heap)))   )
                idx_array[i,:] = cell_number * self.max_UE
            idx_array = np.array(idx_array)
            UE_candidate = np.array(UE_candidate)
            UE_candidate_idx = np.array(UE_candidate)
            UE_candidate = UE_candidate + idx_array
            UE_candidate = UE_candidate.tolist()
            
            #clustering_random_action is the UE idex for the macro cell
            clustering_random_action = list()
            for i in range(self.num_BS):
                random_sample = choice(UE_candidate[i])
                clustering_random_action.append(random_sample)
            
            #clustering_random_action = np.argmax(h_gain, axis=0)
            #clustering_random_action = ga_action2
            clustering_random_action = np.zeros((self.num_BS,1),dtype = np.int32)
            for i in range(self.num_BS):
                cell_number = int(i / self.cell_BS)
                clustering_random_action[i] = np.argmax(h_gain[cell_number*self.max_UE : (cell_number+1)*self.max_UE], axis=0)[i]
                clustering_random_action[i] = clustering_random_action[i] + cell_number * self.max_UE
            clustering_random_action = clustering_random_action.flatten()
            
            #choose action is the action index for the DQN
            choose_action = list()
            for i in range(self.num_BS):
                correspond_idx = UE_candidate[i].index(clustering_random_action[i])
                choose_action.append(correspond_idx)
            
            Comp_index         = np.zeros((self.num_BS,self.total_UE))
            interference_index = np.zeros((self.num_BS,self.total_UE))
            Comp_index[range(self.num_BS),clustering_random_action] = 1
            Comp_index = Comp_index.transpose()
            Comp_number = np.sum(Comp_index,axis=1)
            Comp_number = Comp_number.reshape(self.total_UE,1)
            
            
            interference_index = np.array(( Comp_index - 1 ) * (-1))
            Comp_indicate = np.array(Comp_index)
            
            Comp_index = Comp_index.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
            c_power = self.max_power_W * np.ones((self.num_BS,1))
            c_main_path = H2 * Comp_index
            c_main_path = c_main_path * np.conj(MR_precoder)
            #c_main_path = np.sum(abs(c_main_path)**2,axis=-1) * path_loss
            c_main_path = np.linalg.norm(c_main_path,axis=-1)**2 * path_loss
            cm_path = np.array(c_main_path)
            c_main_path = np.dot(c_main_path, c_power)
            
            inter_index = ( Comp_index - 1 ) * (-1)
            c_inter_path = H2 * inter_index
            c_inter_path = c_inter_path * np.conj(MR_precoder)
            #c_inter_path = np.sum(abs(c_inter_path)**2,axis=-1) * path_loss
            c_inter_path = np.linalg.norm(c_inter_path,axis=-1)**2 * path_loss
            ci_path = np.array(c_inter_path)
            c_inter_path = np.dot(c_inter_path, c_power)
            
            c_min_sinr = np.minimum( c_main_path / (c_inter_path + self.n_power_W) , 1000) # capped sinr max 30dB
            c_sinr = c_main_path / (c_inter_path + self.n_power_W)
            
            c_nor_rate = np.log2(1. + c_sinr) / Comp_number
            c_nor_rate[ Comp_number==0 ] = 0
            min_c_nor_rate = np.log2(1. + c_min_sinr) / Comp_number
            min_c_nor_rate [ Comp_number==0 ] = 0
                                    
            '--- (End) Randomly Clustering ----'
            '---- (Start) Calculate clustering reward ----'
            '----- 移除 BS k original serving UE  --------'                        
            Comp_rre = np.zeros((self.num_BS,self.num_BS,self.total_UE))
            Comp_rre[:,range(self.num_BS),clustering_random_action] = 1
            
            Comp_rre[range(self.num_BS),range(self.num_BS),clustering_random_action] = 0                
            Comp_rre = Comp_rre.transpose(0,2,1)
            rre_Comp_number = Comp_rre.sum(2)
            rre_Comp_number = rre_Comp_number.reshape(self.num_BS,self.total_UE,1)
            
            Comp_rre = Comp_rre.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
            rre_main_path = H2 * Comp_rre
            rre_main_path = rre_main_path * np.conj(MR_precoder)
            #rre_main_path = np.sum(abs(rre_main_path)**2,axis=-1) * path_loss
            rre_main_path = np.linalg.norm(rre_main_path,axis=-1)**2 * path_loss
            rre_main_path = np.dot(rre_main_path, c_power)
            
            inter_rre = (Comp_rre - 1) * (-1)
            rre_inter_path = H2 * inter_rre
            rre_inter_path = rre_inter_path * np.conj(MR_precoder)
            #rre_inter_path = np.sum(abs(rre_inter_path)**2,axis=-1) * path_loss
            rre_inter_path = np.linalg.norm(rre_inter_path,axis=-1)**2 * path_loss
            rre_inter_path = np.dot(rre_inter_path, c_power)
            
            rre_sinr = rre_main_path / (rre_inter_path + self.n_power_W)
            rre_nor_rate = np.log2(1. + rre_sinr) / rre_Comp_number
            rre_nor_rate[rre_Comp_number == 0] = 0   # remove BSk serving UE's nor-rate.
            
            '----- 移除 BS k original serving UE  --------'
            '----- 計算 BS k 對非serving UE 的干擾大小 -----'
            c_re_inter = np.tile(interference_index,[self.num_BS,1,1])
            c_re_inter[range(self.num_BS),:,range(self.num_BS)] = 0
            c_re_inter = c_re_inter.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
            
            c_re_inter_path = H2 * c_re_inter
            c_re_inter_path = c_re_inter_path * np.conj(MR_precoder)
            #c_re_inter_path = np.sum(abs(c_re_inter_path)**2,axis=-1) * path_loss
            c_re_inter_path = np.linalg.norm(c_re_inter_path,axis=-1)**2 * path_loss
            ci_re_path = np.array(c_re_inter_path)
            c_re_inter_path = np.dot(c_re_inter_path, c_power)
            
            #c_re_sinr = np.minimum( main_path / (re_inter_path + self.n_power_W) , 1000)
            c_re_sinr = c_main_path / (c_re_inter_path + self.n_power_W)
            c_re_Comp_num = np.tile(Comp_number,[self.num_BS,1,1])
            c_re_nor_rate = ( np.log2(1. + c_re_sinr) / c_re_Comp_num ) 
            c_re_nor_rate[ c_re_Comp_num==0 ] = 0
                        
            c_re =  c_re_nor_rate - c_nor_rate
            c_re_sum = np.sum(c_re,axis=1)
                        
            '----- 計算 BS k 對非serving UE 的干擾大小 -----'
            
            '--------- 計算 BS k 服務非serving UE造成的nor rate 的增減 ---------'
            comp_other_indx = np.zeros((self.num_BS,self.num_BS,self.total_UE))
            comp_other_indx[:,range(self.num_BS),clustering_random_action] = 1
            comp_other_indx = comp_other_indx.transpose(0,2,1)
            for i in range(self.num_BS):
                cell_number = int(i/self.cell_BS)
                comp_other_indx[i, cell_number*self.max_UE : (cell_number+1)*self.max_UE, i] = 1
            comp_other_number = comp_other_indx.sum(2)
            comp_other_number = comp_other_number.reshape(self.num_BS,self.total_UE,1)
            
            comp_other_indx = comp_other_indx.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
            other_main_path = H2 * comp_other_indx
            other_main_path = other_main_path * np.conj(MR_precoder)
            #other_main_path = np.sum(abs(other_main_path)**2,axis=-1) * path_loss
            other_main_path = np.linalg.norm(other_main_path,axis=-1)**2 * path_loss
            other_main_path = np.dot(other_main_path, c_power)
            
            comp_inter_other_indx = ( comp_other_indx - 1 ) * (-1)
            other_inter_path = H2 *comp_inter_other_indx
            other_inter_path = other_inter_path * np.conj(MR_precoder)
            #other_inter_path = np.sum(abs(other_inter_path)**2,axis=-1) * path_loss
            other_inter_path = np.linalg.norm(other_inter_path,axis=-1)**2 * path_loss
            other_inter_path = np.dot(other_inter_path, c_power)
            
            other_sinr = other_main_path / (other_inter_path + self.n_power_W)
            other_nor_rate = np.log2(1. + other_sinr) / comp_other_number
            other_nor_rate[comp_other_number == 0] = 0
            other_nor_rate_re = np.array(other_nor_rate)
            
            other_rate = []
            for i in range(self.num_BS):
                temp = np.delete(c_nor_rate, clustering_random_action[i])
                other_rate.append(temp)
            other_rate = np.array(other_rate).reshape(self.num_BS,self.total_UE-1,1)
            
            other_nor_rate = []
            for i in range(self.num_BS):
                temp = np.delete(other_nor_rate_re[i], clustering_random_action[i])
                other_nor_rate.append(temp)
            other_nor_rate = np.array(other_nor_rate).reshape(self.num_BS,self.total_UE-1,1)
            
            term1 = np.tile((c_nor_rate[clustering_random_action] - rre_nor_rate[range(self.num_BS),clustering_random_action]),(self.total_UE-1,1,1)).transpose()
            #term1 = np.tile((c_nor_rate[clustering_random_action] - rre_nor_rate[range(self.num_BS),clustering_random_action]),(self.total_UE,1,1)).transpose()
            
            term1 = term1.reshape(self.num_BS, self.total_UE-1, 1)
            term2 =  other_rate - other_nor_rate  #-1 * other_nor_rate#other_rate - other_nor_rate # -1 * other_nor_rate
            temp  = np.array( term1 + term2)
            
            rewardA =[]
            for i in range(self.num_BS):
                cell_number = int(i/self.cell_BS)
                reward = temp[i][cell_number * self.max_UE : (cell_number+1) * self.max_UE - 1]
                reward = np.min(temp[i][cell_number * self.max_UE : (cell_number+1) * self.max_UE - 1], axis = 0)
                rewardA.append(reward)
            rewardA = np.array(rewardA).reshape(self.num_BS,1)
            serving_rate = rewardA                         
            #serving_rate =  np.mean(rewardA)
            c_re_sum = c_re_sum
            #clustering_reward = np.tile(np.mean(c_nor_rate),(self.num_BS,1)) + serving_rate
            clustering_reward = c_nor_rate[clustering_random_action] - c_re_sum #serving_rate
            
            '---- (End) calculate clustering reward ----'
            
            '-------- (Start) Randomly PA ---------'
            
            Comp_number = Comp_number.flatten()#np.sum(Comp_index,axis=1)
            PA_random_action = np.random.randint(0, high = PA_action_num, size = (self.num_BS))
            PA_random_action = (PA_action_num-1) * np.ones((1,self.num_BS), dtype=np.int8).flatten()
            
            PA_power = power_set[PA_random_action]
            PA_main_path  = np.dot(cm_path, PA_power)
            PA_inter_path = np.dot(ci_path, PA_power)
            
            min_sinr = np.minimum( PA_main_path / (PA_inter_path + self.n_power_W) , 1000) # capped sinr max 30dB
            sinr = PA_main_path / (PA_inter_path + self.n_power_W)
            PA_nor_rate = np.log2(1. + sinr) / Comp_number
            PA_nor_rate[ Comp_number==0 ] = 0   #分母是0時，得到 0
            min_nor_rate = np.log2(1. + min_sinr) / Comp_number
            min_nor_rate [ Comp_number==0 ] = 0
            
            '-------- (End) Randomly PA ---------'
            
            '---- (Start) Calculate PA reward ----'
            
            pa_re_inter_path = np.dot(ci_re_path , PA_power)
            #pa_re_sinr = np.minimum( main_path / (re_inter_path + self.n_power_W) , 1000)
            pa_re_sinr = PA_main_path / (pa_re_inter_path + self.n_power_W)
            pa_re_Comp_num = np.tile(Comp_number,[self.num_BS,1])
            PA_re_nor_rate = ( np.log2(1. + pa_re_sinr) / pa_re_Comp_num ) 
            PA_re_nor_rate[ pa_re_Comp_num==0 ] = 0
            PA_re =  PA_re_nor_rate - PA_nor_rate
            PA_re_sum = np.sum(PA_re,axis=1)
            PA_reward = PA_nor_rate[clustering_random_action] - PA_re_sum
                        
            '---- (End) Calculate PA reward ----'
                        
            '---- (Start) Gernerate clustering first step ----'            
            
            previous_PA_main_channel = []
            previous_PA_interference = []
            previous_interfered_channel = []
            previous_PA_inter_power =[]
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS )  # 1cell 3BS 
                cell_BS_idx = int( i % self.cell_BS )
                
                intra_interference = np.array(h_gain[clustering_random_action[i],cell_number * self.cell_BS : (cell_number+1)*self.cell_BS])
                intra_interference = np.roll(intra_interference, -1 * cell_BS_idx)
                main_channel = intra_interference[0]                
                intra_interference = intra_interference[1:]
                inter_interference = np.array(h_gain[clustering_random_action[i],:])
                inter_interference = np.delete(inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                
                max_inter_idx = heapq.nlargest( PA_highest_num, range(len(inter_interference)), inter_interference.take)
                inter_interference = inter_interference[max_inter_idx]
                
                interfered_link = h_gain[clustering_random_action,i]
                interfered_link[interfered_link==h_gain[clustering_random_action[i],i]] = 0
                max_interfered_idx = heapq.nlargest( PA_highest_num, range(len(interfered_link)), interfered_link.take)
                interfered_link = interfered_link[max_interfered_idx]
                
                interfernce = np.hstack((intra_interference, inter_interference))
                PA_channel     = np.hstack((main_channel, interfernce, interfered_link))
                
                h_mean = np.mean(PA_channel)
                h_std  = np.std(PA_channel)
                
                main_channel = (main_channel - h_mean) / h_std
                interfernce  = (interfernce - h_mean) / h_std
                interfered_link = (interfered_link - h_mean) / h_std
                
                previous_PA_main_channel.append(main_channel)
                previous_PA_interference.append(interfernce)
                previous_interfered_channel.append(interfered_link)
                previous_PA_inter_power.append(PA_power[max_inter_idx])
                
            previous_PA_main_channel = np.array(previous_PA_main_channel).reshape(self.num_BS,1)
            previous_PA_interference = np.array(previous_PA_interference)
            previous_interfered_channel = np.array(previous_interfered_channel)
            previous_PA_inter_power = np.array(previous_PA_inter_power)
            previous_PA_inter_power = (previous_PA_inter_power - mean_power) / std_power
            
            
            '--- Clustering previous input with scalable -----' 
            #temp = np.array(h_gain * np.tile(c_power, (self.total_UE,1)).reshape(self.total_UE,self.num_BS ))
            temp = np.array(h_gain)
            previous_channel = []
            for i in range(self.num_BS):
                previous_channel.append( temp[UE_candidate[i],i] )
            previous_channel = np.array(previous_channel)
            
            
            previous_main_channel = []
            for i in range(self.num_BS):
                cell_num    = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1     = previous_channel[cell_num * self.cell_BS : (cell_num+1) * self.cell_BS]   #[[BS1_max,BS1_sec,BS1_third][BS2_max,BS2_sec...][...]]
                flag = np.roll(temp1, -1 * cell_BS_idx * self.max_UE  )
                #flag = (flag - mean_UE) / std_UE
                previous_main_channel.append(flag.flatten())
            previous_main_channel = np.array(previous_main_channel)
            
            #UE_interference       = []  #include intra and 5 max inter interference
            UE_inter_interference = []
            UE_intra_interference = []
            previous_inter_power  = []
            for i in range(self.total_UE):
                cell_number = int(i / self.max_UE)
                intra_interference = np.array( temp[i,cell_number*self.cell_BS : (cell_number+1)*self.cell_BS])
                inter_interference = np.array(temp[i,:])
                inter_interference = np.delete(inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                max_inter_idx = heapq.nlargest( scheduling_highest_num, range(len(inter_interference)), inter_interference.take)
                inter_interference = inter_interference[max_inter_idx]
                inter_power        = c_power[max_inter_idx]
                UE_inter_interference.append(inter_interference)
                UE_intra_interference.append(intra_interference)
                previous_inter_power.append(inter_power)
           
            UE_inter_interference = np.array(UE_inter_interference)
            UE_intra_interference = np.array(UE_intra_interference)
            
            UE_interference = np.hstack((UE_intra_interference,UE_inter_interference))
            previous_inter_power  = np.array(previous_inter_power)
            previous_inter_power  =   previous_inter_power / self.max_power_W#(previous_inter_power - mean_power) / std_power
            
            previous_BS_input = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                intra = np.roll(UE_intra_interference[cell_number* self.max_UE : (cell_number+1)* self.max_UE], -1 * cell_BS_idx, axis=1)
                
                feature1 = np.hstack((previous_main_channel[i], intra[UE_candidate_idx[i]].flatten(), UE_inter_interference[UE_candidate[i]].flatten()))
                g_mean    = np.mean(feature1)
                g_std     = np.std(feature1)
                feature1 = ( feature1 - g_mean ) / g_std
                feature2 = previous_inter_power[UE_candidate[i]].flatten()
                previous_BS_input.append(np.hstack((feature1,feature2)))
            previous_BS_input = np.array(previous_BS_input)
                        
            
            UE_candidate = np.array(UE_candidate)            
            previous_repeat_candidate = []
            for k in range(self.num_BS):
                for i in range(Clustering_action_num):
                    BS_reapet = np.zeros((self.num_BS, Clustering_action_num))
                    repeat = np.where(UE_candidate == UE_candidate[k][i])
                    BS_reapet[repeat[0],repeat[1]] = 1
                    cell_num = int (k / self.cell_BS)
                    idx          = cell_num*self.cell_BS
                    cell_num     = int(k / self.cell_BS)
                    cell_BS_idx  = int(k % self.cell_BS)   #corrseponding label of BS for each cell
                    if cell_BS_idx == 0:
                        BS_reapet = BS_reapet[[1 + idx ,2 + idx],:].flatten()
                        previous_repeat_candidate.append(BS_reapet)
                    if cell_BS_idx == 1:
                        BS_reapet = BS_reapet[[2 + idx ,0 + idx ],:].flatten()
                        previous_repeat_candidate.append(BS_reapet)
                    if cell_BS_idx == 2:
                        BS_reapet = BS_reapet[[0 + idx ,1 + idx],:].flatten()
                        previous_repeat_candidate.append(BS_reapet)   
            previous_repeat_candidate = np.array(previous_repeat_candidate).reshape(self.num_BS, (self.cell_BS -1) * Clustering_action_num * Clustering_action_num )
            
            pre_reward = np.array(clustering_reward)
            previous_reward = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1 = pre_reward[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                temp2 = np.roll(temp1 , -1 * cell_BS_idx)
                previous_reward.append(temp2)
            previous_reward = np.array(previous_reward).reshape(self.num_BS,self.cell_BS)
            
            
            pre_power = np.array(c_power)
            previous_power = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1 = pre_power[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                temp2 = np.roll(temp1 , -1 * cell_BS_idx)
                previous_power.append(temp2)
            previous_power = np.array(previous_power).reshape(self.num_BS,self.cell_BS)
            previous_power = previous_power / self.max_power_W #(previous_power - mean_power) / std_power
            
            '--- Clustering previous input with scalable -----' 
            
            '----- next time slot ----'
            
            count += 1
            H2 = np.array(H_set[count,:,:,:])
            h_gain = np.array(channel_gain[count,:,:])
            MR_precoder = np.array(MRT_precoding[count,:,:,:])
            
            UE_candidate = list()
            idx_array = np.zeros((self.num_BS,Clustering_action_num), dtype = np.int32)          
            for i in range(self.num_BS):
                cell_number = int( i / 3 )  # 1cell 3BS
                heap = list(h_gain[cell_number * self.max_UE : ((cell_number + 1) * self.max_UE),i])
                UE_candidate.append(list(map(heap.index , heapq.nlargest(Clustering_action_num, heap)))   )
                idx_array[i,:] = cell_number * self.max_UE
            idx_array = np.array(idx_array)
            UE_candidate = np.array(UE_candidate)
            UE_candidate_idx = np.array(UE_candidate)
            UE_candidate = UE_candidate + idx_array
            UE_candidate = UE_candidate.tolist()
            
            '----- next time slot ----'
            
            '--- Clustering cuerrent input with scalable -----' 
            #temp = np.array(h_gain * np.tile(PA_power, (self.total_UE,1)).reshape(self.total_UE,self.num_BS ))
            temp = np.array(h_gain)
            current_channel = []
            for i in range(self.num_BS):
                current_channel.append( temp[UE_candidate[i],i] )
            current_channel = np.array(current_channel)
            
            current_main_channel = []
            for i in range(self.num_BS):
                cell_num    = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1     = current_channel[cell_num * self.cell_BS : (cell_num+1) * self.cell_BS]   #[[BS1_max,BS1_sec,BS1_third][BS2_max,BS2_sec...][...]]
                flag = np.roll(temp1, -1 * cell_BS_idx * self.max_UE  )
                #flag = (flag - mean_UE) / std_UE
                current_main_channel.append(flag.flatten())
                
            current_main_channel = np.array(current_main_channel)
            
            
            #UE_interference       = []  #include intra and 5 max inter interference
            UE_inter_interference = []
            UE_intra_interference = []
            current_inter_power  = []
            for i in range(self.total_UE):
                cell_number = int(i / self.max_UE)
                intra_interference = np.array( temp[i,cell_number*self.cell_BS : (cell_number+1)*self.cell_BS])
                inter_interference = np.array( temp[i,:])
                inter_interference = np.delete(inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                max_inter_idx = heapq.nlargest( scheduling_highest_num, range(len(inter_interference)), inter_interference.take)
                inter_interference = inter_interference[max_inter_idx]
                inter_power        = PA_power[max_inter_idx]                
                UE_inter_interference.append(inter_interference)
                UE_intra_interference.append(intra_interference)
                current_inter_power.append(inter_power)
           
            UE_inter_interference = np.array(UE_inter_interference)
            UE_intra_interference = np.array(UE_intra_interference)
            
            current_inter_power  = np.array(current_inter_power)
            current_inter_power  = current_inter_power / self.max_power_W#(current_inter_power - mean_power) / std_power
            
            
            current_BS_input = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                intra = np.roll(UE_intra_interference[cell_number* self.max_UE : (cell_number+1)* self.max_UE], -1 * cell_BS_idx, axis=1)
                feature1 = np.hstack((current_main_channel[i], intra[UE_candidate_idx[i]].flatten(), UE_inter_interference[UE_candidate[i]].flatten()))
                g_mean    = np.mean(feature1)
                g_std     = np.std(feature1)
                feature1 = ( feature1 - g_mean ) / g_std
                feature2 = current_inter_power[UE_candidate[i]].flatten()
                current_BS_input.append(np.hstack((feature1,feature2)))
            current_BS_input = np.array(current_BS_input)
            
            
            UE_candidate = np.array(UE_candidate)            
            current_repeat_candidate = []
            for k in range(self.num_BS):
                for i in range(Clustering_action_num):
                    BS_reapet = np.zeros((self.num_BS, Clustering_action_num))
                    repeat = np.where(UE_candidate == UE_candidate[k][i])
                    BS_reapet[repeat[0],repeat[1]] = 1
                    cell_num = int (k / self.cell_BS)
                    idx          = cell_num*self.cell_BS
                    cell_num     = int(k / self.cell_BS)
                    cell_BS_idx  = int(k % self.cell_BS)   #corrseponding label of BS for each cell
                    if cell_BS_idx == 0:
                        BS_reapet = BS_reapet[[1 + idx ,2 + idx],:].flatten()
                        current_repeat_candidate.append(BS_reapet)
                    if cell_BS_idx == 1:
                        BS_reapet = BS_reapet[[2 + idx ,0 + idx ],:].flatten()
                        current_repeat_candidate.append(BS_reapet)
                    if cell_BS_idx == 2:
                        BS_reapet = BS_reapet[[0 + idx ,1 + idx],:].flatten()
                        current_repeat_candidate.append(BS_reapet)   
            current_repeat_candidate = np.array(current_repeat_candidate).reshape(self.num_BS, (self.cell_BS -1) * Clustering_action_num * Clustering_action_num )
            
            cur_power = np.array(PA_power).reshape(self.num_BS,1)
            current_power = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1 = cur_power[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                temp2 = np.roll(temp1 , -1* cell_BS_idx)
                current_power.append(temp2)
            current_power = np.array(current_power).reshape(self.num_BS,self.cell_BS)
            power_matrix = np.array((current_power - mean_power) / std_power)
            
            current_power = current_power / self.max_power_W#(current_power - mean_power) / std_power
            
            '--- Clustering cuerrent input with scalable -----'
            
            clustering_input_state = np.hstack(( previous_BS_input, previous_repeat_candidate, previous_power, current_BS_input, current_repeat_candidate, current_power))
            
            current_UA_PA_BS_input = np.array(current_BS_input)
            current_UA_PA_power = np.array(current_power)
            DQN_input_state = np.array(clustering_input_state)
            
            '---- (Start) Gernerate PA first step ---'
            
            #power_matrix = np.array(current_power)
            rate = np.array(PA_nor_rate[clustering_random_action]).reshape(self.num_BS,1)
            rate_matrix = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                temp1 = rate[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                temp2 = np.roll(temp1 , -cell_BS_idx)
                rate_matrix.append(temp2)
            rate_matrix = np.array(rate_matrix).reshape(self.num_BS,self.cell_BS)
            
            reward = np.array(PA_reward).reshape(self.num_BS,1)
            input_reward = []
            previous_repeat_matrix = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                
                temp1 = reward[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                temp2 = np.roll(temp1 , -cell_BS_idx)
                input_reward.append(temp2)
                                
                repeat = Comp_indicate[clustering_random_action[i]]
                repeat = repeat[cell_number * self.cell_BS : (cell_number+1) * self.cell_BS]
                repeat = np.roll(repeat,  -1 * cell_BS_idx)
                previous_repeat_matrix.append(repeat)
            
            input_reward = np.array(input_reward).reshape(self.num_BS,self.cell_BS)
            previous_repeat_matrix = np.array(previous_repeat_matrix)

            '---- (End) Gernerate PA first step ---'
            '----------------------------------------------------- Start Time Slot ------------------------------------------------------------'
            for time_s in range(int(self.Ns)-2):
                
                GA_t1 = time.time()
                
                total_combins = int(math.pow(self.max_UE,self.cell_BS))
                equal_power = self.max_power_W * np.ones((self.num_BS,1))
                '------------------------------------------- Global Search -----------------------------------------'
                
                Max_Action = []
                '------------------------------------------- Global Search -----------------------------------------'
                for j in range(self.num_cell):
                    
                    max_reward = 0
                    global_max_action = np.zeros((self.num_BS,),dtype = np.int32) 
                    
                    start_idx =   j    * self.max_UE
                    end_idx   = (j+1)  * self.max_UE
                    
                    GA_power = self.max_power_W * np.ones((self.num_BS,1)) #np.array(PA_power).reshape((self.num_BS,1))#PA_random_action.reshape((self.num_BS,1))
                    h_part_gain = np.array( h_gain[start_idx:end_idx, :] )
                    
                    for i in range(total_combins):
                        action = np.zeros((self.cell_BS,),dtype = np.int32)                    
                        count_c = self.cell_BS - 1
                        while( i/self.max_UE > 0 ):
                            d1 = i%self.max_UE
                            action[count_c] = d1
                            i = int(i/self.max_UE)
                            count_c -=1
                        arrange_idx = np.array(range(self.cell_BS))
                        arrange_idx = arrange_idx + self.cell_BS * j
                        arrange_idx = np.transpose(arrange_idx)
                        
                        GA_idx = np.zeros((self.max_UE,self.num_BS))
                        GA_idx[action,arrange_idx] = 1
                        
                        GA_number = np.sum(GA_idx,axis=1,dtype=np.int8)
                        GA_number = GA_number.reshape(self.max_UE,1)         
                        #GA_idx = GA_idx.repeat(self.BS_antenna).reshape(self.max_UE,self.num_BS,self.BS_antenna)
                        GA_main_path = h_part_gain * GA_idx
                        GA_main_path = np.dot(GA_main_path, equal_power)
                        
                        GA_inter_idx = (GA_idx - 1) * (-1)
                        GA_inter_path = h_part_gain * GA_inter_idx
                        GA_inter_path = np.dot(GA_inter_path, equal_power)
                        
                        GA_sinr = GA_main_path / (GA_inter_path + self.n_power_W)
                        GA_data_rate = np.log2(1. + GA_sinr)
                        
                        GA_nor_rate = np.log2(1. + GA_sinr) / GA_number
                        GA_nor_rate[ GA_number==0 ] = 0

                        
                        '----------------  Generate next step optimal action and input-----------------'
                        if np.sum(GA_data_rate) > max_reward:
                            max_reward = np.sum(GA_data_rate)
                            global_max_action = action + self.max_UE * j 

                    Max_Action.append(global_max_action)
                   # print("global_max_action =\n",global_max_action)
                   # print("np.sum(max_global_data_rate) = ",np.sum(max_global_data_rate))
                Max_Action  = np.array(Max_Action)
                Best_action = Max_Action.flatten()
                
                
                GA_idx = np.zeros((self.num_BS,self.total_UE))
                GA_idx[range(self.num_BS),Best_action] = 1
                GA_idx = GA_idx.transpose()
                
                GA_number = np.sum(GA_idx,axis=1,dtype=np.int8)
                GA_number = GA_number.reshape(self.total_UE,1)
                
                GA_idx = GA_idx.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                GA_main_path = H2 * GA_idx
                GA_main_path = GA_main_path * np.conj(MR_precoder)
                #Greedy_main_path = np.sum(abs(Greedy_main_path)**2,axis=-1) * path_loss
                GA_main_path = np.linalg.norm(GA_main_path,axis=-1)**2 * path_loss
                GA_main_path = np.dot(GA_main_path, equal_power)
                
                GA_inter_idx = (GA_idx - 1) * (-1)
                GA_inter_path = H2 * GA_inter_idx
                GA_inter_path = GA_inter_path * np.conj(MR_precoder)
                #Greedy_inter_path = np.sum(abs(Greedy_inter_path)**2,axis=-1) * path_loss
                GA_inter_path = np.linalg.norm(GA_inter_path,axis=-1)**2 * path_loss
                GA_inter_path = np.dot(GA_inter_path, equal_power)
                
                GA_sinr = GA_main_path / (GA_inter_path + self.n_power_W)
                
                GA_data_rate = np.log2(1. + GA_sinr)
                                
                GA_nor_rate = np.log2(1. + GA_sinr) / GA_number
                GA_nor_rate[ GA_number==0 ] = 0
                
                GA_t2 = time.time()
                '--------- Max-P + Greedy ------'
                
                equal_power = self.max_power_W * np.ones((self.num_BS,1))  # equal max power
                #equal_power = np.array(PA_power).reshape((self.num_BS,1))
                #Greedy_action = ga2_action#np.argmax(h_gain, axis=0)
                Greedy_action = np.zeros((self.num_BS,1),dtype = np.int32)
                for i in range(self.num_BS):
                    cell_number = int(i / self.cell_BS)
                    Greedy_action[i] = np.argmax(h_gain[cell_number*self.max_UE : (cell_number+1)*self.max_UE], axis=0)[i]
                    Greedy_action[i] = Greedy_action[i] + cell_number * self.max_UE
                Greedy_action = Greedy_action.flatten()
                
                
                Greedy_idx = np.zeros((self.num_BS,self.total_UE))
                Greedy_idx[range(self.num_BS),Greedy_action] = 1
                Greedy_idx = Greedy_idx.transpose()
                
                Greedy_number = np.sum(Greedy_idx,axis=1,dtype=np.int8)
                Greedy_number = Greedy_number.reshape(self.total_UE,1)
                
                Greedy_idx = Greedy_idx.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                Greedy_main_path = H2 * Greedy_idx
                Greedy_main_path = Greedy_main_path * np.conj(MR_precoder)
                #Greedy_main_path = np.sum(abs(Greedy_main_path)**2,axis=-1) * path_loss
                Greedy_main_path = np.linalg.norm(Greedy_main_path,axis=-1)**2 * path_loss
                Greedy_main_path = np.dot(Greedy_main_path, equal_power)
                
                Greedy_inter_idx = (Greedy_idx - 1) * (-1)
                Greedy_inter_path = H2 * Greedy_inter_idx
                Greedy_inter_path = Greedy_inter_path * np.conj(MR_precoder)
                #Greedy_inter_path = np.sum(abs(Greedy_inter_path)**2,axis=-1) * path_loss
                Greedy_inter_path = np.linalg.norm(Greedy_inter_path,axis=-1)**2 * path_loss
                Greedy_inter_path = np.dot(Greedy_inter_path, equal_power)
                
                Greedy_sinr = Greedy_main_path / (Greedy_inter_path + self.n_power_W)
                
                Greedy_data_rate = np.log2(1. + Greedy_sinr)
                                
                Greedy_nor_rate = np.log2(1. + Greedy_sinr) / Greedy_number
                Greedy_nor_rate[ Greedy_number==0 ] = 0
                
                
                '--------- Max-P + No CoMP ------'
                
                
                '--------------------- DQN with PA + User Association --------------------------'
                
                DQN_max_action = []
                DQN_max_action = PA_UA_agent.act(DQN_input_state, self.num_BS)
                
                UA_action = np.floor( DQN_max_action/PA_action_num).astype(np.int32)
                PA_action = np.array( DQN_max_action%PA_action_num )
                                
                UE_candidate = np.array(UE_candidate)
                UA_action = UE_candidate[range(self.num_BS),UA_action]  #change action from max-min UE to corresponding UE_idx
                
                
                MAX_E = 4000
                epsilon = INITIAL_EPSILON - episode * (INITIAL_EPSILON-FINAL_EPSILON)/ MAX_E  # decade epsilon
                UA_random_index = np.array(np.random.uniform(size=self.num_BS) < epsilon, dtype = np.int32)
                UA_random_action = list()
                for i in range(self.num_BS):
                    UA_random_sample = choice(UE_candidate[i])  # network UE's indx
                    UA_random_action.append(UA_random_sample)
                
                UA_action_set = np.vstack([UA_action, UA_random_action]) #沿著直方向將矩陣堆疊起來。
                UA_action = UA_action_set[UA_random_index,range(self.num_BS)]                
                
                PA_random_index = np.array(np.random.uniform(size=self.num_BS) < epsilon, dtype = np.int32)
                PA_random_action = np.random.randint(0, high = PA_action_num, size = (self.num_BS))  # M = total_UE
                PA_action_set = np.vstack([PA_action, PA_random_action]) #沿著直方向將矩陣堆疊起來。
                power_index = PA_action_set[PA_random_index,range(self.num_BS)]                
                DQN_power = power_set[power_index]
                
                '------'
                
                DQN_serving_idx    = np.zeros((self.num_BS,self.total_UE))
                DQN_intference_idx = np.zeros((self.num_BS,self.total_UE))
                DQN_serving_idx[range(self.num_BS), UA_action] = 1
                DQN_serving_idx    = DQN_serving_idx.transpose()
                
                
                DQN_serving_number = np.sum(DQN_serving_idx,axis=1)
                DQN_serving_number = DQN_serving_number.reshape(self.total_UE,1)
                
                DQN_intference_idx     = np.array(( DQN_serving_idx - 1 ) * (-1))
                DQN_servering_indicate = np.array(DQN_serving_idx)
                
                DQN_power = DQN_power.reshape((self.num_BS,1))
                
                DQN_serving_idx_expand = DQN_serving_idx.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                DQN_main_path = ( H2 * DQN_serving_idx_expand ) * np.conj(MR_precoder)
                DQN_main_path =  np.linalg.norm(DQN_main_path,axis=-1)**2 * path_loss
                DQN_main_path = np.dot(DQN_main_path, DQN_power)
                
                DQN_intference_idx_expand = ( DQN_serving_idx_expand - 1 ) * (-1)
                DQN_inter_path = ( H2 * DQN_intference_idx_expand ) * np.conj(MR_precoder)
                DQN_inter_path = np.linalg.norm(DQN_inter_path,axis=-1)**2 * path_loss
                DQN_inter_path = np.dot(DQN_inter_path, DQN_power)
                
                DQN_sinr = DQN_main_path / (DQN_inter_path + self.n_power_W)
                DQN_data_rate = np.log2(1. + DQN_sinr)
                DQN_nor_rate  = np.log2(1. + DQN_sinr) / DQN_serving_number
                DQN_nor_rate[DQN_serving_number==0] = 0
                                                
                '------'
                '----- DQN:PA+UA 計算 BS k 對非serving UE 的干擾大小 -----'
                
                DQN_re_inter = np.tile(DQN_intference_idx,[self.num_BS,1,1])
                DQN_re_inter[range(self.num_BS),:,range(self.num_BS)] = 0
                DQN_re_inter_expand = DQN_re_inter.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
                
                DQN_re_inter_path = (H2 * DQN_re_inter_expand) * np.conj(MR_precoder)
                DQN_re_inter_path = np.linalg.norm(DQN_re_inter_path,axis=-1)**2 * path_loss
                DQN_re_inter_path = np.dot(DQN_re_inter_path, DQN_power)
                
                #c_re_sinr = np.minimum( main_path / (re_inter_path + self.n_power_W) , 1000)
                DQN_re_sinr = DQN_main_path / (DQN_re_inter_path + self.n_power_W)
                DQN_re_Comp_num = np.tile(DQN_serving_number,[self.num_BS,1,1])
                
                DQN_re_rate     =   np.log2(1. + DQN_re_sinr)
                DQN_re_nor_rate = ( np.log2(1. + DQN_re_sinr) / DQN_re_Comp_num ) 
                DQN_re_nor_rate[ DQN_re_Comp_num==0 ] = 0
                
                DQN_re =  DQN_re_nor_rate - DQN_nor_rate
                #DQN_re =  DQN_re_rate - DQN_data_rate
                DQN_re_sum = np.sum(DQN_re,axis=1)
                
                DQN_reward = DQN_nor_rate[UA_action] - DQN_re_sum
                
                
                '----- DQN:PA+UA 計算 BS k 對非serving UE 的干擾大小 -----'
                
                
                max_action = []
                Scheduling_DQN_t1 = time.time()
                max_action = Clustering_agent.act(clustering_input_state, self.num_BS)
                Scheduling_DQN_t2 = time.time()
                                
                UE_candidate = np.array(UE_candidate)
                max_action = UE_candidate[range(self.num_BS),max_action]  #change action from max-min UE to corresponding UE_idx
                
                epsilon = 0#INITIAL_EPSILON - episode * (INITIAL_EPSILON-FINAL_EPSILON)/ max_episode  # decade epsilon
                random_index = np.array(np.random.uniform(size=self.num_BS) < epsilon, dtype = np.int32)
                clustering_random_action = list()
                for i in range(self.num_BS):
                    random_sample = choice(UE_candidate[i])
                    clustering_random_action.append(random_sample)
                #clustering_random_action = ga_action2
                
                action_set = np.vstack([max_action, clustering_random_action]) #沿著直方向將矩陣堆疊起來。
                Comp_action = action_set[random_index,range(self.num_BS)]
                
                #change action from UE_idx to corresponding max-min_UE_idx
                UE_candidate = UE_candidate.tolist()
                clustering_action = list()
                for i in range(self.num_BS):
                    correspond_idx = UE_candidate[i].index(Comp_action[i])
                    clustering_action.append(correspond_idx)
                clustering_action = np.array(clustering_action)
                
                
                Comp_index         = np.zeros((self.num_BS,self.total_UE))
                interference_index = np.zeros((self.num_BS,self.total_UE))
                Comp_index[range(self.num_BS),Comp_action] = 1
                Comp_index = Comp_index.transpose()
                Comp_number = np.sum(Comp_index,axis=1)
                Comp_number = Comp_number.reshape(self.total_UE,1)
                
                interference_index = np.array(( Comp_index - 1 ) * (-1))
                Comp_indicate      = np.array(Comp_index)
                
                #PA_power = DQN_power.reshape((self.num_BS,1))
                c_power  = np.array(PA_power).reshape((self.num_BS,1))
                Comp_index = Comp_index.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                c_main_path = H2 * Comp_index
                c_main_path = c_main_path * np.conj(MR_precoder)
                #c_main_path = np.sum(abs(c_main_path)**2,axis=-1) * path_loss
                c_main_path = np.linalg.norm(c_main_path,axis=-1)**2 * path_loss
                cm_path = np.array(c_main_path)
                c_main_path = np.dot(c_main_path, c_power)
                
                inter_index = ( Comp_index - 1 ) * (-1)
                c_inter_path = H2 * inter_index
                c_inter_path = c_inter_path * np.conj(MR_precoder)
                #c_inter_path = np.sum(abs(c_inter_path)**2,axis=-1) * path_loss
                c_inter_path = np.linalg.norm(c_inter_path,axis=-1)**2 * path_loss
                ci_path = np.array(c_inter_path)
                c_inter_path = np.dot(c_inter_path, c_power)
                
                c_min_sinr = np.minimum( c_main_path / (c_inter_path + self.n_power_W) , 1000) # capped sinr max 30dB
                c_sinr = c_main_path / (c_inter_path + self.n_power_W)
                
                c_data_rate = np.log2(1. + c_sinr)
                c_nor_rate = np.log2(1. + c_sinr) / Comp_number
                c_nor_rate[ Comp_number==0 ] = 0
                min_c_nor_rate = np.log2(1. + c_min_sinr) / Comp_number
                min_c_nor_rate [ Comp_number==0 ] = 0
                
                '--- (End) Randomly Clustering ----'
                '---- (Start) Calculate clustering reward ----'
                '----- 移除 BS k original serving UE  --------'
                
                Comp_rre = np.zeros((self.num_BS,self.num_BS,self.total_UE))
                Comp_rre[:,range(self.num_BS),Comp_action] = 1
                Comp_rre[range(self.num_BS),range(self.num_BS),Comp_action] = 0                
                Comp_rre = Comp_rre.transpose(0,2,1)
                rre_Comp_number = Comp_rre.sum(2)
                rre_Comp_number = rre_Comp_number.reshape(self.num_BS,self.total_UE,1)
                
                Comp_rre = Comp_rre.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
                rre_main_path = H2 * Comp_rre
                rre_main_path = rre_main_path * np.conj(MR_precoder)
                #rre_main_path = np.sum(abs(rre_main_path)**2,axis=-1) * path_loss
                rre_main_path = np.linalg.norm(rre_main_path,axis=-1)**2 * path_loss
                rre_main_path = np.dot(rre_main_path, c_power)
                
                inter_rre = (Comp_rre - 1) * (-1)
                rre_inter_path = H2 * inter_rre
                rre_inter_path = rre_inter_path * np.conj(MR_precoder)
                #rre_inter_path = np.sum(abs(rre_inter_path)**2,axis=-1) * path_loss
                rre_inter_path = np.linalg.norm(rre_inter_path,axis=-1)**2 * path_loss
                rre_inter_path = np.dot(rre_inter_path, c_power)
                
                rre_sinr = rre_main_path / (rre_inter_path + self.n_power_W)
                rre_nor_rate = np.log2(1. + rre_sinr) / rre_Comp_number
                rre_nor_rate[rre_Comp_number == 0] = 0   # remove BSk serving UE's nor-rate.
                
                '----- 移除 BS k original serving UE  --------'
                '----- 計算 BS k 對非serving UE 的干擾大小 -----'
                
                c_re_inter = np.tile(interference_index,[self.num_BS,1,1])
                c_re_inter[range(self.num_BS),:,range(self.num_BS)] = 0
                c_re_inter = c_re_inter.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
                
                c_re_inter_path = H2 * c_re_inter
                c_re_inter_path = c_re_inter_path * np.conj(MR_precoder)
                #c_re_inter_path = np.sum(abs(c_re_inter_path)**2,axis=-1) * path_loss
                c_re_inter_path = np.linalg.norm(c_re_inter_path,axis=-1)**2 * path_loss
                ci_re_path = np.array(c_re_inter_path)
                c_re_inter_path = np.dot(c_re_inter_path, c_power)
                
                #c_re_sinr = np.minimum( main_path / (re_inter_path + self.n_power_W) , 1000)
                c_re_sinr = c_main_path / (c_re_inter_path + self.n_power_W)
                c_re_Comp_num = np.tile(Comp_number,[self.num_BS,1,1])
                c_re_nor_rate = ( np.log2(1. + c_re_sinr) / c_re_Comp_num ) 
                c_re_nor_rate[ c_re_Comp_num==0 ] = 0
                c_re =  c_re_nor_rate - c_nor_rate
                c_re_sum = np.sum(c_re,axis=1)
                
                
                '----- 計算 BS k 對非serving UE 的干擾大小 -----'
                '--------- 計算 BS k 服務非serving UE造成的nor rate 的增減 ---------'
                
                comp_other_indx = np.zeros((self.num_BS,self.num_BS,self.total_UE))
                comp_other_indx[:,range(self.num_BS),Comp_action] = 1
                comp_other_indx = comp_other_indx.transpose(0,2,1)
                for i in range(self.num_BS):
                    cell_number = int(i/self.cell_BS)
                    comp_other_indx[i, cell_number*self.max_UE : (cell_number+1)*self.max_UE, i] = 1
                comp_other_indx[range(self.num_BS),:,range(self.num_BS)] = 1
                comp_other_number = comp_other_indx.sum(2)
                comp_other_number = comp_other_number.reshape(self.num_BS,self.total_UE,1)
                
                comp_other_indx = comp_other_indx.repeat(self.BS_antenna).reshape(self.num_BS,self.total_UE,self.num_BS,self.BS_antenna)
                other_main_path = H2 * comp_other_indx
                other_main_path = other_main_path * np.conj(MR_precoder)
                #other_main_path = np.sum(abs(other_main_path)**2,axis=-1) * path_loss
                other_main_path = np.linalg.norm(other_main_path,axis=-1)**2 * path_loss
                other_main_path = np.dot(other_main_path, c_power)
                
                comp_inter_other_indx = ( comp_other_indx - 1 ) * (-1)
                other_inter_path = H2 *comp_inter_other_indx
                other_inter_path = other_inter_path * np.conj(MR_precoder)
                #other_inter_path = np.sum(abs(other_inter_path)**2,axis=-1) * path_loss
                other_inter_path = np.linalg.norm(other_inter_path,axis=-1)**2 * path_loss
                other_inter_path = np.dot(other_inter_path, c_power)
                
                other_sinr = other_main_path / (other_inter_path + self.n_power_W)
                other_nor_rate = np.log2(1. + other_sinr) / comp_other_number
                other_nor_rate[comp_other_number == 0] = 0
                other_nor_rate_re = np.array(other_nor_rate)
                
                other_rate = []
                for i in range(self.num_BS):
                    temp = np.delete(c_nor_rate, Comp_action[i])
                    other_rate.append(temp)
                other_rate = np.array(other_rate).reshape(self.num_BS,self.total_UE-1,1)
                
                other_nor_rate = []
                for i in range(self.num_BS):
                    temp = np.delete(other_nor_rate_re[i], Comp_action[i])
                    other_nor_rate.append(temp)
                other_nor_rate = np.array(other_nor_rate).reshape(self.num_BS,self.total_UE-1,1)
            
                term1 = np.tile((c_nor_rate[Comp_action] - rre_nor_rate[range(self.num_BS),Comp_action]),(self.total_UE-1,1,1)).transpose()
                term1 = term1.reshape(self.num_BS, self.total_UE-1, 1)
                term2 =  other_rate - other_nor_rate #-1 * other_nor_rate#other_rate - other_nor_rate # -1 * other_nor_rate
                temp  = np.array( term1 + term2)
                
                rewardA =[]
                for i in range(self.num_BS):
                    cell_number = int(i/self.cell_BS)
                    reward = temp[i][cell_number * self.max_UE : (cell_number+1) * self.max_UE - 1]
                    reward = np.min(temp[i][cell_number * self.max_UE : (cell_number+1) * self.max_UE - 1], axis = 0)
                    rewardA.append(reward)
                rewardA = np.array(rewardA).reshape(self.num_BS,1)
                serving_rate =  rewardA
                c_re_sum = c_re_sum
                #clustering_reward = np.tile( np.mean(c_nor_rate),(self.num_BS,1)) + serving_rate   # 6.527 gap:0.4
                clustering_reward = c_nor_rate[Comp_action] - c_re_sum #serving_rate #np.tile(np.mean(c_nor_rate[Comp_action]),(self.num_BS,1) ) #
                
                '---- (End) calculate clustering reward ----' 
                
                '--- Gernerate PA next state'
                
                current_PA_main_channel = []
                current_PA_interference = []
                current_repeat_matrix   = []
                current_interfered_channel = []
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS )  # 1cell 3BS 
                    cell_BS_idx = int( i % self.cell_BS )
                    
                    intra_interference = np.array(h_gain[Comp_action[i],cell_number * self.cell_BS : (cell_number+1)*self.cell_BS])
                    intra_interference = np.roll(intra_interference, -1 * cell_BS_idx)
                    main_channel = intra_interference[0]
                    intra_interference = intra_interference[1:]
                    inter_interference = np.array(h_gain[Comp_action[i],:])
                    inter_interference = np.delete(inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                    max_inter_idx = heapq.nlargest(PA_highest_num, range(len(inter_interference)), inter_interference.take)
                    inter_interference = inter_interference[max_inter_idx]
                    
                    interfered_link = h_gain[Comp_action,i]
                    interfered_link[interfered_link==h_gain[Comp_action[i],i]] = 0
                    max_interfered_idx = heapq.nlargest( PA_highest_num, range(len(interfered_link)), interfered_link.take)
                    interfered_link = interfered_link[max_interfered_idx]
                    
                    interfernce = np.hstack((intra_interference, inter_interference))
                    PA_channel     = np.hstack((main_channel, interfernce,interfered_link))
                    h_mean = np.mean(PA_channel)
                    h_std  = np.std(PA_channel)
                    main_channel    = (main_channel - h_mean) / h_std
                    interfernce     = (interfernce - h_mean) / h_std
                    interfered_link = (interfered_link - h_mean) / h_std
                    
                    current_PA_main_channel.append(main_channel)
                    current_PA_interference.append(interfernce)
                    current_interfered_channel.append(interfered_link)
                    
                    repeat = Comp_indicate[Comp_action[i]]
                    repeat = repeat[cell_number * self.cell_BS : (cell_number+1) * self.cell_BS]
                    repeat = np.roll(repeat,  -1 * cell_BS_idx)
                    current_repeat_matrix.append(repeat)
                
                current_PA_main_channel = np.array(current_PA_main_channel).reshape(self.num_BS,1)
                current_PA_interference = np.array(current_PA_interference)
                current_repeat_matrix   = np.array(current_repeat_matrix)
                current_interfered_channel = np.array(current_interfered_channel)
                
                #PA_next_state = np.hstack((previous_PA_main_channel, previous_PA_interference, previous_interfered_channel, previous_repeat_matrix, current_PA_main_channel, current_PA_interference, current_interfered_channel, current_repeat_matrix, power_matrix, rate_matrix))                
                PA_next_state = np.hstack((previous_PA_main_channel, previous_PA_interference, previous_PA_inter_power, previous_interfered_channel, previous_repeat_matrix, current_PA_main_channel, current_PA_interference, current_interfered_channel, current_repeat_matrix, power_matrix, rate_matrix))                
                
                if time_s > 0:
                    PA_agent.remember(PA_input_state, PA_action, PA_reward, PA_next_state)
                
                PA_input_state = PA_next_state
                
                '--- Gernerate PA next state'
                '-------- (Start) DQN-PA ---------'
                   
                Comp_number = Comp_number.flatten()#np.sum(Comp_index,axis=1)
                max_action = []
                
                PA_DQN_t1 = time.time()
                max_action = PA_agent.act(PA_input_state, self.num_BS)
                epsilon = 0#INITIAL_EPSILON - episode * (INITIAL_EPSILON-FINAL_EPSILON)/ max_episode  # decade epsilon
                PA_DQN_t2 = time.time()
                
                random_index = np.array(np.random.uniform(size=self.num_BS) < epsilon, dtype = np.int32)
                PA_random_action = np.random.randint(0, high = PA_action_num, size = (self.num_BS))  # M = total_UE
                action_set = np.vstack([max_action, PA_random_action]) #沿著直方向將矩陣堆疊起來。
                power_index = action_set[random_index,range(self.num_BS)]
                #power_index = (PA_action_num-1) * np.ones((1,self.num_BS),dtype=np.int8).flatten()
                
                PA_power = power_set[power_index]#DQN_power.flatten()
                #PA_power = DQN_power.flatten()
                
                PA_main_path  = np.dot(cm_path, PA_power)
                PA_inter_path = np.dot(ci_path, PA_power)
                min_sinr = np.minimum( PA_main_path / (PA_inter_path + self.n_power_W) , 1000) # capped sinr max 30dB
                sinr = PA_main_path / (PA_inter_path + self.n_power_W)            
                PA_nor_rate = np.log2(1. + sinr) / Comp_number
                PA_nor_rate[ Comp_number==0 ] = 0   #分母是0時，得到 0
                
                PA_data_rate = np.log2(1. + sinr)
                
                min_nor_rate = np.log2(1. + min_sinr) / Comp_number
                min_nor_rate [ Comp_number==0 ] = 0
                
                
                '-------- (End) DQN-PA  ---------'
                '---- (Start) Calculate DQN-PA reward ----'
                
                pa_re_inter_path = np.dot(ci_re_path, PA_power)
                #pa_re_sinr = np.minimum( main_path / (re_inter_path + self.n_power_W) , 1000)
                pa_re_sinr = PA_main_path / (pa_re_inter_path + self.n_power_W)
                pa_re_Comp_num = np.tile(Comp_number,[self.num_BS,1])
                PA_re_nor_rate = ( np.log2(1. + pa_re_sinr) / pa_re_Comp_num ) 
                PA_re_nor_rate[ pa_re_Comp_num==0 ] = 0
                PA_re =  PA_re_nor_rate - PA_nor_rate
                PA_re_sum = np.sum(PA_re,axis=1)
                PA_reward = PA_nor_rate[Comp_action] - PA_re_sum                
                
                '---- (End) Calculate DQN-PA reward ----'
                
                
                '---- WMMSE + No CoMP ------'
                
                
                mmse_t1 = time.time()
                max_wmmse_t = 100
                WMMSE_action  = np.array(Best_action.flatten())
                hkk = np.sqrt( h_gain[WMMSE_action,range(self.num_BS)] )
                v =  np.random.uniform( 0, np.sqrt(self.max_power_W), size = (self.num_BS,1)).flatten()
                
                #u = ((hkk * v) / ((np.dot(h_gain[WMMSE_action, :],v**2)) + self.n_power_W))

                u = ((hkk * v) / ((np.dot(h_gain[WMMSE_action, :],v**2)) + self.n_power_W))
                w =  1. / (1 - u * hkk * v)
                C = np.sum(w)
                
                for wmmse_t in range(max_wmmse_t):
                    C_last = C
                    v = (hkk*u*w) / ((np.dot(h_gain[WMMSE_action, :] * (u**2)  ,v))**2)
                    v = np.minimum(np.sqrt(self.max_power_W), np.maximum(0, v))
                    u = (hkk * v) / ((np.dot(h_gain[WMMSE_action, :],v**2)) + self.n_power_W)
                    w = 1. / (1. - u * hkk * v)
                    C = np.sum(w)     
                    if np.abs(C_last - C) < 1e-3:
                        break
                p_mmse = v**2
                
                p_mmse = np.array(p_mmse).reshape((self.num_BS,1))
                mmse_action = np.array(WMMSE_action)
                mmse_idx = np.zeros((self.total_UE,self.num_BS))
                mmse_idx[mmse_action,range(self.num_BS)] = 1
                mmse_number = np.sum(mmse_idx, axis=1,dtype=np.int8)
                mmse_number = mmse_number.reshape(self.total_UE,1)
                
                mmse_idx = mmse_idx.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                mmse_main_path = H2 * mmse_idx
                mmse_main_path = mmse_main_path * np.conj(MR_precoder)
                mmse_main_path = np.linalg.norm(mmse_main_path,axis=-1)**2 * path_loss
                mmse_main_path = np.dot(mmse_main_path, p_mmse)
                
                mmse_inter_idx = (mmse_idx - 1) * (-1)
                mmse_inter_path = H2 * mmse_inter_idx
                mmse_inter_path = mmse_inter_path * np.conj(MR_precoder)
                mmse_inter_path = np.linalg.norm(mmse_inter_path,axis=-1)**2 * path_loss
                mmse_inter_path = np.dot(mmse_inter_path, p_mmse)
                
                mmse_sinr = mmse_main_path / (mmse_inter_path + self.n_power_W)
                
                
                mmse_data_rate = np.log2(1. + mmse_sinr)
                mmse_nor_rate  = np.log2(1. + mmse_sinr) / mmse_number
                mmse_nor_rate[ mmse_number==0 ] = 0
                
                
                mmse_t2 = time.time()
                
                '---- WMMSE + NC ------'
                
                
                
                '---- (Start) Gernerate clustering next step ---'
                
                '---- PA INPUT ---'
                previous_PA_main_channel = np.array( current_PA_main_channel )
                previous_PA_interference = np.array( current_PA_interference )
                previous_repeat_matrix =   np.array( current_repeat_matrix )
                previous_PA_inter_power = []
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS )  # 1cell 3BS 
                    cell_BS_idx = int( i % self.cell_BS )
                    PA_inter_interference = np.array(h_gain[Comp_action[i],:])
                    PA_inter_interference = np.delete(PA_inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                    max_inter_idx = heapq.nlargest( PA_highest_num, range(len(PA_inter_interference)), PA_inter_interference.take)
                    previous_PA_inter_power.append(PA_power[max_inter_idx])
                previous_PA_inter_power = np.array(previous_PA_inter_power)
                previous_PA_inter_power = (previous_PA_inter_power - mean_power) / std_power
                '---- PA INPUT ---'
                
                '--- Clustering previous input with scale -----'  
                previous_BS_input       = np.array(current_BS_input)
                previous_UA_PA_BS_input = np.array(current_UA_PA_BS_input)
                
                previous_repeat_candidate = np.array(current_repeat_candidate)
                
                previous_power       = np.array(current_power)
                previous_UA_PA_power = np.array(current_UA_PA_power)
                '--- Clustering previous input with scale -----'

                '----- next time slot ----'
                
                count += 1
                H2 = np.array(H_set[count,:,:,:])
                h_gain = np.array(channel_gain[count,:,:])
                MR_precoder = np.array(MRT_precoding[count,:,:,:])
                
                UE_candidate = list()
                idx_array = np.zeros((self.num_BS,Clustering_action_num), dtype = np.int32)          
                for i in range(self.num_BS):
                    cell_number = int( i / 3 )  # 1cell 3BS
                    heap = list(h_gain[cell_number * self.max_UE : ((cell_number + 1) * self.max_UE),i])
                    UE_candidate.append(list(map(heap.index , heapq.nlargest(Clustering_action_num, heap)))   )
                    idx_array[i,:] = cell_number * self.max_UE
                idx_array = np.array(idx_array)
                UE_candidate = np.array(UE_candidate)
                UE_candidate_idx = np.array(UE_candidate)
                UE_candidate = UE_candidate + idx_array
                UE_candidate = UE_candidate.tolist()
                
                '----- next time slot ----'
                
                #temp = np.array(h_gain * np.tile(PA_power, (self.total_UE,1)).reshape(self.total_UE,self.num_BS ))
                temp = np.array(h_gain)
                current_channel = []
                for i in range(self.num_BS):
                    current_channel.append( temp[UE_candidate[i],i] )
                current_channel = np.array(current_channel)
                
                current_main_channel = []
                for i in range(self.num_BS):
                    cell_num    = int( i / self.cell_BS)
                    cell_BS_idx = int( i % self.cell_BS )
                    temp1     = current_channel[cell_num * self.cell_BS : (cell_num+1) * self.cell_BS]   #[[BS1_max,BS1_sec,BS1_third][BS2_max,BS2_sec...][...]]
                    flag = np.roll(temp1, -1 * cell_BS_idx * self.max_UE  )
                    current_main_channel.append(flag.flatten())
                current_main_channel = np.array(current_main_channel)
                
                #UE_interference       = []  #include intra and 5 max inter interference
                UE_inter_interference = []
                UE_intra_interference = []
                current_inter_power       = []
                current_UA_PA_inter_power = []
                for i in range(self.total_UE):
                    cell_number = int(i / self.max_UE)
                    intra_interference = np.array( temp[i,cell_number*self.cell_BS : (cell_number+1)*self.cell_BS])
                    inter_interference = np.array( temp[i,:])
                    inter_interference = np.delete(inter_interference ,range(cell_number*self.cell_BS , (cell_number+1)*self.cell_BS))
                    max_inter_idx = heapq.nlargest(scheduling_highest_num, range(len(inter_interference)), inter_interference.take)
                    inter_interference = inter_interference[max_inter_idx]
                    inter_power        = PA_power[max_inter_idx]
                    UA_PA_inter_power  = DQN_power[max_inter_idx]
                    
                    UE_inter_interference.append(inter_interference)
                    UE_intra_interference.append(intra_interference)
                    current_inter_power.append(inter_power)
                    current_UA_PA_inter_power.append(UA_PA_inter_power)
                    
                UE_inter_interference = np.array(UE_inter_interference)
                UE_intra_interference = np.array(UE_intra_interference)
                current_inter_power  = np.array(current_inter_power)
                current_inter_power  =  current_inter_power / self.max_power_W#(current_inter_power - mean_power) / std_power
                current_UA_PA_inter_power = np.array(current_UA_PA_inter_power)
                current_UA_PA_inter_power = current_UA_PA_inter_power / self.max_power_W
                
                current_BS_input = []
                current_UA_PA_BS_input = []
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS)
                    cell_BS_idx = int( i % self.cell_BS )
                    intra = np.roll(UE_intra_interference[cell_number* self.max_UE : (cell_number+1)* self.max_UE], -1 * cell_BS_idx, axis=1)
                    feature1 = np.hstack((current_main_channel[i], intra[UE_candidate_idx[i]].flatten(), UE_inter_interference[UE_candidate[i]].flatten()))
                    g_mean    = np.mean(feature1)
                    g_std     = np.std(feature1)
                    feature1 = ( feature1 - g_mean ) / g_std
                    feature2 = current_inter_power[UE_candidate[i]].flatten()
                    UA_PA_feature2 =  current_UA_PA_inter_power[UE_candidate[i]].flatten()
                    
                    
                    current_BS_input.append(np.hstack((feature1,feature2)))
                    current_UA_PA_BS_input.append(np.hstack((feature1,UA_PA_feature2)))
                current_BS_input = np.array(current_BS_input)
                current_UA_PA_BS_input = np.array(current_UA_PA_BS_input)
                
                
                UE_candidate = np.array(UE_candidate)            
                current_repeat_candidate = []
                for k in range(self.num_BS):
                    for i in range(Clustering_action_num):
                        BS_reapet = np.zeros((self.num_BS, Clustering_action_num))
                        repeat = np.where(UE_candidate == UE_candidate[k][i])
                        BS_reapet[repeat[0],repeat[1]] = 1
                        cell_num = int (k / self.cell_BS)
                        idx          = cell_num*self.cell_BS
                        cell_num     = int(k / self.cell_BS)
                        cell_BS_idx  = int(k % self.cell_BS)   #corrseponding label of BS for each cell
                        if cell_BS_idx == 0:
                            BS_reapet = BS_reapet[[1 + idx ,2 + idx],:].flatten()
                            current_repeat_candidate.append(BS_reapet)
                        if cell_BS_idx == 1:
                            BS_reapet = BS_reapet[[2 + idx ,0 + idx ],:].flatten()
                            current_repeat_candidate.append(BS_reapet)
                        if cell_BS_idx == 2:
                            BS_reapet = BS_reapet[[0 + idx ,1 + idx],:].flatten()
                            current_repeat_candidate.append(BS_reapet)   
                current_repeat_candidate = np.array(current_repeat_candidate).reshape(self.num_BS, (self.cell_BS -1) * Clustering_action_num * Clustering_action_num )
                
                cur_power       = np.array(PA_power).reshape(self.num_BS,1)
                UA_PA_cur_power = np.array(DQN_power).reshape(self.num_BS,1)
                
                current_power = []
                current_UA_PA_power =[]
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS)
                    cell_BS_idx = int( i % self.cell_BS )
                    temp1 = cur_power[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                    temp2 = np.roll(temp1 , -1 * cell_BS_idx)
                    current_power.append(temp2)
                    
                    temp1 = UA_PA_cur_power[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                    temp2 = np.roll(temp1 , -1 * cell_BS_idx)
                    current_UA_PA_power.append(temp2)
                    
                current_power = np.array(current_power).reshape(self.num_BS,self.cell_BS)
                power_matrix  = np.array ( (current_power - mean_power) / std_power )
                current_power = current_power / self.max_power_W #(current_power - mean_power) / std_power
                
                current_UA_PA_power = np.array(current_UA_PA_power).reshape(self.num_BS,self.cell_BS)
                current_UA_PA_power = current_UA_PA_power / self.max_power_W
                
                ' --- Clustering previous input without scale -----'
                
                
                clustering_next_state = np.hstack(( previous_BS_input, previous_repeat_candidate, previous_power, current_BS_input, current_repeat_candidate, current_power))
                
                DQN_next_state = np.hstack(( previous_UA_PA_BS_input, previous_repeat_candidate, previous_UA_PA_power, current_UA_PA_BS_input, current_repeat_candidate, current_UA_PA_power))
                
                '---- (End) Gernerate clustering next step ---'

                '---- (Start) Gernerate PA next step ---'
                
                rate = np.array(PA_nor_rate[Comp_action]).reshape(self.num_BS,1)
                rate_matrix = []
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS)
                    cell_BS_idx = int( i % self.cell_BS )
                    temp1 = rate[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                    temp2 = np.roll(temp1 , -cell_BS_idx)
                    rate_matrix.append(temp2)
                rate_matrix = np.array(rate_matrix).reshape(self.num_BS,self.cell_BS)
                
                reward = np.array(PA_reward).reshape(self.num_BS,1)
                input_reward = []
                for i in range(self.num_BS):
                    cell_number = int( i / self.cell_BS)
                    cell_BS_idx = int( i % self.cell_BS )
                    temp1 = reward[cell_number* self.cell_BS : (cell_number+1) * self.cell_BS , :]
                    temp2 = np.roll(temp1 , -cell_BS_idx)
                    input_reward.append(temp2)
                input_reward = np.array(input_reward).reshape(self.num_BS,self.cell_BS)
                
                '-------------------------------------- DQN-UAPA ------------------------------'
                #Clustering_agent.remember(clustering_input_state, clustering_action, clustering_reward, clustering_next_state)
                PA_UA_agent.remember(clustering_input_state, DQN_max_action, DQN_reward, clustering_next_state)
                clustering_input_state = clustering_next_state
                DQN_input_state        = DQN_next_state
                
                
                '---- (End) Gernerate PA next step ---'
                
                '-------------------------- PLOT ------------------------'
                
                GA_data_rate = np.array( mmse_data_rate )
                GA_nor_rate = np.array( mmse_nor_rate)
                GA_power    = np.array( p_mmse )
                GA_t1 = 0
                GA_t2 = 0
                GA_number = np.array( GA_number )
                
                '-----Average Data rate -----'
                
                Joint_DQN_sum_data_rate_list.append( np.mean(DQN_data_rate) )
                Separate_DQN_sum_data_rate_list.append( np.mean(PA_data_rate) )
                
                Clustering_sum_data_rate_list.append( np.mean(DQN_data_rate) )
                GA_sum_data_rate_list.append( np.mean(GA_data_rate) )
                Greedy_sum_data_rate_list.append( np.mean(Greedy_data_rate) )
                mmse_sum_data_rate_list.append( np.mean(mmse_data_rate) )
                
                '-----Average Data rate -----'
                
                '---- Average rate to All Algorithm ---'
                
                Joint_DQN_sum_rate    = np.mean(DQN_nor_rate)
                Separate_DQN_sum_rate = np.mean(PA_nor_rate)
                
                DQN_sum_rate        = np.mean(DQN_nor_rate)
                PA_sum_rate         = np.mean(DQN_nor_rate)
                Greedy_sum_rate     = np.mean(Greedy_nor_rate)
                mmse_sum_rate       = np.mean(mmse_nor_rate)
                Max_P_GA_sum_rate   = np.mean(GA_nor_rate)
                #PA_GA_sum_rate      = np.mean(PA_GA_nor_rate)
                
                
                Joint_DQN_sum_rate_list.append(Joint_DQN_sum_rate)
                Separate_DQN_sum_rate_list.append(Separate_DQN_sum_rate)
                PA_sum_rate_list.append(PA_sum_rate)
                Clustering_sum_rate_list.append(DQN_sum_rate)
                Greedy_sum_rate_list.append(Greedy_sum_rate)
                mmse_sum_rate_list.append(mmse_sum_rate)
                GA_sum_rate_list.append(Max_P_GA_sum_rate)
                #PA_GA_sum_rate_list.append(PA_GA_sum_rate)
                '---- Average rate to All Algorithm ---'

                '---- PE to All Algorithm ---'
                #PA_GA_power_eff  = PA_GA_sum_rate / np.sum(PA_GA_power * 1e-3)
                Joint_DQN_power_eff     = Joint_DQN_sum_rate / np.sum( DQN_power )
                Separatet_DQN_power_eff = Separate_DQN_sum_rate / np.sum( PA_power )
                GA_power_eff  = Max_P_GA_sum_rate / np.sum(GA_power)
                DQN_power_eff = DQN_sum_rate / np.sum(DQN_power )
                Greedy_power_eff  = Greedy_sum_rate / np.sum(equal_power)
                mmse_power_eff = mmse_sum_rate / np.sum(p_mmse)
                
                #PA_GA_PE_list.append(PA_GA_power_eff)
                Joint_DQN_PE_list.append(Joint_DQN_power_eff)
                Separate_DQN_PE_list.append(Separatet_DQN_power_eff)
                GA_PE_list.append(GA_power_eff)
                DQN_PE_list.append(DQN_power_eff)
                Greedy_PE_list.append(Greedy_power_eff)
                mmse_PE_list.append(mmse_power_eff)
                '---- PE to All Algorithm ---'
                
                '---- SE to All Algorithm ---'
                #PA_GA_spectrum_eff  = PA_GA_sum_rate / np.sum(PA_GA_power * 1e-3)
                Joint_DQN_spectrum_eff     =  np.sum(DQN_data_rate) / self.num_BS
                Separatet_DQN_spectrum_eff =  np.sum(PA_data_rate) / self.num_BS
                GA_spectrum_eff      =  np.sum(GA_data_rate) / self.num_BS
                DQN_spectrum_eff     =  np.sum(DQN_data_rate) / self.num_BS
                Greedy_spectrum_eff  =  np.sum(Greedy_data_rate) / self.num_BS
                mmse_spectrum_eff    =  np.sum(mmse_nor_rate) / self.num_BS
                    
                Joint_DQN_SE_list.append(Joint_DQN_spectrum_eff)
                Separate_DQN_SE_list.append(Separatet_DQN_spectrum_eff)
                GA_SE_list.append(GA_spectrum_eff)
                DQN_SE_list.append(DQN_spectrum_eff)
                Greedy_SE_list.append(Greedy_spectrum_eff)
                mmse_SE_list.append(mmse_spectrum_eff)
                '---- SE to All Algorithm ---'
                
                '---- Average rate to all Link----'
                
                Joint_DQN_rate.append(np.array(DQN_data_rate).flatten())
                Separate_DQN_rate.append(np.array(PA_data_rate).flatten())
                DQN_rate.append(np.array(DQN_data_rate).flatten())
                Greedy_rate.append(np.array(Greedy_data_rate).flatten())
                GA_rate.append(np.array(GA_data_rate).flatten())
                #mmse_rate.append(np.array(mmse_data_rate).flatten())
                '---- Average rate to all Link----'
                
                
                
                '---- SE to Each Link --- ' 
                
                DQN_nor_rate = np.array(DQN_nor_rate[UA_action]).flatten()
                Joint_DQN_user.append(DQN_nor_rate)
                
                PA_nor_rate = np.array(PA_nor_rate[Comp_action]).flatten()
                Separate_DQN_user.append(PA_nor_rate)
                
                Greedy_nor_rate =np.array(Greedy_nor_rate[Greedy_action]).flatten()
                Greedy_user.append(Greedy_nor_rate)
                
# =============================================================================
#                 mmse_nor_rate =np.array(mmse_nor_rate[WMMSE_action]).flatten()
#                 mmse_user.append(mmse_nor_rate)
# =============================================================================
                
                GA_nor_rate = np.array(GA_nor_rate[Best_action]).flatten()
                ga_user.append(GA_nor_rate)
                '---- SE to Each Link --- ' 
                
                '----- CDF to each Link -----'
                
                Joint_DQN_cdf.append(DQN_nor_rate)
                Separate_DQN_cdf.append(PA_nor_rate)
                
                #DQN_cdf.append(DQN_nor_rate)
                #mmse_cdf.append(mmse_nor_rate)
                Greedy_cdf.append(Greedy_nor_rate)
                #PA_GA_cdf.append(PA_GA_nor_rate)
                GA_cdf.append(GA_nor_rate)
                
                '----- CDF to each Link -----'
                
                
                '-------------------------- PLOT ------------------------'

                
                #mmse_total_time = mmse_total_time+ (mmse_t2 - mmse_t1)
                Scheduling_DQN_total_time  = Scheduling_DQN_total_time + (Scheduling_DQN_t2  - Scheduling_DQN_t1)
                PA_DQN_total_time  = PA_DQN_total_time + (PA_DQN_t2  - PA_DQN_t1)
                GA_total_time  = GA_total_time + (GA_t2  - GA_t1)
                
                for i in range(self.total_UE):
                    Total_JT += 1 
                    if Comp_number[i] > 1:
                        DQN_JT += 1
                    if Greedy_number[i]>1:
                        Greedy_JT += 1
# =============================================================================
#                     if mmse_number[i]>1:
#                         mmse_JT += 1
# =============================================================================
                    if GA_number[i]>1:
                        GA_JT += 1
                        
# =============================================================================
#             if len(PA_agent.memory) > batch_size:
#                 PA_UA_agent.replay(batch_size)
#                 #Clustering_agent.replay(batch_size)
#                 #PA_agent.replay(batch_size)
# =============================================================================
            
            
            Joint_DQN_Average_sum_data_rate_list.append( np.mean(Joint_DQN_sum_data_rate_list) )
            Separate_DQN_Average_sum_data_rate_list.append( np.mean(Separate_DQN_sum_data_rate_list) )
            
            Clustering_Average_sum_data_rate_list.append( np.mean(Clustering_sum_data_rate_list) )
            Greedy_Average_sum_data_rate_list.append( np.mean(Greedy_sum_data_rate_list) )
            GA_Average_sum_data_rate_list.append( np.mean(GA_sum_data_rate_list) )
            #mmse_Average_sum_data_rate_list.append( np.mean(mmse_sum_data_rate_list) )
            
            
            Joint_DQN_Average_rate_list.append(np.mean(Joint_DQN_sum_rate_list))
            Separate_DQN_Average_rate_list.append(np.mean(Separate_DQN_sum_rate_list))
            DQN_Average_rate_list.append(np.mean(Clustering_sum_rate_list))
            Clustering_Average_rate_list.append(np.mean(Clustering_sum_rate_list))
            Greedy_Average_rate_list.append(np.mean(Greedy_sum_rate_list))
            #mmse_Average_rate_list.append(np.mean(mmse_sum_rate_list))
            GA_Average_rate_list.append(np.mean(GA_sum_rate_list))
            #PA_GA_Average_rate_list.append(np.mean(PA_GA_sum_rate_list))
            
            Joint_DQN_Average_PE_list.append(np.mean(Joint_DQN_PE_list))
            Separate_DQN_Average_PE_list.append(np.mean(Separate_DQN_PE_list))
            GA_Average_PE_list.append(np.mean(GA_PE_list))
            #PA_GA_Average_PE_list.append(np.mean(PA_GA_PE_list))
            DQN_Average_PE_list.append(np.mean(DQN_PE_list))
            Greedy_Average_PE_list.append(np.mean(Greedy_PE_list))
            #mmse_Average_PE_list.append(np.mean(mmse_PE_list))
            
            Joint_DQN_Average_SE_list.append(np.mean(Joint_DQN_SE_list))
            Separate_DQN_Average_SE_list.append(np.mean(Separate_DQN_SE_list))
            GA_Average_SE_list.append(np.mean(GA_SE_list))
            #PA_GA_Average_SE_list.append(np.mean(PA_GA_SE_list))
            DQN_Average_SE_list.append(np.mean(DQN_SE_list))
            Greedy_Average_SE_list.append(np.mean(Greedy_SE_list))
            #mmse_Average_SE_list.append(np.mean(mmse_SE_list))
            
            
            if episode % 100 == 0:
                C = np.mean(DQN_Average_rate_list[-100:])
                #score = np.mean(DQN_Average_rate_list[-100:])
                
                Joint_DQN_SDR.append(np.mean(Joint_DQN_Average_sum_data_rate_list[-100:]))
                Separate_DQN_SDR.append(np.mean(Separate_DQN_Average_sum_data_rate_list[-100:]))
                Clustering_SDR.append(np.mean(Clustering_Average_sum_data_rate_list[-100:]))
                Greedy_SDR.append(np.mean(Greedy_Average_sum_data_rate_list[-100:]))
                GA_SDR.append(np.mean(GA_Average_sum_data_rate_list[-100:]))
                #mmse_SDR.append(np.mean(mmse_Average_sum_data_rate_list[-100:]))
                
                
                Time.append(episode)
                Joint_DQN.append(np.mean(Joint_DQN_Average_rate_list[-100:]))
                Separate_DQN.append(np.mean(Separate_DQN_Average_rate_list[-100:]))
                DQN.append(np.mean(DQN_Average_rate_list[-100:]))
                Greedy.append(np.mean(Greedy_Average_rate_list[-100:]))
                GA.append(np.mean(GA_Average_rate_list[-100:]))
                #PA_GA.append(np.mean(PA_GA_Average_rate_list[-100:]))
                #mmse.append(np.mean(mmse_Average_rate_list[-100:]))
                
                Joint_DQN_PE.append(np.mean(Joint_DQN_Average_PE_list[-100:]))
                Separate_DQN_PE.append(np.mean(Separate_DQN_Average_PE_list[-100:]))
                GA_PE.append(np.mean(GA_Average_PE_list[-100:]))
                #PA_GA_PE.append(np.mean(PA_GA_Average_PE_list[-100:]))
                DQN_PE.append(np.mean(DQN_Average_PE_list[-100:]))
                Greedy_PE.append(np.mean(Greedy_Average_PE_list[-100:]))
                #mmse_PE.append(np.mean(mmse_Average_PE_list[-100:]))
                
                Joint_DQN_SE.append(np.mean(Joint_DQN_Average_SE_list[-100:]))
                Separate_DQN_SE.append(np.mean(Separate_DQN_Average_SE_list[-100:]))
                GA_SE.append(np.mean(GA_Average_SE_list[-100:]))
                #PA_GA_SE.append(np.mean(PA_GA_Average_SE_list[-100:]))
                DQN_SE.append(np.mean(DQN_Average_SE_list[-100:]))
                Greedy_SE.append(np.mean(Greedy_Average_SE_list[-100:]))
                #mmse_SE.append(np.mean(mmse_Average_SE_list[-100:]))
                
                print("Training:"+str(self.num_cell)+"BS "+str(self.max_UE)+"UE!!")
                print("Episode(train):%d  Joint_DQN SE: %.3f " %(episode, np.mean(Joint_DQN_Average_SE_list[-100:])))
                print("Episode(train):%d  Separate_DQN SE: %.3f " %(episode, np.mean(Separate_DQN_Average_SE_list[-100:]) ))
                print("Episode(train):%d  Greedy+Max-P SE: %.3f  " %(episode, np.mean(Greedy_Average_SE_list[-100:])))
                print("Episode(train):%d  GA+WMMSE SE: %.3f " %(episode, np.mean(GA_Average_SE_list[-100:])))
                
                print(" ")
                st = time.time()
                DQN_JT = 0
                Greedy_JT  = 0
                GA_JT  = 0
                Total_JT = 0
                Scheduling_DQN_total_time = 0
                PA_DQN_total_time  = 0
                mmse_total_time = 0
                GA_total_time =0
        
                if C > C_max:
                    PA_UA_agent.save("Model_512_256_256_128_64_UAPA_DQN_"+str(self.num_cell)+"BS_"+str(self.max_UE)+"UE.h5")
                    #PA_agent.save("PA_0616_"+ str(self.num_cell) + "cell_"+str(self.max_UE)+"UE_DQN_SE_PA.h5")
                    C_max = C
        #Clustering_agent.save("Clustering_0609_DQN_7cell_"+str(self.max_UE)+"UE_wmmse_PA.h5")
        #PA_agent.save("PA_0609_DQN_"+str(self.num_cell)+"cell_"+str(self.max_UE)+"UE_wmmse_PA_scale.h5")
        
        
        print("Average Joint_DQN SE: %.3f"%( np.mean(Joint_DQN_Average_SE_list)))
        print("Average Separate_DQN SE: %.3f"%( np.mean(Separate_DQN_Average_SE_list)))
        print("Average Greedy + Max-P SE: %.3f"%( np.mean(Greedy_Average_SE_list)))
        print("Average GA + WMMSE- SE: %.3f"%( np.mean(GA_Average_SE_list)))
        
        
        '--------'

        plt.xlabel("Training episodes")
        plt.ylabel("Spectral efficiency per TRP (bps/Hz) ")
        plt.plot(Time, Joint_DQN_SE, label ='DQN (Joint)',c='red', marker='o', mec='r',mfc='w')
        plt.plot(Time, Separate_DQN_SE, label ='DQN (Separate)',c='blue', marker='*', mec='blue',mfc='w')
        plt.plot(Time, GA_SE, label ='GA + WMMSE',c='teal', marker='X', mec='teal',mfc='w')
        plt.plot(Time, Greedy_SE, label ='Greedy + Max-P',c='g', marker='+', mec='g',mfc='w')
        plt.legend()
        #plt.savefig(str(self.num_cell)+'cell'+str(self.max_UE)+'UE_Spectral Efficiency_training.eps',dpi=600)
        plt.show()
        
        np.savetxt("Training_UAPA_JointDQN_SE_batch"+str(batch_size)+"_0811_training.csv", Joint_DQN_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Training_UAPA_Separate_DQN_SE_training.csv", Separate_DQN_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Training_UAPA_GA_SE_training.csv", GA_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Training_UAPA_Greedy_SE_training.csv", Greedy_SE,fmt="%.3f", delimiter=",")
        
        
        return np.mean(Joint_DQN_Average_SE_list), np.mean(Separate_DQN_Average_SE_list), np.mean(Greedy_Average_SE_list) , np.mean(GA_Average_SE_list)
    
        
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.memory = deque(maxlen=50000)
        self.gamma = 0.3
        self.learning_rate = 1e-3
        self.model = self.Build_Model()
    
    def Build_Model(self):

        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer='adam')
        
        return model
    
    def remember(self, state, action, reward, next_state):        
        for i in range(len(state)):
            experience = (state[i],action[i],reward[i],next_state[i])    
            self.memory.append(experience)
        
    def act(self, state, batch_size): 
        act_values = self.model.predict(state, batch_size= batch_size)
        return np.argmax(act_values,axis=1)  # returns action
        
    def replay(self, batch_size):
        
        #print("memory = \n",self.memory)
        minibatch = random.sample(self.memory, batch_size)

        batch_s  = [d[0] for d in minibatch]
        batch_a  = [d[1] for d in minibatch]
        batch_r  = [d[2] for d in minibatch]
        batch_ns = [d[3] for d in minibatch]
        
        batch_s = np.array(batch_s)
        batch_a = np.array(batch_a)
        batch_r = np.array(batch_r)
        batch_ns = np.array(batch_ns)
             
        batch_s = batch_s.reshape((batch_size,self.state_size))
        batch_a = batch_a.reshape((batch_size),)
        batch_r = batch_r.reshape((batch_size),)
        batch_ns = batch_ns.reshape((batch_size,self.state_size))
        
        target = (batch_r ) + self.gamma * np.amax(self.model.predict(batch_ns, batch_size=batch_size ),axis=1)
        target_f = self.model.predict(batch_s, batch_size= batch_size)
        target_f[np.arange(len(target_f)),batch_a] = target#self.learning_rate*target_f[np.arange(len(target_f)),batch_a] + (1-self.learning_rate)*target
        self.model.fit(batch_s, target_f, epochs=1, verbose=0)
   
    def load(self, name):
        self.model.load_weights(name)
            
    def save(self, name):
        self.model.save_weights(name)
      
        
env = Env_cellular(fd, Ts, x_border, y_border, max_UE, L, max_distance, min_distance, max_power, min_power, n_power, Ns)
joint_DQN_Average_SE, separate_DQN_Average_SE, greedy_Average_SE , ga_Average_SE = env.train()
#average_reward = env.test()

        
