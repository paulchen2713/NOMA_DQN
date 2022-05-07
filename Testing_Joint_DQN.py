# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:19:07 2020

Testing UAPA Joint_DQN and Separate_DQN
"""

from collections import deque
import scipy
import scipy.special
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # recommended import according to the docs
import heapq
from random import choice
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

fd = 10    #maximal Doppler frequency
Ts = 20e-3 #coherent time
x_border = 2
y_border = 2
max_UE = 10  # user number in the coverage of one BS 
min_distance = 0.01#0.01 #km
max_distance = 0.25#km
max_power = 38  #dBm
n_power = -114. #dBm
power_num = 10  #power level
Ns = 11
dtype = np.float32

class Env_cellular():
    def __init__(self, fd, Ts, x_border, y_border, max_UE, max_distance, min_distance, max_power, n_power, Ns):
        self.fd = fd  #doppler(Hz)
        self.Ts = Ts  #time interval between adjacent instants
        self.x_border = x_border #km
        self.y_border = y_border #km
        
        self.num_cell = self.x_border * self.x_border
        self.cell_BS  = 3
        self.num_BS = self.num_cell * 3#self.x_border * self.y_border #number of BS (radius 1km ) N
        self.max_UE = max_UE #number of UE in one cell
        self.total_UE = self.num_cell * self.max_UE #self.max_UE * self.num_BS #Total number of UE in network M
        self.max_distance = max_distance #km
        self.min_distance = min_distance #km
        self.max_power = max_power #max transmit power(dBm)
        self.max_power_W =  1e-3 *pow(10., self.max_power/10.) #dBm >> mW
        self.n_power   = n_power                           #noise power
        self.n_power_W =  1e-3* pow(10., self.n_power/10.)   #noise power (dBm to mW)
        self.W = np.ones((self.total_UE), dtype=dtype)  #bandwidth
        self.Ns = Ns #?!
        
        self.BS_antenna = 2 #{2,4,8,16}
        

    def train(self):
        max_episode = 3000
        PA_state_size    = 65
        Clustering_state_size = 266
        PA_action_num = 10
        Clustering_action_num = 5
        Candidate_action_num = 5
        
        PA_agent         = DQNAgent(PA_state_size,PA_action_num)
        Clustering_agent = DQNAgent(Clustering_state_size,Clustering_action_num)
        PA_UA_agent = DQNAgent(Clustering_state_size,Candidate_action_num * PA_action_num)
        
        PA_UA_agent.load("Model_512_256_256_128_64_UAPA_DQN_4BS_10UE.h5")
        Clustering_agent.load("Model_512_256_256_128_64_OnlyUA_DQN_4BS_10UE.h5")
        PA_agent.load("Model_512_256_256_128_64_OnlyPA_DQN_4BS_10UE.h5")
        
        Separate_DQN_SE = []
        Joint_DQN_SE    = []
        Greedy_SE  = []
        GA_SE  = []
        
        Separate_DQN_rate =[]
        Joint_DQN_rate    =[]
        Greedy_rate =[]
        GA_rate  =[]
        
        Separate_DQN_Average_SE_list = []
        Joint_DQN_Average_SE_list = []
        GA_Average_SE_list  = []
        Greedy_Average_SE_list  = []
        
        Time = []
        	
        Joint_DQN_total_time =0
        Separate_DQN_total_time  = 0
        Greedy_MaxP_total_time = 0 
        GA_WMMSE_total_time = 0
        
        power_set  = 1e-3 * pow(10., np.linspace(0,self.max_power, PA_action_num)/10.)        
        mean_power = np.mean(power_set)
        std_power  = np.std(power_set)
        
        PA_highest_num =10
        scheduling_highest_num = 5
        
        for episode in range(1, max_episode+1):
            '--------------------------------------------- Build environment matrix ---------------------------------------------'
            
            Separate_DQN_SE_list = []
            Joint_DQN_SE_list    = []
            GA_SE_list  = []
            Greedy_SE_list  = []            
            count = 0
            '''
            Jakes model
            '''
            H_set = np.zeros([self.Ns,self.total_UE,self.num_BS,self.BS_antenna], dtype= complex)
            rho = np.float32(scipy.special.j0(2*np.pi*self.fd*self.Ts))   #rho=j0(2pi*fd*Ts), j0 = first kind zero-order Bessel function
            H_set[0,:,:,:] = np.sqrt(0.5*(np.random.randn(self.total_UE, self.num_BS, self.BS_antenna) + 1j * np.random.randn(self.total_UE, self.num_BS, self.BS_antenna)))
              
            for i in range(1,self.Ns):
                H_set[i,:,:,:]  = H_set[i-1,:,:,:]*rho + np.sqrt((1.-rho**2)*0.5*(np.random.randn(self.total_UE, self.num_BS, self.BS_antenna) + 1j * np.random.randn(self.total_UE, self.num_BS, self.BS_antenna)))
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
            freq = 2000   #(M)    #carrier freq
            height_bs = 32 #(m)   #BS_ant_height
            height_ue = 1 #(m)    #UE_ant_height
            
            Att_pattern_expand = Att_pattern_expand.transpose()
            
            path_loss = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(height_bs) - ((1.1* np.log10(freq)-0.7)*height_ue-(1.56*np.log10(freq)-0.8)) + (44.9-6.55*np.log10(height_bs))*np.log10(distance) - Att_gain + Att_pattern_expand + (8 * np.random.rand(self.total_UE, self.num_BS))
            path_loss = pow(10.,-path_loss/10.)
            
            channel = np.zeros([self.Ns,self.total_UE,self.num_BS,self.BS_antenna], dtype=complex)
            for i in range(self.BS_antenna):
                channel[:,:,:,i] = H_set[:,:,:,i] *path_loss# ch
                       
            MRT_precoding = np.zeros([self.Ns,self.total_UE,self.num_BS,self.BS_antenna], dtype=complex)
            for i in range(self.BS_antenna):
                MRT_precoding[:,:,:,i] = H_set[:,:,:,i] / np.linalg.norm(H_set, axis=-1)
            
            channel_gain = (np.linalg.norm(H_set * np.conj(MRT_precoding), axis=-1)**2) * path_loss            
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
            
            previous_inter_power  = np.array(previous_inter_power)
            previous_inter_power  =   previous_inter_power / self.max_power_W
            
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
            
            '----- Move to next time slot ----'
            
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
            
            '----- Move to next time slot ----'
            
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
            
            previous_repeat_matrix = []
            for i in range(self.num_BS):
                cell_number = int( i / self.cell_BS)
                cell_BS_idx = int( i % self.cell_BS )
                repeat = Comp_indicate[clustering_random_action[i]]
                repeat = repeat[cell_number * self.cell_BS : (cell_number+1) * self.cell_BS]
                repeat = np.roll(repeat,  -1 * cell_BS_idx)
                previous_repeat_matrix.append(repeat)
            previous_repeat_matrix = np.array(previous_repeat_matrix)
            
            '---- (End) Gernerate PA first step ---'
            '----------------------------------------------------- Start Time Slot ------------------------------------------------------------'
            for time_s in range(int(self.Ns)-2):
                
                '------------------------------------------- (Start) GA + WMMSE  -----------------------------------------'
                GA_WMMSE_t1 = time.time()
                
                total_combins = int(math.pow(self.max_UE,self.cell_BS))
                equal_power = self.max_power_W * np.ones((self.num_BS,1))
                Max_Action = []
                for j in range(self.num_cell):
                    max_reward = 0
                    global_max_action = np.zeros((self.num_BS,),dtype = np.int32) 
                    start_idx =   j    * self.max_UE
                    end_idx   = (j+1)  * self.max_UE
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
                Max_Action  = np.array(Max_Action)
                Best_action = Max_Action.flatten()
                
                max_wmmse_t = 100
                GA_WMMSE_action  = np.array(Best_action.flatten())
                hkk = np.sqrt( h_gain[GA_WMMSE_action,range(self.num_BS)] )
                v =  np.random.uniform( 0, np.sqrt(self.max_power_W), size = (self.num_BS,1)).flatten()
                u = ((hkk * v) / ((np.dot(h_gain[GA_WMMSE_action, :],v**2)) + self.n_power_W))
                w =  1. / (1 - u * hkk * v)
                C = np.sum(w)
                
                for wmmse_t in range(max_wmmse_t):
                    C_last = C
                    v = (hkk*u*w) / ((np.dot(h_gain[GA_WMMSE_action, :] * (u**2)  ,v))**2)
                    v = np.minimum(np.sqrt(self.max_power_W), np.maximum(0, v))
                    u = (hkk * v) / ((np.dot(h_gain[GA_WMMSE_action, :],v**2)) + self.n_power_W)
                    w = 1. / (1. - u * hkk * v)
                    C = np.sum(w)     
                    if np.abs(C_last - C) < 1e-3:
                        break
                p_mmse = v**2
                
                GA_WMMSE_t2 = time.time()
                
                p_mmse = np.array(p_mmse).reshape((self.num_BS,1))
                GA_mmse_action = np.array(GA_WMMSE_action)
                GA_mmse_idx = np.zeros((self.total_UE,self.num_BS))
                GA_mmse_idx[GA_mmse_action,range(self.num_BS)] = 1
                GA_mmse_number = np.sum(GA_mmse_idx, axis=1,dtype=np.int8)
                GA_mmse_number = GA_mmse_number.reshape(self.total_UE,1)
                
                GA_mmse_idx = GA_mmse_idx.repeat(self.BS_antenna).reshape(self.total_UE,self.num_BS,self.BS_antenna)
                GA_mmse_main_path = H2 * GA_mmse_idx
                GA_mmse_main_path = GA_mmse_main_path * np.conj(MR_precoder)
                GA_mmse_main_path = np.linalg.norm(GA_mmse_main_path,axis=-1)**2 * path_loss
                GA_mmse_main_path = np.dot(GA_mmse_main_path, p_mmse)
                
                GA_mmse_inter_idx = (GA_mmse_idx - 1) * (-1)
                GA_mmse_inter_path = H2 * GA_mmse_inter_idx
                GA_mmse_inter_path = GA_mmse_inter_path * np.conj(MR_precoder)
                GA_mmse_inter_path = np.linalg.norm(GA_mmse_inter_path,axis=-1)**2 * path_loss
                GA_mmse_inter_path = np.dot(GA_mmse_inter_path, p_mmse)
                
                GA_mmse_sinr = GA_mmse_main_path / (GA_mmse_inter_path + self.n_power_W)
                GA_mmse_data_rate = np.log2(1. + GA_mmse_sinr)
                GA_mmse_nor_rate  = np.log2(1. + GA_mmse_sinr) / GA_mmse_number
                GA_mmse_nor_rate[ GA_mmse_number==0 ] = 0
                
                '------------------------------------------- (End) GA + WMMSE  -----------------------------------------'
                
                
                '------------------------------------------- (Start) Greedy + MaxP  -----------------------------------------'
                
                Greedy_MaxP_t1 = time.time()
                equal_power = self.max_power_W * np.ones((self.num_BS,1))  # equal max power
                Greedy_action = np.zeros((self.num_BS,1),dtype = np.int32)
                for i in range(self.num_BS):
                    cell_number = int(i / self.cell_BS)
                    Greedy_action[i] = np.argmax(h_gain[cell_number*self.max_UE : (cell_number+1)*self.max_UE], axis=0)[i]
                    Greedy_action[i] = Greedy_action[i] + cell_number * self.max_UE
                Greedy_action = Greedy_action.flatten()
                Greedy_MaxP_t2 = time.time()
                
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
                
                '------------------------------------------- (End) Greedy + MaxP  -----------------------------------------'
                
                
                '------------------------------------------- (Start) Joint DQN  -----------------------------------------'
                
                DQN_max_action = []
                Joint_DQN_t1 = time.time() 
                DQN_max_action = PA_UA_agent.act(DQN_input_state, self.num_BS)
                Joint_DQN_t2 = time.time() 
                
                UA_action = np.floor( DQN_max_action/PA_action_num).astype(np.int32)
                PA_action = np.array( DQN_max_action%PA_action_num )
                                
                UE_candidate = np.array(UE_candidate)
                UA_action = UE_candidate[range(self.num_BS),UA_action]  #change action from max-min UE to corresponding UE_idx
                
                epsilon = 0#INITIAL_EPSILON - episode * (INITIAL_EPSILON-FINAL_EPSILON)/ max_episode  # decade epsilon
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
                
                
                DQN_serving_idx    = np.zeros((self.num_BS,self.total_UE))
                DQN_serving_idx[range(self.num_BS), UA_action] = 1
                DQN_serving_idx    = DQN_serving_idx.transpose()
                
                DQN_serving_number = np.sum(DQN_serving_idx,axis=1)
                DQN_serving_number = DQN_serving_number.reshape(self.total_UE,1)
                                
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
                                                
                '------------------------------------------- (End) Joint DQN  -----------------------------------------'
                
                '------------------------------------------- (Start) Separate DQN  -----------------------------------------'
                '-------- (Start) DQN-UA ---------'
                max_action = []
                UA_DQN_t1 = time.time()
                max_action = Clustering_agent.act(clustering_input_state, self.num_BS)
                UA_DQN_t2 = time.time()
                                
                UE_candidate = np.array(UE_candidate)
                max_action = UE_candidate[range(self.num_BS),max_action]  #change action from max-min UE to corresponding UE_idx
                
                epsilon = 0 #INITIAL_EPSILON - episode * (INITIAL_EPSILON-FINAL_EPSILON)/ max_episode  # decade epsilon
                random_index = np.array(np.random.uniform(size=self.num_BS) < epsilon, dtype = np.int32)
                clustering_random_action = list()
                for i in range(self.num_BS):
                    random_sample = choice(UE_candidate[i])
                    clustering_random_action.append(random_sample)
                
                action_set = np.vstack([max_action, clustering_random_action]) #沿著直方向將矩陣堆疊起來。
                Comp_action = action_set[random_index,range(self.num_BS)]
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
                
                Comp_indicate      = np.array(Comp_index)
                
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
                
                '-------- (End) DQN-UA ---------'
                '----- Gernerate PA next state ------'
                
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
                
                PA_next_state = np.hstack((previous_PA_main_channel, previous_PA_interference, previous_PA_inter_power, previous_interfered_channel, previous_repeat_matrix, current_PA_main_channel, current_PA_interference, current_interfered_channel, current_repeat_matrix, power_matrix, rate_matrix))                
                
# =============================================================================
#                 if time_s > 0:
#                     PA_agent.remember(PA_input_state, PA_action, PA_reward, PA_next_state)
# =============================================================================
                
                PA_input_state = PA_next_state
                
                '----- Gernerate PA next state ------'
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
                PA_power = power_set[power_index]
                                
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
                                                

                '---- (Start) Gernerate DQN next step ---'
                
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
                
                clustering_next_state = np.hstack(( previous_BS_input, previous_repeat_candidate, previous_power, current_BS_input, current_repeat_candidate, current_power))
                DQN_next_state = np.hstack(( previous_UA_PA_BS_input, previous_repeat_candidate, previous_UA_PA_power, current_UA_PA_BS_input, current_repeat_candidate, current_UA_PA_power))
                
                #Clustering_agent.remember(clustering_input_state, clustering_action, clustering_reward, clustering_next_state)  #testing no rememerber
                #PA_UA_agent.remember(clustering_input_state, DQN_max_action, DQN_reward, clustering_next_state)                 #testing no rememerber
                clustering_input_state = clustering_next_state
                DQN_input_state        = DQN_next_state
                
                '---- (End) Gernerate DQN next step ---'

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
                '---- (End) Gernerate PA next step ---'
                
                
                '-------------------------- PLOT ------------------------'
                '---- Average SE to all Link----'
                Joint_DQN_rate.append(np.array(DQN_data_rate).flatten())
                Separate_DQN_rate.append(np.array(PA_data_rate).flatten())
                Greedy_rate.append(np.array(Greedy_data_rate).flatten())
                GA_rate.append(np.array(GA_mmse_data_rate).flatten())
                '---- Average SE to all Link----'
                
                '---- SE to All Algorithm ---'
                Joint_DQN_spectrum_eff     =  np.sum(DQN_data_rate) / self.num_BS
                Separatet_DQN_spectrum_eff =  np.sum(PA_data_rate) / self.num_BS
                GA_spectrum_eff      =  np.sum(GA_mmse_data_rate) / self.num_BS
                Greedy_spectrum_eff  =  np.sum(Greedy_data_rate) / self.num_BS
                    
                Joint_DQN_SE_list.append(Joint_DQN_spectrum_eff)
                Separate_DQN_SE_list.append(Separatet_DQN_spectrum_eff)
                GA_SE_list.append(GA_spectrum_eff)
                Greedy_SE_list.append(Greedy_spectrum_eff)
                '---- SE to All Algorithm ---'
                '-------------------------- PLOT ------------------------'
                
                Joint_DQN_total_time = Joint_DQN_total_time + (Joint_DQN_t2 - Joint_DQN_t1)
                Separate_DQN_total_time  = Separate_DQN_total_time + (UA_DQN_t2  - UA_DQN_t1)+ (PA_DQN_t2  - PA_DQN_t1)
                Greedy_MaxP_total_time = Greedy_MaxP_total_time + (Greedy_MaxP_t2- Greedy_MaxP_t1)
                GA_WMMSE_total_time  = GA_WMMSE_total_time + (GA_WMMSE_t2  - GA_WMMSE_t1) 
                        
            Joint_DQN_Average_SE_list.append(np.mean(Joint_DQN_SE_list))
            Separate_DQN_Average_SE_list.append(np.mean(Separate_DQN_SE_list))
            GA_Average_SE_list.append(np.mean(GA_SE_list))
            Greedy_Average_SE_list.append(np.mean(Greedy_SE_list))
            
            if episode % 100 == 0:
                Time.append(episode)
                Joint_DQN_SE.append(np.mean(Joint_DQN_Average_SE_list[-100:]))
                Separate_DQN_SE.append(np.mean(Separate_DQN_Average_SE_list[-100:]))
                GA_SE.append(np.mean(GA_Average_SE_list[-100:]))
                Greedy_SE.append(np.mean(Greedy_Average_SE_list[-100:]))
                
                print("Testing:"+str(self.num_cell)+"BS "+str(self.max_UE)+"UE!")
                print("Episode(test):%d  Joint_DQN: %.3f  Time cost: %.2fs" %(episode, np.mean(Joint_DQN_Average_SE_list[-100:]), Joint_DQN_total_time))
                print("Episode(test):%d  Separate_DQN: %.3f  Time cost: %.2fs" %(episode, np.mean(Separate_DQN_Average_SE_list[-100:]), Separate_DQN_total_time))
                print("Episode(test):%d  Greedy + Max-P: %.3f  Time cost: %.2fs" %(episode, np.mean(Greedy_Average_SE_list[-100:]),Greedy_MaxP_total_time))
                print("Episode(test):%d  GA + WMMSE : %.3f  Time cost: %.2fs" %(episode, np.mean(GA_Average_SE_list[-100:]), GA_WMMSE_total_time))
                print("  ")
                Joint_DQN_total_time = 0
                Separate_DQN_total_time = 0
                Greedy_MaxP_total_time  = 0
                GA_WMMSE_total_time =0
                
        print("Average Joint_DQN SE: %.3f"%( np.mean(Joint_DQN_Average_SE_list)))
        print("Average Separate_DQN SE: %.3f"%( np.mean(Separate_DQN_Average_SE_list)))
        print("Average Greedy + Max-P SE: %.3f"%( np.mean(Greedy_Average_SE_list)))
        print("Average GA + WMMSE- SE: %.3f"%( np.mean(GA_Average_SE_list)))
        
        '--------'
        
        Joint_DQN_rate = np.array(Joint_DQN_rate)
        Joint_DQN_rate = np.mean(Joint_DQN_rate , axis=0)
        Joint_DQN_rate = Joint_DQN_rate.flatten()
        
        Separate_DQN_rate = np.array(Separate_DQN_rate)
        Separate_DQN_rate = np.mean(Separate_DQN_rate , axis=0)
        Separate_DQN_rate = Separate_DQN_rate.flatten()
        
        Greedy_rate = np.array(Greedy_rate)
        Greedy_rate = np.mean(Greedy_rate , axis=0)
        Greedy_rate = Greedy_rate.flatten()
        
        GA_rate  = np.array(GA_rate)
        GA_rate  = np.mean(GA_rate , axis=0)
        GA_rate = GA_rate.flatten()       
        
        Joint_DQN_rate_ecdf = sm.distributions.ECDF(Joint_DQN_rate)
        Separate_DQN_rate_ecdf = sm.distributions.ECDF(Separate_DQN_rate)
        Greedy_rate_ecdf = sm.distributions.ECDF(Greedy_rate)
        GA_rate_ecdf = sm.distributions.ECDF(GA_rate)
        
        Joint_DQN_x = np.linspace(min(Joint_DQN_rate), max(Joint_DQN_rate))
        Separate_DQN_x = np.linspace(min(Separate_DQN_rate), max(Separate_DQN_rate))
        Greedy_x = np.linspace(min(Greedy_rate), max(Greedy_rate))
        GA_x = np.linspace(min(GA_rate), max(GA_rate))
        
        Joint_DQN_y = Joint_DQN_rate_ecdf(Joint_DQN_x)
        Separate_DQN_y = Separate_DQN_rate_ecdf(Separate_DQN_x)
        Greedy_y = Greedy_rate_ecdf(Greedy_x)
        GA_y = GA_rate_ecdf(GA_x)
        plt.xlabel("Average spectral efficiency (bps/Hz) per UE ")
        plt.ylabel("CDF")
        plt.step(Joint_DQN_x, Joint_DQN_y, label ='DQN (Joint)',c='red')
        plt.step(Separate_DQN_x, Separate_DQN_y, label ='DQN (Separate)',c='blue',linestyle = '-.')
        plt.step(GA_x, GA_y, label ='GA + WMMSE',c='teal',linestyle = '--')
        plt.step(Greedy_x, Greedy_y, label ='Greedy + Max-P',c='g', linestyle = ':')
        plt.legend()
        plt.savefig(str(self.num_cell)+'BS_'+str(self.max_UE)+'UE_SE_perUE_Testing.eps',dpi=600)
        plt.show()
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_JointDQN_SE_x.csv", Joint_DQN_x,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_SeparateDQN_SE_x.csv", Separate_DQN_x,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_GA_SE_x.csv", GA_x,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_Greedy_SE_x.csv", Greedy_x,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_JointDQN_SE_y.csv", Joint_DQN_y,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_SeparateDQN_SE_y.csv", Separate_DQN_y,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_GA_SE_y.csv", GA_y,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_CDF_"+str(self.max_UE)+"UE_Greedy_SE_y.csv", Greedy_y,fmt="%.3f", delimiter=",")
        
        plt.xlabel("Testing episodes")
        plt.ylabel("Spectral efficiency per TRP (bps/Hz)")
        plt.plot(Time, Joint_DQN_SE, label ='DQN (Joint)',c='red', marker='o', mec='r',mfc='w')
        plt.plot(Time, Separate_DQN_SE, label ='DQN (Separate)',c='blue', marker='*', mec='blue',mfc='w')
        plt.plot(Time, GA_SE, label ='GA + WMMSE',c='teal', marker='X', mec='teal',mfc='w')
        plt.plot(Time, Greedy_SE, label ='Greedy + Max-P',c='g', marker='+', mec='g',mfc='w')
        plt.legend()
        plt.savefig(str(self.num_cell)+'BS_'+str(self.max_UE)+'UE_SE_Testing.eps',dpi=600)
        plt.show()
        np.savetxt("Testing_SE_Time.csv", Time,fmt="%.0f", delimiter=",")
        np.savetxt("Testing_JointDQN_"+str(self.max_UE)+"UE_SE.csv", Joint_DQN_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_SeparateDQN_"+str(self.max_UE)+"UE_SE.csv", Separate_DQN_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_GA_"+str(self.max_UE)+"UE_SE.csv", GA_SE,fmt="%.3f", delimiter=",")
        np.savetxt("Testing_Greedy_"+str(self.max_UE)+"UE_SE.csv", Greedy_SE,fmt="%.3f", delimiter=",")
        
        return np.mean(Joint_DQN_Average_SE_list), np.mean(Separate_DQN_Average_SE_list), np.mean(Greedy_Average_SE_list) , np.mean(GA_Average_SE_list)
    
        
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.memory = []
        self.memory = deque(maxlen=50000)
        self.gamma = 0.3   # discount rate  
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
      
        
env = Env_cellular(fd, Ts, x_border, y_border, max_UE, max_distance, min_distance, max_power, n_power, Ns)
joint_DQN_Average_SE, separate_DQN_Average_SE, greedy_Average_SE , ga_Average_SE = env.train()
#average_reward = env.test()

        
