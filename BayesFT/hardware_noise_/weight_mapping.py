from scipy import interpolate
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        return out

model = Net()

def weights_mapping(mean, std, model, device='cuda'):
    """
    Map target weight to true weight. 
    The difference between the target weight and the true weight is due to two factors:
    1.Random heat noises
    2.Non-monotonic characteristics of conductance-Q curves
    """

    """
    To achieve the above functionality, we first have to measure the conductance-Q curve. 
    As the measurement is affected by noises, we measure the curve for several times and get
    the mean value. 
    """
    c_mean_smooth, c_std_smooth = calculate()
    max_index = c_mean_smooth.shape[0]
    c_max = c_mean_smooth.max()
    c_min = c_mean_smooth.min()

    """
    We calculate the LCIS of the curve to find a monotonicly increasing part of the curve.
    """

    start_index, end_index= LCIS(c_mean_smooth) 

    mono_len = end_index - start_index

    #utility最大的曲线的值
    required_len = 55
    if mono_len < required_len: # Non-monotonic characteristics 
        increase_indices = np.arange(start_index, end_index, 1)
        end_index = start_index + required_len
        if end_index > max_index:
            start_index = start_index - end_index + max_index
            end_index = max_index
        ratio = np.zeros(required_len)
        for i in range(required_len):
            idx = i+start_index
            # 只有非递增区域的theta值受non-monotonic characteristics影响
            ratio[i] = 1 if idx in increase_indices else (c_mean_smooth[idx]-c_min)/(c_max-c_min)

    x = np.arange(start_index, end_index, 1)
    y = c_mean_smooth[start_index:end_index]

    interp_func_mean = interpolate.splrep(x, y, s=0)
    y_std = c_std_smooth[start_index:end_index]
    interp_func_std = interpolate.splrep(x, y_std, s=0)
    xfit = np.linspace(start_index, end_index, 100)
    yfit = interpolate.splev(xfit, interp_func_mean, der=0)
    yfit_std = interpolate.splev(xfit, interp_func_std, der=0) #计算取样点的插值结果


    gassian_kernel = torch.distributions.Normal(0.0, 1.0)
    with torch.no_grad():
        for theta in model.parameters():
            abstheta = torch.abs(theta) # 求参数的绝对值
            normalized_theta = abstheta / (torch.max(abstheta)+1e-8) # 归一化
            
            theta_index = normalized_theta*(required_len-1)
            theta_index = theta_index.type(torch.LongTensor) # 求各参数对应的下标位置

            noise_index = normalized_theta*100
            noise_index = noise_index.type(torch.LongTensor)
            noise_index[noise_index>=100]=99
            
            theta_ratio = torch.Tensor(ratio)[theta_index] # theta = theta * (c_real-c_min) / (c_max-c_min)

            noise_sigma = torch.Tensor(np.log(1+np.square(yfit_std/yfit)))[noise_index]  
            theta.mul_(to_device(theta_ratio * torch.exp(noise_sigma*gassian_kernel.sample(noise_sigma.size())), device))

def calculate(file):
    
    df = pd.read_excel(file)

    dropcolumn = []
    for i in range(len(df.columns)):
        if 'Unnamed' in df.columns[i]:
            dropcolumn.append(df.columns[i])
            

    df = df.drop(columns=dropcolumn)

    #calculate mean and std current for each row and remove the NAN
    voltage = df['voltage']
    current_mean_list = []
    current_std_list = []
    for i in range(len(voltage)):
        current_row = df.iloc[[i]].to_numpy()
        current_row = current_row[~np.isnan(current_row)]
        
        low_percentile = np.percentile(current_row, 25)
        high_percentile = np.percentile(current_row, 75)
        current_row = current_row[(current_row >=low_percentile ) & (current_row <= high_percentile)]
        
        current_mean = np.mean(current_row)
        
        
        current_std = np.std(current_row)
        current_mean_list.append(current_mean)
        current_std_list.append(current_std)

    current_mean_list = np.array(current_mean_list)
    current_std_list = np.array(current_std_list)

    voltage_list = df['voltage'].to_numpy()

    conductance_mean = current_mean_list/(voltage_list+1e-9)
    conductance_std  = current_std_list/(voltage_list+1e-9)

    c_mean_smooth = conductance_mean[1:100] 

    c_std_smooth = conductance_std[1:100]


    c_mean_smooth = Kalman1D(c_mean_smooth,damping=1)
    c_std_smooth  = Kalman1D(c_std_smooth,damping=1)


    return c_mean_smooth, c_std_smooth

def Kalman1D(observations,damping=0):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def LCIS(arr): # 求最长连续[递增]子序列，该[递增]序列中最多允许两次递减
    decrease_cnt = 0
    start_index = 0
    end_index = 0
    sub_len = 0
    longest_start = 0
    longest_end = 0
    longest_len = 0
    decrease_point_0 = -1
    decrease_point_1 = -1
    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            end_index += 1
        else:
            decrease_cnt += 1
            if decrease_cnt == 1:
                end_index += 1
                decrease_point_0 = end_index

            elif decrease_cnt == 2:
                end_index += 1
                decrease_point_1 = end_index
                
            else:
                sub_len = end_index - start_index + 1
                if longest_len < sub_len:
                    longest_len = sub_len
                    longest_start = start_index
                    longest_end = end_index
                start_index = decrease_point_0
                decrease_point_0 = decrease_point_1
                decrease_point_1 = i
                end_index = i
                decrease_cnt = 2

    sub_len = end_index - start_index + 1
    if longest_len < sub_len:
        longest_len = sub_len
        longest_start = start_index
        longest_end = end_index

    return longest_start, longest_end

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def main():
    weights_mapping(0, 1, model=model, device='cpu')

if __name__ == '__main__':
    main()
