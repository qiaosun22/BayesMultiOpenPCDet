from scipy import interpolate
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm
# import matplotlib.pyplot as plt

def mlesigma(f_name, st_indx, end_indx):
    """
    我们引入 MLE 方法来估计忆阻器电导值与对应电压水平下电导均值的比值的 mu 和 sigma
    函数返回 sigma 值，它的值反映了等效噪声水平的强度
    '''
    关于噪声分布形式设定与转换的说明：
    X ~ logNorm(a, b)  # m, v 是对数正态分布本身的均值和方差
    # 这说的是 X 是一个值永远为正的随机变量，只有取对数之后才是正态分布
    即 logX ~ Norm(mu, sigma)

    换算关系：
    m = exp(mu + sigma^2/2)  # 注意到 m 是 >0 的
    v = exp(2*mu + sigma^2) * exp(sigma^2 - 1)  

    因此，一个随机变量 Y 如果服从正态分布，即 Y ~ Norm(mu, sigma)
    就一定可以有一个 X = exp(Y), 使得 Y = logX ~ Norm(mu, sigma)

    所以，一个正态分布随机变量取指数得到的变量就服从对数正态分布

    具体到上面的代码，已知gaussian_kernel 服从正态分布，则 
            torch.exp(noise_sigma*gassian_kernel.sample(noise_sigma.size())) 
    服从对数正态分布，其中我们施加的 mu, sigma 都是对应的正态分布的参数，而非对数正态分布的参数

    又由于 (m, v) 和 (mu, sigma) 是一一映射，我们控制谁都一样，
    所以，为了方便和统一，今后：
    1）需要采样对数正态分布样本，都采取先用 mu, sigma 采样正态分布，再 exp 的方式
    2）需要估计对数正态分布参数，都采取先对对数正态分布样本取 log，再用 norm.fit 拟合的方式
    '''
    """
    data = pd.read_excel(f_name, engine='openpyxl').iloc[1: 101, :]
    data = data.iloc[st_indx: end_indx+1, :]
    
    conductance = data.drop(['voltage'], axis=1).div(data['voltage'], axis=0)
    conductance = conductance[(conductance!=np.inf).all(axis=1)]
    
    conductance_by_mean = np.array(conductance.div(conductance.mean(axis=1), axis=0)).reshape(1, -1)[0]
    samples = np.log(conductance_by_mean)
    samples_fine = np.delete(conductance_by_mean, np.isfinite(samples)==False)

    mu_hat, sigma_hat = norm.fit(samples_fine)  # 通过极大似然估计得到 sigma\hat，不固定均值为 0
    # mu_hat, sigma_hat = norm.fit(samples_fine, floc=0)  # 通过极大似然估计得到 sigma\hat，固定均值为 0
    
    return sigma_hat # 我们只关注 sigma_hat


def LCIS(arr):
    """
    求最长连续[递增]*子序列
    返回最长连续递增子序列的
    ---------
    *该[递增]序列并不要求始终严格增，而是最多允许两次递减
    """
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


def calculate_smoothed_cmean(f_name):
    df = pd.read_excel(f_name, engine='openpyxl')
    eps = 1e-12
    current_cols = df.iloc[:, 1:]
    voltage_col = df.iloc[:, :1].values

    conductance_cols = current_cols.div(voltage_col + eps)
    conductance_mean_col = conductance_cols.mean(axis=1).values
    conductance_mean_col_head100 = conductance_mean_col[1: 101]
    c_mean_smooth = Kalman1D(conductance_mean_col_head100)
    # c_mean_smooth = c_mean_smooth.T[0]
    return c_mean_smooth


def Kalman1D(observations, damping=0):
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


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def weight_mapping(f_name, model, noise_sigma, device='cuda'):
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
    c_mean_smooth = calculate_smoothed_cmean(f_name)
    
    max_index = c_mean_smooth.shape[0]
    c_max = c_mean_smooth.max()
    c_min = c_mean_smooth.min()

    """
    We calculate the LCIS of the curve to find a monotonicly increasing part of the curve.
    """
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    start_index, end_index= LCIS(c_mean_smooth) 
    mono_len = end_index - start_index
    
    #utility最大的曲线的值
    required_len = 35
    
    ratio = np.ones(required_len)
    if mono_len < required_len: # Non-monotonic characteristics 
        increase_indices = np.arange(start_index, end_index, 1)
        end_index = start_index + required_len
        if end_index > max_index:
            start_index = start_index - end_index + max_index
            end_index = max_index
        
        for i in range(required_len):
            idx = i + start_index
            # 只有非递增区域的theta值受non-monotonic characteristics影响
            ratio[i] = 1 if idx in increase_indices else (c_mean_smooth[idx]-c_min)/(c_max-c_min)
        
    # noise_sigma = mlesigma(f_name, start_index, end_index)
    gassian_kernel = torch.distributions.Normal(0.0, noise_sigma)
    with torch.no_grad():
        for theta in model.parameters():
            abstheta = torch.abs(theta) # 求参数的绝对值
            normalized_theta = abstheta / (torch.max(abstheta) + 1e-8) # 归一化

            theta_index = normalized_theta * (required_len-1)
            theta_index = theta_index.type(torch.LongTensor) # 求各参数对应的下标位置
            noise_index = normalized_theta * 100
            noise_index = noise_index.type(torch.LongTensor)
            noise_index[noise_index >= 100] = 99
            
            theta_ratio = torch.Tensor(ratio)[theta_index].cuda() # theta = theta * (c_real-c_min) / (c_max-c_min)

            mul_ = theta_ratio * torch.exp(gassian_kernel.sample(theta.size()).cuda())
            theta.mul_(mul_)

            
def weight_mapping_sigma_only(f_name, model, device='cuda'):
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
    c_mean_smooth = calculate_smoothed_cmean(f_name)
    
    max_index = c_mean_smooth.shape[0]
    c_max = c_mean_smooth.max()
    c_min = c_mean_smooth.min()

    """
    We calculate the LCIS of the curve to find a monotonicly increasing part of the curve.
    """

    start_index, end_index= LCIS(c_mean_smooth) 
    mono_len = end_index - start_index
    
    #utility最大的曲线的值
    required_len = 45
    
    ratio = np.ones(required_len)
    if mono_len < required_len: # Non-monotonic characteristics 
        increase_indices = np.arange(start_index, end_index, 1)
        end_index = start_index + required_len
        if end_index > max_index:
            start_index = start_index - end_index + max_index
            end_index = max_index
        
        for i in range(required_len):
            idx = i + start_index
            # 只有非递增区域的theta值受non-monotonic characteristics影响
            ratio[i] = 1 if idx in increase_indices else (c_mean_smooth[idx]-c_min)/(c_max-c_min)
        
    noise_sigma = mlesigma(f_name, start_index, end_index)
    gassian_kernel = torch.distributions.Normal(0.0, noise_sigma)
    with torch.no_grad():
        for theta in model.parameters():
#             abstheta = torch.abs(theta) # 求参数的绝对值
#             normalized_theta = abstheta / (torch.max(abstheta) + 1e-8) # 归一化

#             theta_index = normalized_theta * (required_len-1)
#             theta_index = theta_index.type(torch.LongTensor) # 求各参数对应的下标位置
#             noise_index = normalized_theta * 100
#             noise_index = noise_index.type(torch.LongTensor)
#             noise_index[noise_index >= 100] = 99
            
#             theta_ratio = torch.Tensor(ratio)[theta_index].cuda() # theta = theta * (c_real-c_min) / (c_max-c_min)

            mul_ = torch.exp(gassian_kernel.sample(theta.size()).cuda())
            theta.mul_(mul_)


if __name__ == '__main__':
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 3)
        def forward(self, xb):
            xb = xb.view(xb.size(0), -1)
            out = self.linear1(xb)
            return out

    model = Net()
    model.cuda()
    print(model.state_dict())

    f_name = 'I-V_data_30min_AAO_5min second etch_15min_Pb_ED_3h_180C_MAI_200nm_Ag_memory_6V.xlsx' #'I-V_data_25min_AAO_10min_Pb_ED_1h_180C_MAI_memory_8V_2.xlsx'
    weight_mapping(f_name, model, device='cuda')
    print(model.state_dict())
