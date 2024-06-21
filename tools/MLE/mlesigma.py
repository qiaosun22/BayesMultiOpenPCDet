import pandas as pd
import numpy as np
# data = pd.read_excel('I-V_data_30min_AAO_5min second etch_15min_Pb_ED_3h_180C_MAI_200nm_Ag_memory_6V.xlsx')

def mlesigma(f_name):
    # df = pd.read_excel('I-V_data_25min_AAO_10min_Pb_ED_1h_180C_MAI_memory_8V_2.xlsx') #0.55
    # df = pd.read_excel('I-V_data_30min_AAO_5min second etch_15min_Pb_ED_3h_180C_MAI_200nm_Ag_memory_6V.xlsx') #0.33
    # df = pd.read_excel('I-V_data_25min_AAO_10min_Pb_ED_1h_180C_MAI_memory_8V.xlsx') #0.14

    # data = pd.read_excel('../hardware_noise/I-V_data_25min_AAO_10min_Pb_ED_1h_180C_MAI_memory_8V.xlsx') #0.14
    # data = pd.read_excel('../hardware_noise/I-V_data_30min_AAO_5min second etch_15min_Pb_ED_3h_180C_MAI_200nm_Ag_memory_6V.xlsx') #0.33
    data = pd.read_excel(f_name, engine='openpyxl') #0.55

    conductance = data.drop(['voltage'], axis=1).div(data['voltage'], axis=0)

    conductance = conductance[(conductance!=np.inf).all(axis=1)]

    # (conductance.div(conductance.mean(axis=1), axis=0)<0).all(axis=0)


    from scipy.stats import norm
    import matplotlib.pyplot as plt

    conductance_by_mean = np.array(conductance.div(conductance.mean(axis=1), axis=0)).reshape(1, -1)[0]
    samples = np.log(conductance_by_mean)
    samples_fine = np.delete(conductance_by_mean, np.isfinite(samples)==False)
    # plt.hist(samples_fine, bins=100)  # 直方图显示
    # plt.show()
    print(norm.fit(samples_fine))  # 返回极大似然估计
    print(norm.fit(samples_fine, floc=0))  # 返回极大似然估计
    
    


mlesigma('I-V_data_25min_AAO_10min_Pb_ED_1h_180C_MAI_memory_8V_2.xlsx') #0.55
