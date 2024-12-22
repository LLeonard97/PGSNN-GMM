# -*- coding:utf8 -*-
# @Time : 2023/10/15 22:47
# @Author : Xianwei Liu
import numpy as np
import pandas as pd
from sympy import sympify, symbols, evaluate
import sympy as sy
import math
import torch
import matplotlib.pyplot as plt


def calculate_fsite(vs30):
    Fsite = 1
    if vs30 <= 925:
        Fsite = vs30 / 760

    elif vs30 > 925:
        Fsite = 925 / 760
    return Fsite


def SNN_pre(txt_path, Mw, Rjb, Vs30, max_values, need_norm, duration_factor):
    Fsite = calculate_fsite(Vs30)
    with open(txt_path, 'r') as file:
        content = file.readlines()
    content.remove('\n')
    result_all = []
    for i in range(len(content)):
        s = sympify(content[i])
        M, R, lnR, F = symbols('mw rjb lnrjb fsite')
        if need_norm==True:
            expr = s.subs([(M, Mw / float(max_values['Mw'])), (R, Rjb / float(max_values['RJB'])),
                           (lnR, np.log(Rjb+duration_factor) / float(max_values['lnRJB'])),
                           (F, Fsite)])\
                * float(max_values.iloc[0, i+3])
        else:
            expr = s.subs([(M, Mw), (R, Rjb), (lnR, np.log(Rjb)), (F, Fsite)])
        expr_str = str(expr)
        result = float(np.exp(eval(expr_str)))
        result_all.append(result)
    return result_all


def Snn_pre_smooth(coeficients, mw, rjb, vs30, duration_factor):
    result_all = []
    for i in range(coeficients.shape[0]):
        Fsite = calculate_fsite(vs30)
        func = [1, mw, rjb, Fsite, np.log(rjb+duration_factor), mw*np.log(rjb+duration_factor), mw**2,
                mw*rjb, mw*Fsite, rjb*np.log(rjb+duration_factor), (np.log(rjb+duration_factor))**2, 
                np.log(rjb+duration_factor)*Fsite, rjb*Fsite, rjb**2, Fsite**2]
        re = np.exp(sum(x * y for x, y in zip(func, list(coeficients.iloc[i, :]))))
        result_all.append(re)
    return result_all


def pre_scaling(coeficients, mw, rjb, vs30, duration_factor):
    pre_all = []
    for i in range(len(mw)):
        pre = Snn_pre_smooth(coeficients, mw[i], rjb[i], vs30[i], duration_factor)
        pre_all.append(pre)
    return pre_all


def plot_scatter(x, y, xlabel, ylabel, xlim, ylim, filename):
    plt.scatter(x, y, c='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=300)


def main(file_label, work_dir, y_loc, duration_factor):
    coeficients = pd.read_excel(file_label)
    max_values = pd.read_excel('../Data/max_values_NGA.xlsx', engine='openpyxl')
    # Rjb scaling
    rjb = np.linspace(1, 300, 300)
    mw4 = [4 for i in range(len(rjb))]
    mw5 = [5 for i in range(len(rjb))]
    mw6 = [6 for i in range(len(rjb))]
    mw7 = [7 for i in range(len(rjb))]
    vs30 = [760 for i in range(len(rjb))]
    depth1 = [10 for i in range(len(rjb))]
    rjb_mw4_pre = pre_scaling(coeficients, mw4, rjb, vs30, duration_factor)
    rjb_mw5_pre = pre_scaling(coeficients, mw5, rjb, vs30, duration_factor)
    rjb_mw6_pre = pre_scaling(coeficients, mw6, rjb, vs30, duration_factor)
    rjb_mw7_pre = pre_scaling(coeficients, mw7, rjb, vs30, duration_factor)
    col4 = [i+'_Mw4' for i in max_values.columns[y_loc:]]
    col5 = [i+'_Mw5' for i in max_values.columns[y_loc:]]
    col6 = [i+'_Mw6' for i in max_values.columns[y_loc:]]
    col7 = [i+'_Mw7' for i in max_values.columns[y_loc:]]
    df_rjb = pd.DataFrame(np.concatenate((np.array(rjb_mw4_pre), np.array(rjb_mw5_pre),
                                            np.array(rjb_mw6_pre), np.array(rjb_mw7_pre)), axis=1))
    df_rjb.columns = col4 + col5 + col6 + col7
    df_rjb.to_excel(work_dir+'Rjb_scaling.xlsx', engine='openpyxl', index=False)
    # mw scaling
    mw_scali = np.arange(4, 8.2, 0.2)
    rjb_scali_20 = [20 for i in range(len(mw_scali))]
    rjb_scali_50 = [50 for i in range(len(mw_scali))]
    rjb_scali_100 = [100 for i in range(len(mw_scali))]
    rjb_scali_200 = [200 for i in range(len(mw_scali))]
    vs30_scali = [760 for i in range(len(mw_scali))]
    depth_scali =[10 for i in range(len(mw_scali))]
    mw_rjb20_pre = pre_scaling(coeficients, mw_scali, rjb_scali_20, vs30_scali, duration_factor)
    mw_rjb50_pre = pre_scaling(coeficients, mw_scali, rjb_scali_50, vs30_scali, duration_factor)
    mw_rjb100_pre = pre_scaling(coeficients, mw_scali, rjb_scali_100, vs30_scali, duration_factor)
    mw_rjb200_pre = pre_scaling(coeficients, mw_scali, rjb_scali_200, vs30_scali, duration_factor)
    col4_scali = [i + '_Rjb20' for i in max_values.columns[y_loc:]]
    col5_scali = [i + '_Rjb50' for i in max_values.columns[y_loc:]]
    col6_scali = [i + '_Rjb100' for i in max_values.columns[y_loc:]]
    col7_scali = [i + '_Rjb200' for i in max_values.columns[y_loc:]]
    df_mw_scali = pd.DataFrame(np.concatenate((np.array(mw_rjb20_pre), np.array(mw_rjb50_pre),
                                                np.array(mw_rjb100_pre), np.array(mw_rjb200_pre)), axis=1))
    df_mw_scali.columns = col4_scali + col5_scali + col6_scali + col7_scali
    df_mw_scali.to_excel(work_dir+'Mw_scaling.xlsx', engine='openpyxl', index=False)
    #vs30_scaling
    vs30_scale = list(np.arange(10, 2000, 10))
    r50_vs = [10 for i in range(len(vs30_scale))]
    r100_vs = [100 for i in range(len(vs30_scale))]
    m4_vs = [4 for i in range(len(vs30_scale))]
    m5_vs = [5 for i in range(len(vs30_scale))]
    m6_vs = [6 for i in range(len(vs30_scale))]
    m7_vs = [7 for i in range(len(vs30_scale))]
    vs_m4_r50 = pre_scaling(coeficients, m4_vs, r50_vs, vs30_scale, duration_factor)
    vs_m4_r100 = pre_scaling(coeficients, m4_vs, r100_vs, vs30_scale, duration_factor)
    vs_m5_r50 = pre_scaling(coeficients, m5_vs, r50_vs, vs30_scale, duration_factor)
    vs_m5_r100 = pre_scaling(coeficients,  m5_vs, r100_vs, vs30_scale, duration_factor)
    vs_m6_r50 = pre_scaling(coeficients, m6_vs, r50_vs, vs30_scale, duration_factor)
    vs_m6_r100 = pre_scaling(coeficients,  m6_vs, r100_vs, vs30_scale, duration_factor)
    vs_m7_r50 = pre_scaling(coeficients, m7_vs, r50_vs, vs30_scale, duration_factor)
    vs_m7_r100 = pre_scaling(coeficients,  m7_vs, r100_vs, vs30_scale, duration_factor)
    col_vsm4r50_scali = [i + '_m4_Rjb50' for i in max_values.columns[y_loc:]]
    col_vsm4r100_scali = [i + '_m4_Rjb100' for i in max_values.columns[y_loc:]]
    col_vsm5r50_scali = [i + '_m5_Rjb50' for i in max_values.columns[y_loc:]]
    col_vsm5r100_scali = [i + '_m5_Rjb100' for i in max_values.columns[y_loc:]]
    col_vsm6r50_scali = [i + '_m6_Rjb50' for i in max_values.columns[y_loc:]]
    col_vsm6r100_scali = [i + '_m6_Rjb100' for i in max_values.columns[y_loc:]]
    col_vsm7r50_scali = [i + '_m7_Rjb50' for i in max_values.columns[y_loc:]]
    col_vsm7r100_scali = [i + '_m7_Rjb100' for i in max_values.columns[y_loc:]]
    df_vs_scali = pd.DataFrame(np.concatenate((np.array(vs_m4_r50), np.array(vs_m4_r100),
                                                np.array(vs_m5_r50), np.array(vs_m5_r100),
                                                np.array(vs_m6_r50), np.array(vs_m6_r100),
                                                np.array(vs_m7_r50), np.array(vs_m7_r100)), axis=1))
    df_vs_scali.columns = col_vsm4r50_scali+col_vsm4r100_scali+col_vsm5r50_scali+col_vsm5r100_scali\
                            +col_vsm6r50_scali+col_vsm6r100_scali+col_vsm7r50_scali+col_vsm7r100_scali
    df_vs_scali.to_excel(work_dir + 'Vs30_scaling.xlsx', engine='openpyxl', index=False)
    # psa_scaling
    mw_psa = [6.5]
    rjb_psa = [10, 25, 60, 150]
    vs30_psa = [270, 760]
    depth_psa = [10]
    df_psa = pd.DataFrame()
    for i in range(len(mw_psa)):
        psa1 = pre_scaling(coeficients,  [mw_psa[i]], [rjb_psa[0]], [vs30_psa[0]],duration_factor)
        psa2 = pre_scaling(coeficients, [mw_psa[i]], [rjb_psa[0]], [vs30_psa[1]], duration_factor)
        psa3 = pre_scaling(coeficients, [mw_psa[i]], [rjb_psa[1]], [vs30_psa[0]], duration_factor)
        psa4 = pre_scaling(coeficients, [mw_psa[i]], [rjb_psa[1]], [vs30_psa[1]], duration_factor)
        psa5 = pre_scaling(coeficients, [mw_psa[i]], [rjb_psa[2]], [vs30_psa[0]], duration_factor)
        psa6 = pre_scaling(coeficients,  [mw_psa[i]], [rjb_psa[2]], [vs30_psa[1]], duration_factor)
        psa7 = pre_scaling(coeficients,  [mw_psa[i]], [rjb_psa[3]], [vs30_psa[0]], duration_factor)
        psa8 = pre_scaling(coeficients,  [mw_psa[i]], [rjb_psa[3]], [vs30_psa[1]], duration_factor)
        df = pd.DataFrame(np.concatenate((np.array(psa1).T, np.array(psa2).T, np.array(psa3).T, np.array(psa4).T,np.array(psa5).T, np.array(psa6).T, np.array(psa7).T, np.array(psa8).T), axis=1))
        df.columns = ['Mw'+str(mw_psa[i])+'Rjb10vs270', 'Mw'+str(mw_psa[i])+'Rjb10vs760',
                        'Mw'+str(mw_psa[i])+'Rjb25vs270', 'Mw'+str(mw_psa[i])+'Rjb25vs760',
                        'Mw'+str(mw_psa[i])+'Rjb60vs270', 'Mw'+str(mw_psa[i])+'Rjb60vs760',
                        'Mw'+str(mw_psa[i])+'Rjb150vs270', 'Mw'+str(mw_psa[i])+'Rjb150vs760',]
        df_psa = pd.concat([df_psa, df], axis=1)
        df_psa.to_excel(work_dir+'PSA_pre.xlsx', engine='openpyxl', index=False)
        
if __name__ == '__main__':
    file_label = '../Result/coef_smoothed_pgsnn.xlsx'
    work_dir = '../Result/'
    y_loc = 3
    duration_factor = 10
    main(file_label, work_dir, y_loc, duration_factor)
