

import numpy as np 
from scipy.stats import gamma, uniform, truncnorm
import statistics
from random import randint


DELTA=10**(-5)
C=1
SIGMA=1.2
T=10


K=np.linspace(1,100,10)  #shape paramter
THETA=np.linspace(1,100,10) #scale paramter 
DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def sensitivity():
    """
    this return the sensitivitiy, we consider it as 1. 

    """
    return 1 


def M(alpha, k, theta):

    return (1 + alpha * theta)**(-k) 

def compute_std_r2dp(alpha, k, theta, t):
    l1_r2dp=1
    for i in range(0,t):
        index=randint(1,99)
        t_values = np.linspace(0, 1/theta - 0.01, 100)

        mgf_value= (1 - theta * t_values[index]) ** (-k)
    
        l1_r2dp *=mgf_value
    return l1_r2dp 






def get_optimal_k_theta():
    all_r2dps_over_T={}
    for t in range(0,T):
        STD_R2DP_MIN=10**8
        std_gause={}
        for alpha in DEFAULT_ALPHAS:
            for k in K:
                for theta in THETA:
                    l1_r2dp=compute_std_r2dp(alpha, k, theta, t)
                    if STD_R2DP_MIN>l1_r2dp:
                        STD_R2DP_MIN=l1_r2dp
                        std_gause.update({'alpha':alpha,'k':k, 'theta':theta, 'l1':STD_R2DP_MIN})
        print(f"t:{t}, alpha:{alpha}, k: {std_gause['k']}, and theta: {std_gause['theta']}, l1:{STD_R2DP_MIN}")
        all_r2dps_over_T.update({t:std_gause})

    return all_r2dps_over_T











if __name__=="__main__":

    all_r2dps_over_T = get_optimal_k_theta()

    print("I am done")
