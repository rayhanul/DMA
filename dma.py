import numpy as np 
from scipy.stats import gamma, uniform, truncnorm
import statistics
from random import randint
import itertools
import matplotlib.pyplot as plt 


DELTA=10**(-5)
C=1
SIGMA=1.2
EPSILON=1.5
T=10


K=np.linspace(1,100,50)  #shape paramter of gamma distribution  
THETA=np.linspace(1,100,50) #scale paramter of gamma distribution 

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def sensitivity():
    """
    this return the sensitivitiy, we consider it as 1. 

    """
    return 1 


def M(alpha, k, theta):

    return (1 + alpha * theta)**(-k) 

def compute_std_r2dp(alpha, k, theta, t):
    l1_r2dp=np.float128(1.0)
    for i in range(0,t):
        index=randint(1,99)
        t_values = np.linspace(0, 1/theta - 0.01, 100)

        mgf_value= (1 - theta * t_values[index]) ** (-k)
    
        l1_r2dp *=mgf_value
    return l1_r2dp 

def get_product_T(t, alpha, theta, k):
    if t==1:
        values=(1-(alpha-1) * theta)**(-k)
        return values 
    else :
        values=[ (1-(alpha-1) * theta)**(-k) for t1 in range(1,t)]
    
    return np.prod(values)
    
def get_log_value(t, alpha, theta, k, DELTA):

    value=(alpha/((2 * alpha) -1)) * get_product_T(t, alpha, theta, k) + (np.log((1/DELTA)))/(alpha-1) 

    return np.log(value)


def get_minimum_for_alphas(t, DEFAULT_ALPHAS, theta, k, DELTA):

    all_values=[ (1/(alpha-1)) * get_log_value(t, alpha, theta, k, DELTA) for alpha in DEFAULT_ALPHAS]
    min_value=np.min(all_values)
    return min_value


def get_optimal_k_theta():
    
    all_r2dps_over_T={}
    for t in range(1,T):
        STD_R2DP_MIN=10**8
        K_THETA= itertools.product(K, THETA)

        std_gause={}
        # for alpha in DEFAULT_ALPHAS:
        alpha=1
        # for k in K:
        #     for theta in THETA:

        for k, theta in K_THETA:
            if get_minimum_for_alphas(t, DEFAULT_ALPHAS, theta, k, DELTA) <=EPSILON:
                l1_r2dp=compute_std_r2dp(alpha, k, theta, t)
                if STD_R2DP_MIN>l1_r2dp:
                    STD_R2DP_MIN=l1_r2dp
                    std_gause.update({'alpha':alpha,'k':k, 'theta':theta, 'l1':STD_R2DP_MIN})
        print(f"t:{t}, alpha:{std_gause['alpha']}, k: {std_gause['k']}, and theta: {std_gause['theta']}, l1:{std_gause['l1']}")
        all_r2dps_over_T.update({t:std_gause})

    return all_r2dps_over_T






def plot_r2dps(data):

    time_series=data.keys()
    r2dp_series=[item['l1'] for item in data.values()]

    plt.plot(time_series, r2dp_series)
    plt.xlabel("Time")
    plt.ylabel("STD_R2dp")
    # plt.grid(True)
    plt.show()







if __name__=="__main__":

    all_r2dps_over_T = get_optimal_k_theta()

    plot_r2dps(all_r2dps_over_T)


