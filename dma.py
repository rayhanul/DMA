import numpy as np 
from scipy.stats import gamma, uniform, truncnorm
from scipy import integrate
import statistics
from random import randint
import itertools
import matplotlib.pyplot as plt 


DELTA=10**(-5)
C=1
SIGMA=1.2
EPSILON=1.5
T=10


K=np.linspace(1,100,50)  #shape paramter of gamma distribution. 
K=np.round(K).astype(int) # Ensuring K as integer.
THETA=np.linspace(1,100,50) #scale paramter of gamma distribution .

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def sensitivity():
    """
    this return the sensitivitiy, we consider it as 1. 

    """
    return 1 

def get_gaussian_epsilon_T(t, alpha, sigma, delta):

    value= (alpha * t) / (2 * sigma**2 ) + np.log(delta)/(alpha-1)

    return value 
def M(x, k, theta):

    return (1 + x * theta)**(-k) 

def get_product_M_T_times(f, times):

    return f**times  

def compute_std_r2dp(alpha, k, theta, t):
    l1_r2dp=np.float128(1.0)
    l1_r2dp, _ = integrate.quad(lambda x: get_product_M_T_times(M(x, k, theta), t), 0, 1/theta)
    return l1_r2dp

def get_product_T(t, alpha, theta, k):

    values=[ (1-((alpha-1) * theta))**(-k) for t1 in range(1,t+1)]
    return np.prod(values)
    
def get_log_value(t, alpha, theta, k, DELTA):

    value=(alpha/((2 * alpha) -1)) * get_product_T(t, alpha, theta, k) + (np.log((1/DELTA)))/(alpha-1) 
    val= np.log(value)
    return val 


def get_minimum_for_alphas_T(t, DEFAULT_ALPHAS, theta, k, DELTA):

    all_values=[ (1/(alpha-1)) * get_log_value(t, alpha, theta, k, DELTA) for alpha in DEFAULT_ALPHAS]
    min_index=np.argmin(all_values)
    min_value= all_values[min_index]
    alpha=DEFAULT_ALPHAS[min_index]

    return min_index, min_value, alpha


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
            _, min_value, alpha = get_minimum_for_alphas_T(t, DEFAULT_ALPHAS, theta, k, DELTA)
            gaussian_epsilon=get_gaussian_epsilon_T(t, alpha, SIGMA, DELTA)

            if min_value <=gaussian_epsilon:
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


