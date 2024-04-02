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
T=15


K=np.random.randint(1,20,30)  #shape paramter of gamma distribution. 
# K=np.round(K).astype(int) # Ensuring K as integer.

def get_random_theta(num_samples):

    samples_part1 = np.round(np.random.uniform(0, 1, num_samples // 2), 2) 


    samples_part2 = np.random.uniform(0, 10, num_samples // 2)

    
    samples = np.concatenate((samples_part1, samples_part2))

    return samples

def get_random_sample_theta(num_samples):
    """
    generate samples for theta using log.space 
    """
    samples_0_1 = np.linspace(0, 1, num_samples // 2)
    samples_10e_5_0 = np.linspace(10**-5, 0, num_samples // 4)
    samples_1_100 = np.linspace(1, 100, num_samples // 4)

    theta_samples = np.concatenate((samples_0_1, samples_10e_5_0, samples_1_100))
    # print(theta_samples)
    return theta_samples


# THETA=np.round(np.random.uniform(0, 1, 50), 2) 

# THETA=np.round(np.random.uniform(1, 10, 15), 2)
# THETA=get_random_theta(20)
THETA=np.round(np.random.uniform(0, 1, 30), 2) 

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


# def get_gaussian_epsilon_T(t, alpha, sigma, delta):

#     value= (alpha * t) / (2 * sigma**2 ) + np.log(delta)/(alpha-1)

#     return value 


# def get_product_M_T_times(f, times):

#     return f**times  

# def compute_std_r2dp(alpha, k, theta, t):
#     l1_r2dp=np.float128(1.0)
#     l1_r2dp, _ = integrate.quad(lambda x: get_product_M_T_times(M(x, k, theta), t), 0, 1/theta)
#     return l1_r2dp

def get_product_T(t, alpha, theta, k):

    values=[ (1-((alpha-1) * theta))**(-k) for t1 in range(1,t+1)]
    return np.prod(values)
    
def get_log_value(t, alpha, theta, k, DELTA):

    value=(alpha/((2 * alpha) -1)) * get_product_T(t, alpha, theta, k) + (np.log((1/DELTA)))/(alpha-1) 
    if value <=0:
        return 1
    val= np.log(value)
    return val 


# def get_minimum_for_alphas_T(t, DEFAULT_ALPHAS, theta, k, DELTA):

#     all_values=[ (1/(alpha-1)) * get_log_value(t, alpha, theta, k, DELTA) for alpha in DEFAULT_ALPHAS]
#     min_index=np.argmin(all_values)
#     min_value= all_values[min_index]
#     alpha=DEFAULT_ALPHAS[min_index]

#     return min_index, min_value, alpha


def get_epsilon_R2DP_T(t, history, default_alphas, theta, k, DELTA):

    def inner_gamma(alpha, k, theta):
        """
        return (1-((\alpha-1) * theta))^(-k)
        """
        return np.power((1- np.multiply((alpha-1), theta)), -k)

    def get_epsilon(alpha, k, theta, DELTA):
        log_value=np.float128(0.0)
        values=val= (1/(alpha-1)) 
        log_value= np.multiply(alpha/((2 * alpha) -1), inner_gamma(alpha, k, theta)) 
        if len(history)>0:
            for key, val in history.items():
                log_value = np.multiply(log_value, inner_gamma(alpha, val["k"], val["theta"]))
            log_value += (np.log((1/DELTA)))/(alpha-1)

            if log_value <=0:
                log_value=1e-10
            log_output=np.log(log_value)
            values =np.multiply(values, log_output)

        values+= (np.log((1/DELTA)))/(alpha-1) 

        return values
    
    all_epsilon_values=[ get_epsilon(alpha, k, theta, DELTA) for alpha in default_alphas]

    min_epsilon_index=np.argmin(all_epsilon_values)
    min_epsilon=all_epsilon_values[min_epsilon_index]
    min_alpha=default_alphas[min_epsilon_index]

    return min_epsilon, min_alpha


def get_gamma_mgf_for_T(t, k, theta):
    
    integral_value=np.float128(1.0)
    integral_value, error_estimate = integrate.quad(lambda x: mgf_gamma(theta, k, x)**t, 0, 1/theta )

    return integral_value


def get_optimal_alpha_gaussian(time, sigma, delta):
    """

    time: number of iteration

    sigma:standard deviation

    delta: failure probability 

    return the optimal order in the gaussian mechanism
    """

    return 1+ np.sqrt((2 * sigma**2) + (np.log(1/delta)/time)) 

def get_epsilon_gaussian(time, sigma, delta):

    return (time + 2 * sigma * np.sqrt(2 * time * np.log(1/delta)))/ (2* sigma**2) 

def get_optimum_sigma_gaussian(time, epsilon_bound, delta):

    """
    return the sigma values 
    """

    minus_b=2 * np.sqrt(2 * time * np.log(1/delta))
    discriminant= np.sqrt((8 * time * np.log(1/delta)) + (8 * time * epsilon_bound))
    two_a=4 * epsilon_bound

    value_1= (minus_b + discriminant)/two_a
    value_2= (minus_b - discriminant)/two_a

    return max(value_1, value_2)

def get_utility_gaussian(time, sigma):
    """
    return l1 utility of gaussian mechanism.
    """

    return sigma * np.sqrt(time)

def mgf_gamma(theta, k, t):
    """
    return the moment generating function of a gamma distribution for negative t. 
    """
    # if theta==1.0 or theta ==1:
    #     theta=theta+0.1
    return (1 + t * theta)**(-k) 
    
def get_utility_r2dp(theta, k):
    """
    return l1 utility of R2DP considerng moment generating function of gamma distribution. 
    """
    integral_value=np.float128(0.0)

    integral_value, error_estimate = integrate.quad(lambda t: mgf_gamma(theta, k, t), 0, 1/theta)

    return integral_value

def get_product_gamma_mgf(history, x, k, theta):
    t_gamma=mgf_gamma(theta, k, x)
    all_mgf_gammas=1
    for key, val in history.items():
        all_mgf_gammas *= mgf_gamma(val["theta"], val["k"], x)
    all_mgf_gammas *=t_gamma

    return all_mgf_gammas
 
def get_utility_R2DP_T(history, theta, k):
    integral_value, error_estimate = integrate.quad(lambda t: get_product_gamma_mgf(history, t, k, theta), 0, np.inf)
    return integral_value 


def get_optimal_k_theta():
    
    all_r2dps_over_T={}
    for t in range(1,T):
        STD_R2DP_MIN = float('inf') 
        K_THETA= itertools.product(K, THETA)

        std_gause={}

        alpha_star_gaussian=get_optimal_alpha_gaussian(t, SIGMA, DELTA)
        epsilon_star_gaussian=get_epsilon_gaussian(t, SIGMA, DELTA)
        sigma_star_gaussian = get_optimum_sigma_gaussian(t, epsilon_star_gaussian, DELTA)

        for k, theta in K_THETA:
            epsilon_R2DP, min_alpha = get_epsilon_R2DP_T(t, all_r2dps_over_T, DEFAULT_ALPHAS, theta, k, DELTA)
            if epsilon_R2DP <=epsilon_star_gaussian:
                l1_r2dp= get_utility_R2DP_T(all_r2dps_over_T, theta, k)
                if STD_R2DP_MIN>l1_r2dp:
                    STD_R2DP_MIN=l1_r2dp
                    std_gause.update({'k':k, 'theta':theta, 'l1':STD_R2DP_MIN, 'epsilon': epsilon_R2DP})
        print(f"t:{t}, k: {std_gause['k']}, theta: {std_gause['theta']}, l1:{std_gause['l1']}, Epsilon:{std_gause['epsilon']}")
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

    alpha_star_gaussian=get_optimal_alpha_gaussian(T, SIGMA, DELTA)
    epsilon_star_gaussian=get_epsilon_gaussian(T, SIGMA, DELTA)
    sigma_star_gaussian = get_optimum_sigma_gaussian(T, epsilon_star_gaussian, DELTA)

    all_r2dps_over_T = get_optimal_k_theta()



    plot_r2dps(all_r2dps_over_T)


