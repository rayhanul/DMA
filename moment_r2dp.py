import random
import numpy as np 
from scipy import integrate
import scipy.stats as stats
from plotter import Plotter 


class DynamicMomentR2DP:

    def __init__(self, number_paramters):
        self.DEFAULT_ALPHAS= [x for x in range(2, 200)]
        self.K=np.random.randint(1,20,number_paramters)
        self.THETA=np.linspace(0.001, 10, number_paramters)
        

    def get_l1_Gaussian(self, sigma, t):

        """
        sigma: 
        t: time 


        return l1 utility of gaussian mechanism.
        """
                
        return sigma * np.sqrt(t) 
    


    def get_epsilon_gaussian(self, time, sigma, delta):

        """
        return privacy of RDP mechanism with Gaussian noise

        """

        return (time + 2 * sigma * np.sqrt(2 * time * np.log(1/delta)))/ (2* sigma**2) 

    def get_optimum_sigma_gaussian(self, time, epsilon_bound, delta):

        """
        return the optimal sigma values for the Gaussian Mechanism
        """

        minus_b=2 * np.sqrt(2 * time * np.log(1/delta))
        discriminant= np.sqrt((8 * time * np.log(1/delta)) + (8 * time * epsilon_bound))
        two_a=4 * epsilon_bound

        value_1= (minus_b + discriminant)/two_a
        value_2= (minus_b - discriminant)/two_a

        return max(value_1, value_2)    
        
    def M(self, k, theta, x):
        """
        return the moment generating function of a gamma distribution for t : (1-(x * theta))^(-k)
        """

        return np.power((1- np.multiply(x, theta)), -k)

    def get_l1_R2DP(self, k, theta, previous_utility):
        """
        return l1 utility for R2DP Mechanism 
        """
        def get_product_M(k, theta, x, history):
            # M takes negative value of x
            t_gamma=self.M(k, theta, -x)
            all_mgf_gammas=1
            for key, val in history.items():
                all_mgf_gammas *= self.M(val["k"], val["theta"], -x)
            all_mgf_gammas *=t_gamma

            return all_mgf_gammas
        
        integral_value, error_estimate = integrate.quad(lambda t: get_product_M(k, theta, t, previous_utility), 0, np.inf)
        return integral_value 

    def get_epsilon_R2DP(self, t, k, theta, delta, previous_epsilons):
        """
        Computed as min_{alpha in 2:200} frac{1}{alpha-1} log[  frac{alpha}{2 alpha-1} prod_{t=1}^T (1-(alpha-1)theta_t)^{-k_t} + frac{log (1/delta)}{alpha-1}]
        
        return epsilon value for R2DP Mechanism
        """

        def get_epsilon(alpha, k, theta, delta):
            """
            This is inline funciton.
            it calculates frac{1}{alpha-1} log[  frac{alpha}{2 alpha-1} prod_{t=1}^T (1-(alpha-1)theta_t)^{-k_t} + frac{log (1/delta)}{alpha-1}
            """
            log_value=np.float128(0.0)
            log_value= np.multiply(alpha/((2 * alpha) -1), self.M(k, theta, alpha-1)) 
            if len(previous_epsilons) >0 and t>1:
                for key, val in previous_epsilons.items():
                    log_value = np.multiply(log_value, self.M( val["k"], val["theta"], (alpha-1)))
                log_value += (np.log((1/delta)))/(alpha-1)

            log_output=np.log(log_value)
            values =np.multiply((1/(alpha-1)) , log_output)
            return values
        
        alphas_less_than_1_over_delta= [alpha for alpha in self.DEFAULT_ALPHAS if alpha < 1/theta]

        all_epsilon_values=[ get_epsilon(alpha, k, theta, delta) for alpha in alphas_less_than_1_over_delta]
        
        positive_epsilons = [(value, idx) for idx, value in enumerate(all_epsilon_values) if value > 0]

        if not positive_epsilons:
            return None, None  
        min_epsilon, min_alpha = min(positive_epsilons, key=lambda x: x[0])
        
        return min_epsilon, min_alpha

    def get_usefullness_Gaussian(self, epsilon, delta, sigma, sensitivity=1 ):

        """
        it calculate usefulness of Gaussian Mechanism as 1- 2 * E(Q(x))

        where Q(x) is the Definition F.1, page 695 of R2DP paper 
        return the usefulness of Gaussian mechanism 
        """

        def Q(x):
            
            val, _=integrate.quad(lambda u: np.exp(-u**2/2), x, np.inf)
            return 1/np.sqrt(2 * np.pi) * val 
        
        # def gaussian_dist():
        #     return stats.norm.rvs()
        
        def E():
            # gamma=(sensitivity/epsilon) * np.log(1/delta)
            gamma = 0.1
            values = [ random.uniform(0,1) * Q(gamma/sigma) for _ in range(50)]

            return np.mean(values)  

        return  1- 2 * E()
    

    def get_usefullness_R2DP(self, k, theta, epsilon, delta, sensitivity=1):
        """
        return usefulness of R2DP mechanism 
        """
        # gamma= (sensitivity/epsilon) * np.log(1/delta)
        gamma=0.1
        return 1-self.M(k, theta, -gamma)

    def get_R2DP_nosies(self, sigma, delta, total_epsilon):
        """
        sigma: 
        delta: 
        total_epsilon: budget 
        

        It captures optimal k, theta, alpha, l1_R2DP, l1_Gaussian and store in a variable 

        
        return all paramters optimal value 

        """
        t=1
        epsilon_R2DP=0
        l1_R2DP=0
        previous_epsilons_utility={}
        while epsilon_R2DP <= total_epsilon:

            epsilon_Gaussian_t= self.get_epsilon_gaussian(t, sigma, delta)
            sigma=self.get_optimum_sigma_gaussian(t, epsilon_Gaussian_t, delta)
            l1_R2DP_optimal=float("inf")

            for k in self.K:

                for theta in self.THETA:

                    epsilon_R2DP_t, best_alpha = self.get_epsilon_R2DP(t, k, theta, delta, previous_epsilons_utility )
                    
                    if epsilon_R2DP_t==None:
                        continue

                    if epsilon_R2DP_t < epsilon_Gaussian_t:

                        l1_R2DP=self.get_l1_R2DP( k, theta, previous_epsilons_utility)

                        l1_Gaussian=self.get_l1_Gaussian(sigma, t) # optimum sigma value based on epsilon

                        usefulness_R2DP=self.get_usefullness_R2DP(k, theta, epsilon_R2DP_t, delta)
                        usefulness_Gaussian=self.get_usefullness_Gaussian(epsilon_Gaussian_t, delta, sigma)

                        if l1_R2DP < l1_R2DP_optimal:
                            l1_R2DP_optimal=l1_R2DP
                            epsilon_R2DP=epsilon_R2DP_t
                            previous_epsilons_utility.update({t:{
                                'k':k, 
                                'theta':theta, 
                                'alpha': best_alpha,
                                'l1_R2DP': l1_R2DP_optimal,
                                'epsilon_R2DP': epsilon_R2DP_t, 
                                'useful_R2DP': usefulness_R2DP, 
                                'l1_Gaussian': l1_Gaussian,
                                'epsilon_Gaussian': epsilon_Gaussian_t ,
                                'useful_Gaussian':usefulness_Gaussian
                            }})
                        
            t=t+1


        return previous_epsilons_utility

















        