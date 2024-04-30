import random
import numpy as np 
from scipy import integrate
import scipy.stats as stats
from plotter import Plotter 

from scipy.stats import gamma, laplace 


class DynamicMomentR2DP:

    def __init__(self, number_paramters):
        random.seed(40)
        np.random.seed(30)
        self.DEFAULT_ALPHAS= [x for x in range(2, 50)]
        self.K=np.random.randint(1,20,number_paramters)
        self.THETA=np.linspace(0.001, 10, number_paramters)
        self.total_Time=30 
        

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
        
    def M(self, k, theta, alpha):
        """
        return the moment generating function of a gamma distribution for t : (1-(x * theta))^(-k)
        """

        return np.power((1- np.multiply(alpha, theta)), -k)
    

    def get_l1_R2DP(self, t, k, theta, previous_utility):
        """
        return l1 utility for R2DP Mechanism 
        """

        def integrand(x, k_list, theta_list):
            product=1 
            for k, theta in zip(k_list, theta_list):
                product *= self.M(k,theta, -x)
            return product
        
        k_list= [values['k'] for key, values in previous_utility.items()]
        k_list.append(k)
        theta_list= [values['theta'] for key, values in previous_utility.items()]
        theta_list.append(theta)

        integral_value, error_estimate = integrate.quad(integrand, 0, np.inf, args=(k_list, theta_list))
        # l1_R2DP_upto_t=0
        # if len(previous_utility)>0:
        #     l1_R2DP_upto_t= np.sum([values['l1_R2DP'] for key, values in previous_utility.items()])
        # total_l1= integral_value + l1_R2DP_upto_t

        return integral_value

    # def get_l1_R2DP(self, t, k, theta, previous_utility):
    #     """
    #     return l1 utility for R2DP Mechanism 
    #     """
    #     def get_product_M(k, theta, x, history):
    #         # M takes negative value of x
    #         t_gamma=self.M(k, theta, -x)
    #         all_mgf_gammas=1
    #         for key, val in history.items():
    #             all_mgf_gammas *= self.M(val["k"], val["theta"], -x)
    #         all_mgf_gammas *=t_gamma

    #         return all_mgf_gammas
        
    #     integral_value, error_estimate = integrate.quad(lambda t: get_product_M(k, theta, t, previous_utility), 0, np.inf)
    #     l1_R2DP_upto_t=0
    #     if len(previous_utility)>0:
    #         l1_R2DP_upto_t= np.sum([values['l1_R2DP'] for key, values in previous_utility.items()])
            

    #         # In case no positive epsilon found, and we have previous epsilons, we sum up and select the last alpha 


        
    #     total_l1= integral_value + l1_R2DP_upto_t

    #     return total_l1/t 

    def get_epsilon_R2DP(self, t, k, theta, alpha, delta, previous_epsilons):
        """
        Computed as min_{alpha in 2:200} frac{1}{alpha-1} log[  frac{alpha}{2 alpha-1} prod_{t=1}^T (1-(alpha-1)theta_t)^{-k_t} + frac{log (1/delta)}{alpha-1}]
        
        return epsilon value for R2DP Mechanism
        """

        def get_epsilon(k, theta, alpha, delta):
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
        
        # alphas_less_than_1_over_delta= [alpha for alpha in self.DEFAULT_ALPHAS if alpha < 1/theta]

        # if len(alphas_less_than_1_over_delta) ==0:
        #     alphas_less_than_1_over_delta=[random.uniform(0.0001, 1/theta) for _ in range(10)]


        # all_epsilon_values=[ get_epsilon(alpha, k, theta, delta) for alpha in alphas_less_than_1_over_delta]
        epsilon=get_epsilon(k, theta, alpha, delta)
        # positive_epsilons = [(value, idx) for idx, value in enumerate(all_epsilon_values) if value > 0]

        # if not positive_epsilons:
        #     if len(previous_epsilons)>0:
        #         epsilon_upto_t= np.sum([values['epsilon_R2DP'] for key, values in previous_epsilons.items()])
        #         all_alphas=[values['alpha'] for key, values in previous_epsilons.items()]
        #         last_alpha=all_alphas[-1]
        #         # In case no positive epsilon found, and we have previous epsilons, we sum up and select the last alpha 
        #         return epsilon_upto_t,   last_alpha 
        #     return None, None 
        
        # min_epsilon, min_alpha = min(positive_epsilons, key=lambda x: x[0])

        # epsilon_upto_t=0
        # if len(previous_epsilons)>0:
        #     epsilon_upto_t= np.sum([values['epsilon_R2DP'] for key, values in previous_epsilons.items()])
        # total_epsilon=epsilon_upto_t  
        return epsilon 

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
        # min_val= np.min([ self.M(k, theta, -gamma) for theta in self.THETA])
        
        alphas_less_than_1_over_theta= [alpha for alpha in self.DEFAULT_ALPHAS if alpha < (1/theta)]

        min_val=np.prod([self.M(k, theta, -1* alpha* gamma) for alpha in alphas_less_than_1_over_theta])



        return 1-min_val

    def get_Gaussian(self, time, delta, total_epsilon):

        """
        time- number of iterations R2DP is run
        total_epsilon-  total budget 
        delta - 
        """
        Gaussian_l1_epsilon_utility={}
        
        sigma=self.get_optimum_sigma_gaussian(time, total_epsilon, delta)
        for time in range(1,time+1):
            Gaussian_l1_epsilon_utility.update({time:{
                'epsilon':self.get_epsilon_gaussian(time, sigma, delta), 
                'l1':self.get_l1_Gaussian(sigma, time)}})
            
        return Gaussian_l1_epsilon_utility

    def get_R2DP_nosies(self, sigma, delta, total_epsilon, budget_spend_pattern="normal", step=0.001):
        """
        sigma: 
        delta: 
        total_epsilon: budget 
        
        budget_spend_pattern: 
                normal - 
                early: spend too much early and reduces spending over time
                late: increases spending over time 


        It captures optimal k, theta, alpha, l1_R2DP, l1_Gaussian and store in a variable 

        
        return all paramters optimal value 

        """

        valid_paramters=[(k, theta, alpha) for alpha in self.DEFAULT_ALPHAS for theta in self.THETA for k in self.K if alpha < (1/theta) and k > (((-1) * (np.log((2*alpha)-1)/alpha)) / (np.log(1-((alpha-1) * theta))))]
        valid_paramters = sorted(valid_paramters, key=lambda valid_paramters: valid_paramters[0])
        # print(f'param length : {len(valid_paramters)}')
        t=1
        epsilon_R2DP=0
        l1_R2DP=0
        previous_epsilons_utility={}
        
        sigma=self.get_optimum_sigma_gaussian(self.total_Time, total_epsilon, delta)
        # 
        while epsilon_R2DP <= total_epsilon:

            # epsilon_Gaussian_t= self.get_epsilon_gaussian(t, sigma, delta)
            

            # sigma=self.get_optimum_sigma_gaussian(t, total_epsilon, delta)
            l1_R2DP_optimal=float("inf")
            best_params={}


            for k_theta_alpha in valid_paramters:
                k=k_theta_alpha[0]
                theta=k_theta_alpha[1]
                alpha=k_theta_alpha[2]

                epsilon_R2DP_t = self.get_epsilon_R2DP(t, k, theta, alpha, delta, previous_epsilons_utility )
                
                # spending pattern

                if budget_spend_pattern=="late": 
                    epsilon_R2DP_t=epsilon_R2DP_t+ step * t
                elif budget_spend_pattern=="early":
                    epsilon_R2DP_t=epsilon_R2DP_t - step * t


                if epsilon_R2DP_t==None or epsilon_R2DP_t<0:
                    continue
                
                # print(f'R2DP epsilon at time {t} : {epsilon_R2DP_t} total epsilon: {(total_epsilon/self.total_Time) * t }')
                if epsilon_R2DP_t < (total_epsilon/self.total_Time) * t :

                    l1_R2DP=self.get_l1_R2DP( t, k, theta, previous_epsilons_utility)

                    # l1_Gaussian=self.get_l1_Gaussian(sigma, t) # optimum sigma value based on epsilon


                    # usefulness_Gaussian=self.get_usefullness_Gaussian(epsilon_Gaussian_t, delta, sigma)

                    if l1_R2DP < l1_R2DP_optimal:
                        l1_R2DP_optimal=l1_R2DP
                        epsilon_R2DP=epsilon_R2DP_t
                        best_params={
                            'k':k, 
                            'theta':theta, 
                            'alpha': alpha,
                            'l1': l1_R2DP_optimal,
                            'epsilon': epsilon_R2DP_t, 
                            'useful': 0, 
                            # 'l1_Gaussian': l1_Gaussian,
                            # 'epsilon_Gaussian': epsilon_Gaussian_t ,
                            # 'useful_Gaussian':usefulness_Gaussian
                            }
            

            # if len(best_params)>0:
            #     usefulness_R2DP=self.get_usefullness_R2DP(k, best_params['theta'], best_params['epsilon'], delta)
            #     best_params['useful']=usefulness_R2DP
            #     previous_epsilons_utility.update({t:best_params})  
            # else: 
            #     key=list(previous_epsilons_utility.keys())[-1]
            #     val=previous_epsilons_utility[key]
            #     previous_epsilons_utility.update({t:val})  

            if len(best_params)>0:
                usefulness_R2DP=self.get_usefullness_R2DP(k, best_params['theta'], best_params['epsilon'], delta)
                # print(f"Time: {t}, epsilon: {best_params['epsilon']}")
                best_params['useful']=usefulness_R2DP
                previous_epsilons_utility.update({t:best_params})  
            else: 
                break 

            t=t+1
            
        return previous_epsilons_utility


    def get_R2DP_Gaussian_noise(self):

        def laprnd(mu, b, size):
            return laplace.rvs(loc=mu, scale=b, size=size)

        num_samples=1200
        valid_paramters=[(k, theta, alpha) for alpha in self.DEFAULT_ALPHAS for theta in self.THETA for k in self.K if alpha < (1/theta) and k > ((-1) * (np.log(2*alpha-1)/alpha) / (np.log(1-(alpha-1) * theta)))]
        R2DP_noise=[]
        Gaussian_noise=[]
        random_params=random.sample(valid_paramters, 100)
        prev_gamma=0
        for param in random_params: 

            gamma_sample1 = gamma.rvs(param[0], scale=param[1], size=num_samples)
            # gamma_sample1 = gamma.rvs(2, scale=1, size=num_samples)
            # gamma_sample2 = gamma.rvs(k2, scale=theta2, size=ns)
            # gamma_sample3 = gamma.rvs(k3, scale=theta3, size=ns)

            # Calculate inverses
            prev_gamma += gamma_sample1
            inverse_sum1 = 1. / prev_gamma
            # inverse_sum2 = 1. / (gamma_sample1 + gamma_sample2)
            # inverse_sum3 = 1. / (gamma_sample1 + gamma_sample2 + gamma_sample3)

            # Generate Laplace samples
            laplace_sample1 = laprnd(0, inverse_sum1, num_samples)
            gaussian_sample=np.random.laplace(0, inverse_sum1, num_samples)
            # laplace_sample2 = laprnd(0, inverse_sum2, ns)
            # laplace_sample3 = laprnd(0, inverse_sum3, ns)

            # Compute L1 distances
            R2DP_noise.append(np.mean(np.abs(laplace_sample1)))
            Gaussian_noise.append(np.mean(np.abs(gaussian_sample)))

            # y2.append(np.mean(np.abs(laplace_sample2)))
            # y3.append(np.mean(np.abs(laplace_sample3)))         

        return R2DP_noise, Gaussian_noise 

















        
