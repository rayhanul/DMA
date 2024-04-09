
from moment_r2dp import * 



class MonteCarlo:

    def __init__(self, num_trails):
        self.trails=num_trails

    def simulate(self, gamma_params):

        number_of_parameters = 20
        dynamic_r2dp = DynamicMomentR2DP(number_of_parameters)
        

        results=[]
        for iter in range(self.trails):
            sigma = np.random.gamma(*gamma_params['sigma'])
            delta = np.random.gamma(*gamma_params['delta'])
            total_epsilon = np.random.gamma(*gamma_params['total_epsilon'])

            result = dynamic_r2dp.get_R2DP_nosies(sigma, delta, total_epsilon)

            results.append(result)

            print(f"Iteration : {iter}, epsilon R2DP: {result["epsilon_R2DP"]}, epsilon Gaussian: {result["epsilon_Gaussin"]}")
        return results





if __name__=="__main__":

    num_trails=100
    gamma_params = {
    'sigma': (2, 0.5),  
    'delta': (2, 0.1),  
    'total_epsilon': (2, 1.0)  
}
    
    monte_carlo = MonteCarlo(num_trails)

    simulation_results = monte_carlo.simulate(gamma_params)




