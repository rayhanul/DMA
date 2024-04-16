

from moment_r2dp import *

from plotter import Plotter 










if __name__=="__main__":
    sigma=1.2
    delta = 10**(-5)
    num_params=5
    dma=DynamicMomentR2DP(num_params)

    plotter=Plotter()


    # epsilon_utility=dma.get_R2DP_nosies(sigma, delta, total_epsilon=0.5)
    
    # 1: plot time vs l1 for differnt total epsilon
    # 2: plot epsilon vs l1 considering highest l1 achieved for each epsilon 
    # 3: plot l1 for different delta in log scale considering fixed epsilon 
    # 4 : default behavior which plot l1 for differnet time

    
    type_plot=6
    #plot 1 for all T 
    if type_plot==1:
        #  0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3
        total_epsilons=[0.5, 0.75, 1, 1.25, 1.5]
        epsilons_utility={}
        for total_epsilon in total_epsilons:
            result = dma.get_R2DP_nosies(sigma=1.2, delta=10**(-5), total_epsilon=total_epsilon)

            keys=list(result.keys())
            
            plotter.plot_time_vs_l1_for_fixed_epsilon(keys,result, total_epsilon)


    elif type_plot==2:

    #plot 1 data preparation ...

        total_epsilons=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        epsilons_utility={}
        for total_epsilon in total_epsilons:
            result = dma.get_R2DP_nosies(sigma=1.2, delta=10**(-5), total_epsilon=total_epsilon)

            last_value=list(result.values())[-1]

            epsilons_utility.update({total_epsilon: {
                'l1_R2DP': last_value["l1_R2DP"], 
                'l1_Gaussian': last_value["l1_Gaussian"]
            }})

        plotter.plot_epsilons_vs_l1(total_epsilons,epsilons_utility)
        plotter.plot_bar_char_epsilons_vs_l1(total_epsilons, epsilons_utility)

    elif type_plot==3: 
    # plot 2 
    # , 
    
        deltas = [10**(-2), 10**(-5), 10**(-10), 10**(-15), 10**(-20)]
        delta_utility={}
        for delta in deltas:

            result = dma.get_R2DP_nosies(sigma=1.2, delta=delta, total_epsilon=0.5)
            last_value=list(result.values())[-1]
            delta_utility.update({delta: {
                'l1_R2DP': last_value["l1_R2DP"], 
                'l1_Gaussian': last_value["l1_Gaussian"], 
                'useful_R2DP': last_value["useful_R2DP"], 
                'useful_Gaussian': last_value["useful_Gaussian"]
            }})
        
        plotter.plot_delta_vs_l1(deltas,delta_utility)

        plotter.plot_delta_vs_usefulness(deltas,delta_utility)

    
    else :
    # plot 3 : fix epsilon and delta and plot L1(eps,delta,t) over time t for both noises
        total_epsilon=1
        epsilon_utility=dma.get_R2DP_nosies(sigma=1.2, delta=10**(-5), total_epsilon=1)

        times=list(epsilon_utility.keys())[-1]

        all_epsilon_Gaussin= dma.get_epsilon_gaussian(times, sigma, delta)
        sigma=dma.get_optimum_sigma_gaussian(times, total_epsilon, delta)

        Gaussian_l1_epsilon={}
        for time in range(times):
            Gaussian_l1_epsilon.update({time:{
                'epsilon':dma.get_epsilon_gaussian(time, sigma, delta), 
                'l1':dma.get_l1_Gaussian(sigma, time)}})



        # plotter.plot_l1_for_different_time(epsilon_utility)
        plotter.plot_epsilon_for_different_time(epsilon_utility, Gaussian_l1_epsilon)
        # plotter.plot_usefulness_for_different_time(epsilon_utility)






