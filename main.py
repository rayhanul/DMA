

from moment_r2dp import *

from plotter import Plotter 










if __name__=="__main__":
    sigma=1.2
    delta = 10**(-5)
    num_params=5
    total_epsilon=0.1 
    dma=DynamicMomentR2DP(num_params)

    plotter=Plotter()


    # epsilon_utility=dma.get_R2DP_nosies(sigma, delta, total_epsilon=0.5)
    
    # 1: plot time vs l1 for differnt total epsilon
    # 2: plot epsilon vs l1 considering highest l1 achieved for each epsilon 
    # 3: plot l1 for different delta in log scale considering fixed epsilon 
    # 4 : default behavior which plot l1 for differnet time

    
    type_plot=4
    #plot 1 for all T 
    if type_plot==1:
        #  0.75, 1, 1.25, 1.5, 
        total_epsilons=[0.2, 0.5, 1, 1.5, 2, 4, 8, 16]
        epsilons_utility={}
        for total_epsilon in total_epsilons:
            R2DP_data = dma.get_R2DP_nosies(sigma=1.2, delta=10**(-5), total_epsilon=total_epsilon)

            keys=list(R2DP_data.keys())
            time=list(R2DP_data.keys())[-1]

            Gaussian_data=dma.get_Gaussian(time, delta, total_epsilon)

            
            plotter.plot_time_vs_l1_for_fixed_epsilon(keys,R2DP_data, Gaussian_data,  total_epsilon)


    elif type_plot==2:

    #plot 1 data preparation ...
        # 
        total_epsilons=[0.1, 0.5, 1, 1.5, 2, 4, 8, 16]
        epsilons_utility_R2DP={}
        delta_utility_Gaussian={}
        for total_epsilon in total_epsilons:
            R2DP_data = dma.get_R2DP_nosies(sigma=1.2, delta=10**(-5), total_epsilon=total_epsilon)
            
            last_value_R2DP=list(R2DP_data.values())[-1]
            time=list(R2DP_data.keys())[-1]

            Gaussian_data=dma.get_Gaussian(time, delta, total_epsilon)
            last_value_Gaussian=list(Gaussian_data.values())[-1]


            epsilons_utility_R2DP.update({total_epsilon: {
                'l1': last_value_R2DP["l1"]
            }})


            delta_utility_Gaussian.update({total_epsilon: {
                'l1': last_value_Gaussian["l1"]
            }})



        plotter.plot_epsilons_vs_l1(total_epsilons,epsilons_utility_R2DP, delta_utility_Gaussian)
        plotter.plot_bar_char_epsilons_vs_l1(total_epsilons, epsilons_utility_R2DP, delta_utility_Gaussian)

    elif type_plot==3: 
    # plot 2 
    # , 

        # , 10**(-20), 10**(-30), 10**(-40)
    
        deltas = [10**(-1), 10**(-5), 10**(-10), 10**(-15), 10**(-20), 10**(-30), 10**(-40)]
        delta_utility_R2DP={}
        delta_utility_Gaussian={}
        for delta in deltas:

            R2DP_epsilon_utility = dma.get_R2DP_nosies(sigma=1.2, delta=delta, total_epsilon=1)

            time=list(R2DP_epsilon_utility.keys())[-1]

            gaussian_l1_epsilon=dma.get_Gaussian(time, delta, total_epsilon)

            last_value_R2DP=list(R2DP_epsilon_utility.values())[-1]

            last_value_Gaussian=list(gaussian_l1_epsilon.values())[-1]


            delta_utility_R2DP.update({delta: {
                'l1': last_value_R2DP["l1"], 
                'useful': last_value_R2DP["useful"], 
            }})

            delta_utility_Gaussian.update({delta: {
                'l1': last_value_Gaussian["l1"], 
                'useful': 0, 
            }})

        
        plotter.plot_delta_vs_l1(deltas,delta_utility_R2DP, delta_utility_Gaussian)

        # plotter.plot_delta_vs_usefulness(deltas,delta_utility_R2DP, delta_utility_Gaussian)

    
    elif type_plot==4 :
    # plot 3 : fix epsilon and delta and plot L1(eps,delta,t) over time t for both noises
        total_epsilon=1.7
        l1_budget=200
        R2DP_epsilon_utility=dma.get_R2DP_nosies(sigma, delta, total_epsilon)

        times=list(R2DP_epsilon_utility.keys())[-1]



        # all_epsilon_Gaussin= dma.get_epsilon_gaussian(times, sigma, delta)
        # sigma=dma.get_optimum_sigma_gaussian(times, total_epsilon, delta)

        # Gaussian_l1_epsilon={}
        # for time in range(times):
        #     Gaussian_l1_epsilon.update({time:{
        #         'epsilon':dma.get_epsilon_gaussian(time, sigma, delta), 
        #         'l1':dma.get_l1_Gaussian(sigma, time)}})

        gaussian_l1_epsilon=dma.get_Gaussian(times, delta, total_epsilon)

        title=r"total $\epsilon$ ={0}, $\delta = {1}$".format(total_epsilon, delta)

        plotter.plot_l1_epsilon_vs_time(R2DP_epsilon_utility, gaussian_l1_epsilon, title, total_epsilon, l1_budget, True)


        #plotting time vs noise
        noise_output=dma.get_noise_R2DP_Gaussian(R2DP_epsilon_utility, gaussian_l1_epsilon)

        plotter.plot_time_vs_noise_in_line_chart(noise_output, total_epsilon)

        plotter.plot_time_vs_noise_in_boxplot(noise_output, total_epsilon, 8)

        plotter.plot_PDF_Gaussian_And_R2DP(noise_output, total_epsilon)

        # ploting time vs l1, time vs epsilon and time vs epsilon 

        # plotter.plot_l1_for_different_time(R2DP_epsilon_utility, gaussian_l1_epsilon, title)
        # plotter.plot_epsilon_for_different_time(R2DP_epsilon_utility, gaussian_l1_epsilon, title)

        # plotter.plot_usefulness_for_different_time(R2DP_epsilon_utility, gaussian_l1_epsilon, title)

    else: 
        r2dp, gaussian = dma.get_R2DP_Gaussian_noise()

        plotter.plot_r2dp_gaussian_noise(r2dp, gaussian)
    






