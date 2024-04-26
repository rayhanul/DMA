
import matplotlib.pyplot as plt 
import datetime 
import numpy as np 


class Plotter: 

    def __init__(self) -> None:
        pass

    def plot_r2dp_gaussian_noise(self, r2dp_data, gaussian_data):

        len_keys=len(r2dp_data)

        x_axis=[i for i in range(len_keys)]

        plt.figure()
        plt.plot(x_axis, r2dp_data, label='R2DP')
        plt.plot(x_axis, gaussian_data, label='Gaussian')

        plt.xlabel('Number of Samples')
        plt.ylabel('Mean L1 Distance')
        plt.title('L1 Distance Evaluation')
        plt.legend()
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"noise_vs_mean_l1_{timestamp}.png"
        plt.savefig(file_name)
        plt.show()

        



    def plot_time_vs_l1_for_fixed_epsilon(self, key, R2DP_data, Gaussian_data,  total_epsilon):

        l1_r2dps=[ values['l1'] for key, values in R2DP_data.items()]
        l1_gaussian=[ values['l1'] for key, values in Gaussian_data.items()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Time")
        plt.ylabel(r'$l_1$ metric')
        plt.title(r"$\epsilon$={0}".format(total_epsilon), fontsize=14, color='blue', fontweight='bold')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time {total_epsilon}_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()
        plt.close()

    


    def plot_epsilons_vs_l1(self, key, data_R2DP, data_Gaussian):

        l1_r2dps=[ values['l1'] for key, values in data_R2DP.items()]
        l1_gaussian=[ values['l1'] for key, values in data_Gaussian.items()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$l_1$ metric")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Epsilons_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

    def plot_bar_char_epsilons_vs_l1(self, key, data_R2DP, data_Gaussian):

        l1_r2dps=[ values['l1'] for key, values in data_R2DP.items()]
        l1_gaussian=[ values['l1'] for key, values in data_Gaussian.items()]
        
        bar_width=0.10
        r1 = np.arange(len(l1_r2dps))
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(10, 6))
        plt.bar(r1, l1_r2dps, color='blue', width=bar_width, edgecolor='grey', label='R2DP')
        plt.bar(r2, l1_gaussian, color='green', width=bar_width, edgecolor='grey', label='Gaussian')


        plt.xlabel(r"$\epsilon$")
        plt.xticks([r + bar_width/2 for r in range(len(key))], key)
        plt.ylabel(r'$l_1$ metric')
        plt.title(r"$\epsilon$ vs $l_1$ metric", fontsize=14, color='blue', fontweight='bold')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"bar_char_epsilons_vs_l1_{timestamp}.png"
        plt.savefig(file_name)



        plt.show()


    def plot_delta_vs_l1(self, key, data_R2DP, data_Gaussian):

        l1_r2dps=[ values['l1'] for key, values in data_R2DP.items()]
        l1_gaussian=[ values['l1'] for key, values in data_Gaussian.items()]
        
        key=key

        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$l_1$ metric")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"delta_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

        
    def plot_delta_vs_usefulness(self, key, data_R2DP, data_Gaussian):

        usefulness_r2dps=[ values['usefulP'] for key, values in data_R2DP.items()]
        usefulness_gaussian=[ values['useful'] for key, values in data_Gaussian.items()]
        
        key=np.log(key)

        plt.figure(figsize=(10, 6))
        plt.plot(key, usefulness_r2dps, '-b', label='Usefulness R2DP')
        plt.plot(key, usefulness_gaussian, '-r', label='Usefulness Gaussian')

        plt.xlabel(r"$\delta$")
        plt.ylabel("Usefulness")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"delta_vs_usefulness_{timestamp}.png"
        plt.savefig(file_name)


    def plot_l1_for_different_time(self, r2dp_data, gaussian_data, title):

        l1_r2dps=[ values['l1'] for key, values in r2dp_data.items()]
        l1_gaussian=[ values['l1'] for key, values in gaussian_data.items()]
        
        key=r2dp_data.keys()

        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Time")
        plt.ylabel(r"$l_1$ metric")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Utility_{timestamp}.png"
        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.savefig(file_name)

        plt.show()
        plt.close()

    def plot_epsilon_for_different_time(self, data_R2DP, data_Gaussian, title):

        epsilon_r2dps=[ values['epsilon'] for key, values in data_R2DP.items()]
        epsilon_gaussian=[ values['epsilon'] for key, values in data_Gaussian.items()]
        
        key=data_R2DP.keys()

        plt.figure(figsize=(10, 6))
        plt.plot(key, epsilon_r2dps, '-b', label='Epsilon R2DP')
        plt.plot(key, epsilon_gaussian, '-r', label='Epsilon Gaussian')

        plt.xlabel("Time")
        plt.ylabel(r"$\epsilon$")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Epsilon_{timestamp}.png"
        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.savefig(file_name)

        plt.show()
        plt.close()



    def plot_usefulness_for_different_time(self, r2dp_data, gaussian_data, title):

        useful_r2dps=[ values['useful'] for key, values in r2dp_data.items()]
        useful_gaussian=[ values['useful'] for key, values in gaussian_data.items()]
        
        key=r2dp_data.keys()

        plt.figure(figsize=(10, 6))
        plt.plot(key, useful_r2dps, '-b', label='Usefulness R2DP')
        plt.plot(key, useful_gaussian, '-r', label='Usefulness Gaussian')

        plt.xlabel("Time")
        plt.ylabel("Usefulness")
        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Usefulness_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()


        