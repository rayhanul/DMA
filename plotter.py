
import matplotlib.pyplot as plt 
import datetime 
import numpy as np 


class Plotter: 

    def __init__(self) -> None:
        pass


    def plot_time_vs_l1_for_fixed_epsilon(self, key, data, total_epsilon):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Time")
        plt.ylabel(r'$L_1$ metric')
        plt.title(r"$\epsilon$={0}".format(total_epsilon), fontsize=14, color='blue', fontweight='bold')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time {total_epsilon}_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()
        plt.close()

    


    def plot_epsilons_vs_l1(self, key, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Epsilon")
        plt.ylabel("Utility")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Epsilons_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

    def plot_bar_char_epsilons_vs_l1(self, key, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        bar_width=0.10
        r1 = np.arange(len(l1_r2dps))
        r2 = [x + bar_width for x in r1]

        plt.figure(figsize=(10, 6))
        plt.bar(r1, l1_r2dps, color='blue', width=bar_width, edgecolor='grey', label='R2DP')
        plt.bar(r2, l1_gaussian, color='green', width=bar_width, edgecolor='grey', label='Gaussian')


        plt.xlabel(r"$\epsilon$")
        plt.xticks([r + bar_width/2 for r in range(len(key))], key)
        plt.ylabel(r'$L_1$ metric')
        plt.title(r"$\epsilon$ vs $l_1$ metric", fontsize=14, color='blue', fontweight='bold')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"bar_char_epsilons_vs_l1_{timestamp}.png"
        plt.savefig(file_name)



        plt.show()


    def plot_delta_vs_l1(self, key, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        key=np.log(key)

        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel(r"$\delta$")
        plt.ylabel("Utility")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"delta_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

        
    def plot_delta_vs_usefulness(self, key, data):

        usefulness_r2dps=[ values['useful_R2DP'] for key, values in data.items()]
        usefulness_gaussian=[ values['useful_Gaussian'] for key, values in data.items()]
        
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


    def plot_l1_for_different_time(self, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        key=data.keys()

        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Time")
        plt.ylabel("Utility")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Utility_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

    def plot_epsilon_for_different_time(self, data_R2DP, data_Gaussian):

        epsilon_r2dps=[ values['epsilon_R2DP'] for key, values in data_R2DP.items()]
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
        plt.savefig(file_name)

        plt.show()



    def plot_usefulness_for_different_time(self, data):

        useful_r2dps=[ values['useful_R2DP'] for key, values in data.items()]
        useful_gaussian=[ values['useful_Gaussian'] for key, values in data.items()]
        
        key=data.keys()

        plt.figure(figsize=(10, 6))
        plt.plot(key, useful_r2dps, '-b', label='Usefulness R2DP')
        plt.plot(key, useful_gaussian, '-r', label='Usefulness Gaussian')

        plt.xlabel("Time")
        plt.ylabel("Usefulness")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Usefulness_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()


        