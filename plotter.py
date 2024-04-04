
import matplotlib.pyplot as plt 
import datetime 
import numpy as np 


class Plotter: 

    def __init__(self) -> None:
        pass

    def plot_epsilons_vs_l1(self, key, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        key=np.log(key)

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


    def plot_delta_vs_l1(self, key, data):

        l1_r2dps=[ values['l1_R2DP'] for key, values in data.items()]
        l1_gaussian=[ values['l1_Gaussian'] for key, values in data.items()]
        
        key=np.log(key)

        plt.figure(figsize=(10, 6))
        plt.plot(key, l1_r2dps, '-b', label='L1 R2DP')
        plt.plot(key, l1_gaussian, '-r', label='L1 Gaussian')

        plt.xlabel("Delta")
        plt.ylabel("Utility")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"delta_vs_l1_{timestamp}.png"
        plt.savefig(file_name)

        plt.show()

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


        