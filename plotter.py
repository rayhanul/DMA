
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

    def plot_PDF(self, data):

        print("data")

    def plot_time_vs_noise_in_boxplot(self, input_data, epsilon, num_time=6):

        file_name= f'time_output_epsilon_{epsilon}_boxplot.jpg'
        d1_r2dp =  [ values['sample_r2dp'] for key, values in input_data.items() if key < num_time]
        d2_gaussian =  [ values['sample_gaussian'] for key, values in input_data.items() if key < num_time]

        fig, ax = plt.subplots()

        # Data for plotting
        data = [group for pair in zip(d1_r2dp, d2_gaussian) for group in pair]  # Flatten and interleave d_1 and d_2

        # Positions for each dataset
        positions = [1 + 2*i for i in range(num_time-1)] + [2 + 2*i for i in range(num_time-1)]
        positions.sort()

        # Colors for differentiation
        colors = ['lightblue' if i % 2 == 0 else 'lightgreen' for i in range(2*(num_time-1))]

        # Creating box plots
        box = ax.boxplot(data, positions=positions, patch_artist=True, notch=True, widths=0.6)

        # Coloring the boxes
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Adding titles and labels
        ax.set_title('Comparison of Groups in Datasets R2DP and Gaussian')
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
        ax.set_xticks([1.5 + 2*i for i in range(num_time-1)])  # Set x-ticks to be between the groups
        ax.set_xticklabels([f'{i+1}' for i in range(num_time-1)])  # Label groups

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='R2DP'),
                        Patch(facecolor='lightgreen', label='Gaussian')]
        ax.legend(handles=legend_elements, loc='upper right')

        # Show the plot
        plt.grid(True)

        plt.savefig(file_name)
        plt.show()

    def plot_time_vs_noise_in_line_chart(self, data, epsilon):

        times=list(data.keys())
        file_name= f'time_output_epsilon_{epsilon}_line_plot.png'
        avg_output_r2dp =  [ values['avg_output_r2dp'] for key, values in data.items()]
        avg_output_gaussian =  [ values['avg_output_gaussian'] for key, values in data.items()]


        plt.figure(figsize=(10, 5))

        plt.plot(times, avg_output_r2dp, label='Output R2DP', marker='o', linestyle='-', color='b')
        plt.plot(times, avg_output_gaussian, label='Output Gaussian', marker='x', linestyle='--', color='r')

        plt.title('Output Over Time')
        plt.xlabel('Time')
        plt.ylabel('Output')
        plt.legend()

        plt.grid(True)
        plt.savefig(file_name)
        
        plt.show()

        



    def plot_l1_epsilon_vs_time(self, r2dp_data, gaussian_data, title, total_epsilon, l1_budget, is_l1_within_limit=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name=f"Time_vs_Epsilon_{total_epsilon}_Utility.png"
        if is_l1_within_limit: 


            l1_r2dps=[ values['l1'] for key, values in r2dp_data.items() if values['l1'] < l1_budget]
            l1_gaussian=[ values['l1'] for key, values in gaussian_data.items() if values['l1'] < l1_budget]

            epsilon_r2dps=[ values['epsilon'] for key, values in r2dp_data.items() if values['l1'] < l1_budget]
            epsilon_gaussian=[ values['epsilon'] for key, values in gaussian_data.items() if values['l1'] < l1_budget]

            key_r2dp=[ key for key, values in r2dp_data.items() if values['l1'] < l1_budget]
            key_gaussian=[ key for key, values in gaussian_data.items() if values['l1'] < l1_budget]
            file_name=f"Time_vs_Epsilon_{total_epsilon}_Utility_with_budget_{l1_budget}.png"
            
        else:

            key_r2dp=list(r2dp_data.keys())
            key_gaussian=list(gaussian_data.keys())

            l1_r2dps=[ values['l1'] for key, values in r2dp_data.items()]
            l1_gaussian=[ values['l1'] for key, values in gaussian_data.items()]

            epsilon_r2dps=[ values['epsilon'] for key, values in r2dp_data.items()]
            epsilon_gaussian=[ values['epsilon'] for key, values in gaussian_data.items()]

        fig, ax1=plt.subplots()

        ax1.set_xlabel('time')

        ax1.set_ylabel(r"$\epsilon$")
        line1=ax1.plot(key_r2dp, epsilon_r2dps, color='tab:red', label='Epsilon R2DP')[0]
        line2=ax1.plot(key_gaussian, epsilon_gaussian, color='tab:blue', label='Epsilon Gaussian')[0]


        ax2 = ax1.twinx() 

        ax2.set_ylabel(r"$l_1$ metric")
        line3=ax2.plot(key_r2dp, l1_r2dps, color='tab:red', label='L1 R2DP', linestyle='--')[0]
        line4=ax2.plot(key_gaussian, l1_gaussian, color='tab:blue', label='L1 Gaussian', linestyle='--')[0]

        lines = [line1, line2, line3, line4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')


        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.savefig(file_name)


        fig.tight_layout()  
        plt.show()
        plt.close()
        

        

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


        