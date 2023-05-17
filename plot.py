import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import datetime as dt
import glob
import csv
import seaborn as sns
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="Specify which kind of data supported by the Modular Reinforcement Learning Framework you want to plot. Choose between 'rating' or 'training'.",
                        type=str, required=True)
    parser.add_argument('-p', '--path', help="Specify the path to the data file.", type=str, required=True)
    parser.add_argument('-s', '--save', help="Save the plot to the specified path.", type=str, required=False)
    parser.add_argument('-sh', '--show', help="Show the plot.", type=bool, required=False)
    parser.add_argument('--dpi', help="Specify the DPI of the plot. Hence the resolution if the plot should be saved.", type=int, required=False, default=400)
    parser.add_argument('-t', '--title', help="Specify the title of the plot.", type=str, required=False, default=None)
    parser.add_argument('-vt', '--value_type', help="Specify the value type of the plot. Choose between 'Reward', 'Loss', ...", type=str, required=False, default='Reward')
    parser.add_argument('-sl', '--show_legend', help="Show the legend of the plot.", type=bool, required=False, default=True)
    parser.add_argument('-sm', '--smoothing', help="Specify the smoothing factor of the plot. Choose between 0 and 0,999.", type=float, required=False, default=0.99)
    parser.add_argument('-ro', '--remove_outliers', help="Remove outliers from the plot.", type=bool, required=False, default=True)
    args = parser.parse_args()
    
    if args.mode == 'rating':
        plot = plot_rating(args.path, args.title, args.show_legend)
    elif args.mode == 'training':
        plot = plot_training(args.path, args.title, args.value_type, args.show_legend, args.smoothing, args.remove_outliers)

    if args.save is not None:
        now = dt.datetime.now()
        formatted_datetime = now.strftime('%Y%m%d_%H%M%S')
        plot.savefig(f"{args.save}/{formatted_datetime}_{args.mode}_plot.png", dpi=args.dpi)
    if args.show is True:        
        plot.show(block=True)

def plot_rating(path, title, show_legend=True):
    df = pd.read_csv(path)
    player_keys = df['player_key'].unique()

    x_vals = []
    y_rating_vals = {}
    y_deviation_vals = {}
    confidence_intervals = {}
    y_volatility_vals = {}

    for player_key in player_keys:
        player_df = df[df['player_key'] == player_key]
        
        for i, row in player_df.iterrows():       
            if player_key not in y_rating_vals:
                y_rating_vals[player_key] = []
            y_rating_vals[player_key].append(row['rating'])
            
            if player_key not in y_deviation_vals:
                y_deviation_vals[player_key] = []
            y_deviation_vals[player_key].append(row['rating_deviation'])

            if player_key not in confidence_intervals:
                confidence_intervals[player_key] = []
            confidence_intervals[player_key].append(row['rating_deviation'] * 2 )
            
            if player_key not in y_volatility_vals:
                y_volatility_vals[player_key] = []
            y_volatility_vals[player_key].append(row['volatility'])

    num_rows = df.shape[0]
    for i in range(0, (int)(num_rows/len(player_keys))):
        x_vals.append(i)

    fig, axs = plt.subplots(2, figsize=(10,12))
    color_list = plt.cm.tab20(np.linspace(0, 1, len(player_keys)))
    linestyles = ['-', '--', '-.', ':']
    for i, player_key in enumerate(player_keys):
        axs[0].plot(x_vals, y_rating_vals[player_key], label=f'Rating ({player_key})', color=color_list[i], linestyle=linestyles[i % len(linestyles)])
        axs[0].errorbar(x_vals, y_rating_vals[player_key], yerr=confidence_intervals[player_key], fmt='o', capsize=4, color=color_list[i])
        axs[1].plot(x_vals, y_volatility_vals[player_key], label=f'Volatility ({player_key})', color=color_list[i], linestyle=linestyles[i % len(linestyles)])

    axs[0].set_xlabel('Rating Period')
    axs[0].set_ylabel('Rating and Rating Deviation (Errorbars)')
    axs[1].set_xlabel('Rating Period')
    axs[1].set_ylabel('Volatility σ')

    if title is not None:
        fig.suptitle(title)
    else:
        fig.suptitle('Player Ratings and Volatility Over Time')

    if show_legend is True:
        axs[0].legend(loc="upper center", bbox_to_anchor=(0.25, -1.35), ncol=1)
        axs[1].legend(loc="upper center", bbox_to_anchor=(0.75, -0.15), ncol=1)
    plt.subplots_adjust(bottom=0.25, hspace=0.2)
    return plt
   
def plot_training(path, title, value_type='Reward', show_legend=True, smoothing=0.99, remove_outliers=True):
    # Add wildcard support for path
    path = path + '/*.csv'
    columns_to_check = ['Value', 'Wall time', 'Step']
    tensorboard_files = []
    for file in glob.glob(path):
        if check_columns(file, columns_to_check):
            tensorboard_files.append(file)
            print(f"Plotting data from {file} because it does contain the columns to qualify as a tensorboard export.")
        else:
            print(f"Skipping {file} because it does not contain the columns to qualify as a tensorboard export.")

    # read each csv file in directory and add a column for the agent number
    data = pd.concat([pd.read_csv(f).assign(Agent=f"Agent {i}") for i, f in enumerate(tensorboard_files)], ignore_index = True)

    data[value_type] = data['Value']
    data['Time [h]'] = data['Wall time']
    data['Time [h]'] = data['Time [h]'].apply(datetime.datetime.fromtimestamp)
    # Calculate the difference between the first and last datetime values
    difference = data['Time [h]'].iloc[-1] - data['Time [h]'].iloc[0]
    # Calculate the difference in hours
    hours = difference.total_seconds() / 3600
    # calculate time past since training started
    data['Time [h]'] = data['Time [h]'].apply(lambda x: (x - data['Time [h]'].iloc[0]).total_seconds() / 3600)

    data_wide = data.pivot(index='Time [h]', columns='Agent', values=value_type)
    for i in range(len(tensorboard_files)):
        data_wide[f'Agent {i}'] = data_wide[f'Agent {i}'].ewm(alpha=(1-smoothing)).mean()

    print(f"Total training time in hours: {hours}")

    # Remove outliers if specified
    if remove_outliers is True:
        data = remove_outliers1(data, 'Value')
    data[value_type] = data['Value']
    data['Time [h]'] = data['Wall time']
    data['Time [h]'] = data['Time [h]'].apply(datetime.datetime.fromtimestamp)
    # Calculate the difference between the first and last datetime values
    difference = data['Time [h]'].iloc[-1] - data['Time [h]'].iloc[0]
    # Calculate the difference in hours
    hours_no_outliers = difference.total_seconds() / 3600
    # calculate time past since training started
    data['Time [h]'] = data['Time [h]'].apply(lambda x: (x - data['Time [h]'].iloc[0]).total_seconds() / 3600)

    start_hours = hours - hours_no_outliers
    # add start hours to the data
    data['Time [h]'] = data['Time [h]'].apply(lambda x: x + start_hours)

    data_wide = data.pivot(index='Time [h]', columns='Agent', values=value_type)
    for i in range(len(tensorboard_files)):
        data_wide[f'Agent {i}'] = data_wide[f'Agent {i}'].ewm(alpha=(1-smoothing)).mean()

    # Plot the data
    sns.set_style("darkgrid")
    if title is not None:
        sns.lineplot(data=data_wide).set_title(title)
    else:
        sns.lineplot(data=data_wide).set_title(f'{value_type }-Entwicklung')    
    plt.ylabel(f'{value_type } (Exponentiell geglätteter Durchschnitt)')
    # legend
    if show_legend is True:
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    return plt

def check_columns(file_path, columns):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        return all(column in header for column in columns)
    
def remove_outliers1(data, column):
    '''
    Removes outliers from a dataframe using the IQR method.
    '''
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
if __name__ == '__main__':
    main()
