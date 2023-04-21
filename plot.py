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
    args = parser.parse_args()
    
    if args.mode == 'rating':
        plot = plot_rating(args.path, args.title)
    elif args.mode == 'training':
        plot = plot_training(args.path, args.title)

    if args.save is not None:
        now = dt.datetime.now()
        formatted_datetime = now.strftime('%Y%m%d_%H%M%S')
        plot.savefig(f"{args.save}/{formatted_datetime}_{args.mode}_plot.png", dpi=args.dpi)
    if args.show is True:        
        plot.show(block=True)

def plot_rating(path, title):
    # Load the CSV file using pandas
    df = pd.read_csv(path)

    # Get a list of all player keys in the DataFrame
    player_keys = df['player_key'].unique()

    # Initialize variables to track x and y values
    x_vals = []
    y_rating_vals = {}
    y_deviation_vals = {}
    confidence_intervals = {}

    # Loop through each player key
    for player_key in player_keys:
        # Filter the DataFrame to only rows with this player key
        player_df = df[df['player_key'] == player_key]
        
        # Loop through each row for this player key
        for i, row in player_df.iterrows():       
            
            # Add the rating value for this player key
            if player_key not in y_rating_vals:
                y_rating_vals[player_key] = []
            y_rating_vals[player_key].append(row['rating'])
            
            # Add the rating deviation value for this player key
            if player_key not in y_deviation_vals:
                y_deviation_vals[player_key] = []
            y_deviation_vals[player_key].append(row['rating_deviation'])

            # Add the confidence interval for this player key
            if player_key not in confidence_intervals:
                confidence_intervals[player_key] = []
            confidence_intervals[player_key].append(row['rating_deviation'] * 2 )

    # Get amount of rows in the DataFrame
    num_rows = df.shape[0]
    # Create a list from 0 to the number of rows divided by the number of players
    for i in range(0, (int)(num_rows/len(player_keys))):
        x_vals.append(i)

    # Create a new plot with all data
    plt.figure(figsize=(10,6))
    color_list = plt.cm.tab20(np.linspace(0, 1, len(player_keys))) # Create a list of colors for each player key
    for i, player_key in enumerate(player_keys):
        plt.plot(x_vals, y_rating_vals[player_key], label=f'Rating ({player_key})', color=color_list[i])
        plt.errorbar(x_vals, y_rating_vals[player_key], yerr=confidence_intervals[player_key], fmt='o', capsize=4, color=color_list[i])


    # Add labels and legend to the plot
    plt.xlabel('Rating Period')
    plt.ylabel('Rating and Rating Deviation (Errorbars)')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Player Ratings Over Time')
    # plot the legend below the plot   
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    plt.subplots_adjust(bottom=0.25)    
    return plt

def plot_training(path, title):
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

    data['Reward'] = data['Value']
    data['Time [h]'] = data['Wall time']
    data['Time [h]'] = data['Time [h]'].apply(datetime.datetime.fromtimestamp)
    # Calculate the difference between the first and last datetime values
    difference = data['Time [h]'].iloc[-1] - data['Time [h]'].iloc[0]
    # Calculate the difference in hours
    hours = difference.total_seconds() / 3600
    # calculate time past since training started
    data['Time [h]'] = data['Time [h]'].apply(lambda x: (x - data['Time [h]'].iloc[0]).total_seconds() / 3600)

    data_wide = data.pivot(index='Time [h]', columns='Agent', values='Reward')
    for i in range(8):
        data_wide[f'Agent {i}'] = data_wide[f'Agent {i}'].ewm(alpha=(1-0.999)).mean()

    print(f"Total training time in hours: {hours}")

    # Plot the data
    sns.set_style("darkgrid")
    if title is not None:
        sns.lineplot(data=data_wide).set_title(title)
    else:
        sns.lineplot(data=data_wide).set_title('Durchschnittliche Reward-Entwicklung')    
    plt.ylabel('Reward')
    return plt

def check_columns(file_path, columns):
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        return all(column in header for column in columns)
    
if __name__ == '__main__':
    main()


