import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import datetime as dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="Specify which kind of data supported by the Modular Reinforcement Learning Framework you want to plot. Choose between 'rating' or 'training'.",
                        type=str, required=True)
    parser.add_argument('-p', '--path', help="Specify the path to the data file.", type=str, required=True)
    parser.add_argument('-s', '--save', help="Save the plot to the specified path.", type=str, required=False)
    parser.add_argument('-sh', '--show', help="Show the plot.", type=bool, required=False)
    args = parser.parse_args()
    
    if args.mode == 'rating':
        plot = plot_rating(args.path)
    elif args.mode == 'training':
        plot = plot_training()

    if args.save is not None:
        now = dt.datetime.now()
        formatted_datetime = now.strftime('%Y%m%d_%H%M%S')
        plot.savefig(f"{args.save}/{formatted_datetime}_{args.mode}_plot.png", dpi=400)
    if args.show is True:        
        plot.show(block=True)

def plot_rating(path):
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
    plt.title('Player Ratings Over Time')
    # plot the legend below the plot   
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    plt.subplots_adjust(bottom=0.25)    
    return plt

def plot_training():
    return True

if __name__ == '__main__':
    main()


