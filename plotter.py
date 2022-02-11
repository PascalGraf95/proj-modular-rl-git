import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import json
import os
import collections
import numpy as np



# ---------------------------------------------------------------------------------------
# Plot the measurement V1 -- headway and speeds
# ---------------------------------------------------------------------------------------
def PlotMeasurementV1(inputData):

    # Get the first stuff
    firstData = inputData[list(inputData.keys())[0]]

    # Set up the axes and the figure
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(firstData["scenarioname"], fontsize=16)

    # Configure the velocity axis
    ax1.set_title("Behaviour by velocity")
    ax1.set_ylabel("velocity [kph]")
    ax1.set_xlabel("time [s]")
    ax1.set_ylim(0, 120)
    ax1.set_yticks(np.arange(0, 120, step=10))
    ax1.plot(firstData["timestamps"], firstData["speedrestriction"], "r--", label="Speed Restriction [kph]")
    ax1.plot(firstData["timestamps"], firstData["setspeed"], "g--", label="Speed Setting [kph]")
    ax1.plot(firstData["timestamps"], firstData["targetspeed"], "k.-", label="Speed Target [kph]")
    
    # loop over all and pront them
    for key in inputData.keys():
        ax1.plot(inputData[key]["timestamps"], inputData[key]["egospeed"], label="Speed Ego [kph] - " + str(key))

    ax1.legend()
    ax1.grid()
    
    # Configure the headway axis
    ax2.set_title("Behaviour by headway")
    ax2.set_ylabel("headway [s]")
    ax2.set_xlabel("time [s]")
    ax2.set_ylim(0, 4)
    patches = []
    targetUpperThreshold = 2.1
    targetLowerThreshold = 1.9
    toleranceUpperThreshold = 2.25
    toleranceLowerThreshold = 1.75
    targetRangeUpper = [targetUpperThreshold for x in firstData["timestamps"]]
    targetRangeLower = [targetLowerThreshold for x in firstData["timestamps"]]
    toleranceRangeUpper = [toleranceUpperThreshold for x in firstData["timestamps"]]
    toleranceRangeLower = [toleranceLowerThreshold for x in firstData["timestamps"]]

    # loop over all and pront them
    for key in inputData.keys():
        ax2.plot(inputData[key]["timestamps"], inputData[key]["headway"], label="Headway [s] - " + str(key))
    
    ax2.plot(firstData["timestamps"], targetRangeUpper, "g:")
    ax2.plot(firstData["timestamps"], targetRangeLower, "g:")
    ax2.plot(firstData["timestamps"], toleranceRangeUpper, "y:")
    ax2.plot(firstData["timestamps"], toleranceRangeLower, "y:")
    boxTolerance = mpatches.FancyBboxPatch(
        (firstData["timestamps"][0], toleranceLowerThreshold), firstData["timestamps"][-1] - firstData["timestamps"][0], toleranceUpperThreshold-toleranceLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTolerance)
    boxTarget = mpatches.FancyBboxPatch(
        (firstData["timestamps"][0], targetLowerThreshold), firstData["timestamps"][-1] - firstData["timestamps"][0], targetUpperThreshold-targetLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTarget)
    collection = PatchCollection(patches, facecolor=["y", "g"], alpha=0.4)
    ax2.add_collection(collection)
    ax2.legend()
    ax2.grid()

    # Set the figure attributes and save
    fig.tight_layout()
    fig.savefig(firstData["scenarioname"] + ".png")
    plt.clf()
    plt.cla()
    fig = None


# ---------------------------------------------------------------------------------------
# Plot the measurement V2 -- headway, speeds and accelerartions
# ---------------------------------------------------------------------------------------
def PlotMeasurementV2(path, inputData):

    # Get the first stuff
    firstData = inputData[list(inputData.keys())[0]]

    # Set up the axes and the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(firstData["scenarioname"], fontsize=16)

    # Configure the velocity axis
    ax1.set_title("Behaviour by velocity")
    ax1.set_ylabel("velocity [kph]")
    ax1.set_xlabel("time [s]")
    ax1.set_ylim(0, 120)
    ax1.set_yticks(np.arange(0, 120, step=10))
    ax1.plot(firstData["timestamps"], firstData["speedrestriction"], "r--", label="Speed Restriction [kph]")
    ax1.plot(firstData["timestamps"], firstData["setspeed"], "g--", label="Speed Setting [kph]")
    ax1.plot(firstData["timestamps"], firstData["targetspeed"], "k.-", label="Speed Target [kph]")
    
    # loop over all and print them
    for key in inputData.keys():
        ax1.plot(inputData[key]["timestamps"], inputData[key]["egospeed"], label="Speed Ego [kph] - " + str(key))

    # Legend and grid
    ax1.legend()
    ax1.grid()
    
    # Configure the headway axis
    ax2.set_title("Behaviour by headway")
    ax2.set_ylabel("headway [s]")
    ax2.set_xlabel("time [s]")
    ax2.set_ylim(0, 4)
    patches = []
    targetUpperThreshold = 2.1
    targetLowerThreshold = 1.9
    toleranceUpperThreshold = 2.25
    toleranceLowerThreshold = 1.75
    targetRangeUpper = [targetUpperThreshold for x in firstData["timestamps"]]
    targetRangeLower = [targetLowerThreshold for x in firstData["timestamps"]]
    toleranceRangeUpper = [toleranceUpperThreshold for x in firstData["timestamps"]]
    toleranceRangeLower = [toleranceLowerThreshold for x in firstData["timestamps"]]

    # loop over all and print them
    for key in inputData.keys():
        ax2.plot(inputData[key]["timestamps"], inputData[key]["headway"], label="Headway [s] - " + str(key))
    
    # Plot the rest
    ax2.plot(firstData["timestamps"], targetRangeUpper, "g:")
    ax2.plot(firstData["timestamps"], targetRangeLower, "g:")
    ax2.plot(firstData["timestamps"], toleranceRangeUpper, "y:")
    ax2.plot(firstData["timestamps"], toleranceRangeLower, "y:")
    boxTolerance = mpatches.FancyBboxPatch(
        (firstData["timestamps"][0], toleranceLowerThreshold), firstData["timestamps"][-1] - firstData["timestamps"][0], toleranceUpperThreshold-toleranceLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTolerance)
    boxTarget = mpatches.FancyBboxPatch(
        (firstData["timestamps"][0], targetLowerThreshold), firstData["timestamps"][-1] - firstData["timestamps"][0], targetUpperThreshold-targetLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTarget)
    collection = PatchCollection(patches, facecolor=["y", "g"], alpha=0.4)
    ax2.add_collection(collection)
    ax2.legend()
    ax2.grid()

    # Configure the acceleration axis
    ax3.set_title("Behaviour by accelerations")
    ax3.set_ylabel("acceleration [m/s²]")
    ax3.set_xlabel("time [s]")
    ax3.set_ylim(-3, 3)
    ax3.set_yticks(np.arange(-3, 3, step=0.5))
    
    # loop over all and print them
    for key in inputData.keys():
        ax3.plot(inputData[key]["timestamps"], inputData[key]["acc_requested"], label="Requested Acceleration [m/s²] - " + str(key))
        #ax3.plot(inputData[key]["timestamps"], inputData[key]["acc_measured"], label="Measured Acceleration [m/s²] - " + str(key))
        #ax3.plot(inputData[key]["timestamps"], inputData[key]["acc_calc"], label="Measured calculated Acceleration [m/s²] - " + str(key))
        #ax3.plot(inputData[key]["timestamps"], inputData[key]["headway"], label="Headway [s] - " + str(key))

    # Legend and grid
    ax3.legend()
    ax3.grid()

    # Set the figure attributes and save
    fig.tight_layout()
    fig.savefig(path + firstData["scenarioname"] + ".png")
    plt.clf()
    plt.cla()
    fig = None


# ---------------------------------------------------------------------------------------
# SAVE MEASUREMENT
# ---------------------------------------------------------------------------------------
def save_measurement(jsonMeasurement, name):
    
    # Dump the measurement dict as a json file
    with open(name, 'w') as f:
        json.dump(jsonMeasurement, f)

# ---------------------------------------------------------------------------------------
# READ SCENARIO
# ---------------------------------------------------------------------------------------
def read_json(path):

    # Opening JSON file
    f = open(path,)
    
    # returns JSON object as a dictionary
    data = json.load(f)

    # return the read data
    return data


MEASUREMENT_PATH = "/home/ai-admin/Measurements/V3/"

# main loop
if __name__ == "__main__":

    scenariodict = {}

    # Loop in the folder
    for file in os.listdir(MEASUREMENT_PATH):

        # Check if the file is in fact a measurement
        if file.endswith(".json") and "measurement" in file: 
            
            #print(file)
            name = file.split("_")
            scenario = name[0] + "_" + name[1]
            version = name[-1].replace(".json", "")
            #print(scenario, version)

            # Check if the scenario is in the dict
            if scenario in scenariodict.keys():
                scenariodict[scenario].append(version)
            
            else:
                scenariodict[scenario] = [version]

    # Print the dict
    print(scenariodict)

    # Determine complete scenarios
    #versions = ["V4", "V5", "V6", "V8", "V9"]
    #versions = ["V6", "V8", "V9"]
    #versions = ["V8", "V9", "V10"]
    #versions = ["V8"]
    versions = ["V16-TRY1-CONT"]
    versions = ["V16-TRY2"]
    versions = ["V17"]

    # Determine the complete scenarios
    completeScenarios = []
    for key in scenariodict.keys():

        # Initially assume, all versions are there for the scenario
        AllVersionsThere = True

        # Loop over all versions
        for version in versions:
            if version in scenariodict[key]:
                pass
            else:
                AllVersionsThere = False

        # Check if really all are there
        if AllVersionsThere:
            completeScenarios.append(key)
    
    # Print the dict of complete scenarios
    print(completeScenarios)

    # Loop over the complete scenarios and plot each of them
    for scenario in completeScenarios:

        data = {name: read_json(MEASUREMENT_PATH + scenario + "_measurement_" + name + ".json") for name in versions}

        # Plot the measurements
        PlotMeasurementV2(MEASUREMENT_PATH, data)