import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import json


MEASUREMENT_PATH = ""

def PlotMeasurement(data):

    # Set up the axes and the figure
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(data["scenarioname"], fontsize=16)

    # Configure the velocity axis
    ax1.set_title("Behaviour by velocity")
    ax1.set_ylabel("velocity [kph]")
    ax1.set_xlabel("time [s]")
    ax1.plot(data["timestamps"], data["speedrestriction"], "r:", label="Speed Restriction [kph]")
    ax1.plot(data["timestamps"], data["setspeed"], "g:", label="Speed Setting [kph]")
    ax1.plot(data["timestamps"], data["targetspeed"], "k.-", label="Speed Target [kph]")
    ax1.plot(data["timestamps"], data["egospeed"], "m.-", label="Speed Ego [kph]")
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
    targetRangeUpper = [targetUpperThreshold for x in data["timestamps"]]
    targetRangeLower = [targetLowerThreshold for x in data["timestamps"]]
    toleranceRangeUpper = [toleranceUpperThreshold for x in data["timestamps"]]
    toleranceRangeLower = [toleranceLowerThreshold for x in data["timestamps"]]
    ax2.plot(data["timestamps"], data["headway"], "k.-", label="Headway [s]")
    ax2.plot(data["timestamps"], targetRangeUpper, "g:")
    ax2.plot(data["timestamps"], targetRangeLower, "g:")
    ax2.plot(data["timestamps"], toleranceRangeUpper, "y:")
    ax2.plot(data["timestamps"], toleranceRangeLower, "y:")
    boxTolerance = mpatches.FancyBboxPatch(
        (data["timestamps"][0], toleranceLowerThreshold), data["timestamps"][-1] - data["timestamps"][0], toleranceUpperThreshold-toleranceLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTolerance)
    boxTarget = mpatches.FancyBboxPatch(
        (data["timestamps"][0], targetLowerThreshold), data["timestamps"][-1] - data["timestamps"][0], targetUpperThreshold-targetLowerThreshold, mpatches.BoxStyle("Round", pad=0.0))
    patches.append(boxTarget)
    collection = PatchCollection(patches, facecolor=["y", "g"], alpha=0.4)
    ax2.add_collection(collection)
    ax2.legend()
    ax2.grid()

    # Set the figure attributes and save
    fig.tight_layout()
    fig.savefig(data["scenarioname"] + "_.png")
    plt.cla()


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


# main loop
if __name__ == "__main__":

    jsonMeasurement = "42_1.json_measurement.json"
    data = read_json(jsonMeasurement)
    PlotMeasurement(data)