#! /usr/bin/env python
import json
import glob
import matplotlib.pyplot as plt


def load_json_files():
    # Load JSON files containing "rms_rel_phot" in the filename
    json_files = glob.glob("*rms_rel_phot*.json")
    if len(json_files) != 2:
        raise ValueError("Please ensure there are exactly two JSON files with 'rms_rel_phot' in their names.")

    with open(json_files[0], 'r') as f1, open(json_files[1], 'r') as f2:
        print(f"Loading {json_files[0]} and {json_files[1]}")
        data1 = json.load(f1)
        data2 = json.load(f2)

    return data1, data2


def compute_rms_ratios(data1, data2):
    # Extract fields as lists
    tic_ids1 = data1["TIC_IDs"]
    rms1 = data1["RMS_list"]
    tmag1 = data1["Tmag_list"]

    tic_ids2 = data2["TIC_IDs"]
    rms2 = data2["RMS_list"]

    if len(tic_ids1) != len(tic_ids2) or len(rms1) != len(rms2):
        raise ValueError("Mismatched data lengths between JSON files. Ensure both files have the same TIC_IDs.")

    # Compute RMS ratio and collect Tmag values
    rms_ratio = []
    tmag_values = []

    for i in range(len(tic_ids1)):
        if rms2[i] != 0:  # Avoid division by zero
            ratio = rms1[i] / rms2[i]
            rms_ratio.append(ratio)
            tmag_values.append(tmag1[i])

    return tmag_values, rms_ratio


def plot_rms_ratio(tmag_values, rms_ratio):
    plt.figure(figsize=(8, 5))
    plt.scatter(tmag_values, rms_ratio, color='blue', alpha=0.7)
    plt.xlabel('Tmag')
    plt.ylabel('RMS Ratio')
    plt.title('RMS Ratio as a function of Tmag')
    plt.grid(True)
    plt.show()


def main():
    data1, data2 = load_json_files()
    tmag_values, rms_ratio = compute_rms_ratios(data1, data2)
    plot_rms_ratio(tmag_values, rms_ratio)


if __name__ == "__main__":
    main()