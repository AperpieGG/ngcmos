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
    # Create dictionaries to map TIC_ID to RMS and Tmag for easier access
    rms_ratio = []
    tmag_values = []

    # Ensure the data files have the same TIC_IDs
    for entry1, entry2 in zip(data1, data2):
        if entry1['TIC_ID'] == entry2['TIC_ID']:
            rms1 = entry1['RMS']
            rms2 = entry2['RMS']
            tmag = entry1['Tmag']
            if rms2 != 0:  # Avoid division by zero
                ratio = rms1 / rms2
                rms_ratio.append(ratio)
                tmag_values.append(tmag)
        else:
            raise ValueError(f"Mismatched TIC_IDs: {entry1['TIC_ID']} and {entry2['TIC_ID']}")

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