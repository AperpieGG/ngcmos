#!/usr/bin/env python

"""
This scripts grabs the diagnostics plots, shifts plots and the mp4 movies
from the previous night
"""
import os
from datetime import datetime, timedelta
import download

path = '/home/ops/data/shifts_plots'


def find_last_night():
    # Get the previous date in the format YYYYMMDD
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    return previous_date


def get_files(path, night):
    files = os.listdir(path)
    if files:
        files = [f for f in files if f.endswith(f'{night}.mp4') or f.endswith(f'{night}.pdf')]
    return files


if __name__ == "__main__":
    last_night = find_last_night()
    files = get_files(path, last_night)
    print(f"Current night directory: {last_night}")
    print(f"Files: {files}")
    files = [os.path.join(path, f) for f in files]
    print(files)


# take the absolute path of the printed files and pass them as cron-jobs with download function
