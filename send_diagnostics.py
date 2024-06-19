#!/usr/bin/env python
import os
from datetime import datetime, timedelta

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
