#!/usr/bin/env python
import argparse
import os
import time
from tqdm import tqdm


def argument_parser():
    p = argparse.ArgumentParser(description='Download data from observational facilities')
    p.add_argument('file_list', type=str, help='Path to the file containing list of files to download')
    p.add_argument('destination', type=str, help='Destination to download to')
    p.add_argument('--dryrun', help='dry-run only', action='store_true')
    return p.parse_args()


def sync(args, file_list, destination):
    flags = "-avz"
    if args.dryrun:
        flags = flags + "n"

    with open(file_list, "r") as f:
        files = f.readlines()

    # Use tqdm to wrap around files list for progress bar
    for file in tqdm(files, desc="Downloading", unit="file"):
        file = file.strip()
        if file:
            comm = f"rsync {flags} -e 'ssh -oHostKeyAlgorithms=+ssh-rsa' ops@10.2.5.115:{file} {destination}"
            os.system(comm)


def main():
    start_time = time.time()
    args = argument_parser()
    sync(args, args.file_list, args.destination)
    print("Time taken: ", time.time() - start_time)


if __name__ == '__main__':
    main()