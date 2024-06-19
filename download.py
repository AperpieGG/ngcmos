#!/Users/u5500483/anaconda3/bin/python
# script to rsync from serves staging archive
import argparse
import os
from datetime import time
import time
from astropy.io import fits


def argument_parser():
    p = argparse.ArgumentParser(description='Download data from observational facilities')
    p.add_argument('source', type=str, help='Destination to download to')
    p.add_argument('destination', type=str, help='Sources to download from')
    p.add_argument('--dryrun', help='dry-run only', action='store_true')
    return p.parse_args()


def sync(args, source, destination):

    flags = "-avz"
    if args.dryrun:
        flags = flags + "n"
    comm = "rsync {} --ignore-existing -e 'ssh -oHostKeyAlgorithms=+ssh-rsa' {} {}".format(flags, source, destination)
    print(comm)
    os.system(comm)


def main():
    start_time = time.time()
    args = argument_parser()
    sync(args, args.source, args.destination)
    print("Time taken: ", time.time() - start_time)


if __name__ == '__main__':
    main()
