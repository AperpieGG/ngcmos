#! /usr/bin/env python

# for TIC-260647166: NG1226-5115 809
# for TIC-455000299: NG1159-7601 813

import argparse
import numpy as np
import warnings
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy import units as u
from astroplan import FixedTarget, Observer, time_grid_from_range

warnings.filterwarnings(action='ignore', category=SyntaxWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


# Arg parser
def arg_parse():
    """
    Parse the command line arguments
    """
    p = argparse.ArgumentParser("Reduce whole actions.")

    p.add_argument('-a',
                   '--ra',
                   help='The RA of the target.',
                   default=None, type=float)

    p.add_argument('-b',
                   '--dec',
                   help='The Declination of the target',
                   default=None, type=float)

    p.add_argument('-c',
                   '--start',
                   help='The start date',
                   default=None, type=str)

    p.add_argument('-d',
                   '--end',
                   help='The end date',
                   default=None, type=str)

    p.add_argument('-e',
                   '--duration',
                   help='The duration in minutes',
                   default=30, type=int)
    p.add_argument('-f',
                   '--field',
                   help='The field name',
                   default='test', type=str)
    p.add_argument('-g',
                   '--camera',
                   help='The camera name',
                   default='test', type=int)
    return p.parse_args()


twilight = -18 * u.deg

if __name__ == "__main__":
    # First, parse the args
    args = arg_parse()

    # establish the coordinates
    coordinates = SkyCoord(args.ra, args.dec, unit='deg')
    fixed_coordinates = FixedTarget(name='test', coord=coordinates)

    # Now establish the time range
    current_time = Time(args.start + 'T12:00:00')
    end_time = Time(args.end + 'T12:00:00')
    observer = Observer.at_site('Paranal')

    while current_time < end_time:
        # First get the times
        sunset = observer.sun_set_time(
            current_time,
            which='next',
            horizon=twilight,
            n_grid_points=150,
        )
        sunrise = observer.sun_rise_time(
            current_time,
            which='next',
            horizon=twilight,
            n_grid_points=150,
        )
        time_ut = time_grid_from_range([sunset, sunrise], time_resolution=4 * u.min)

        # Calculate airmass
        airmass = observer.altaz(time_ut, fixed_coordinates).secz
        # Mask out nonsense airmasses
        masked_airmass = np.ma.array(airmass, mask=airmass < 1)

        # Now get best airmass
        best_idx = np.ma.argmin(masked_airmass)

        # Now get window eitherside
        start = time_ut[best_idx] - TimeDelta(args.duration * u.min) / 2
        end = time_ut[best_idx] + TimeDelta(args.duration * u.min) / 2

        # Now check with basic logic
        if start < sunset:
            # Start time is before sunset, so adjust
            start = sunset
            end = sunset + TimeDelta(args.duration * u.min)
        elif end > sunrise:
            # Start time is before sunset, so adjust
            start = sunrise - TimeDelta(args.duration * u.min)
            end = sunrise

            # Finally, check if duration is too long such that we need to cap either
        # We should not get here if it is reasonable
        if start < sunset:
            start = sunset
        if end > sunrise:
            end = sunrise
        duration = end - start

        start_time_str = start.datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-7]  # Include 'T' separator and milliseconds
        end_time_str = end.datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-7]  # Include 'T' separator and milliseconds

        line = '{:} {:} {:} {:} {:} 30 15 0 0'.format(args.field, args.camera,
                                                      current_time.datetime.strftime('%Y-%m-%d'),
                                                      start_time_str,
                                                      end_time_str,
                                                      duration.to_value('minute'))

        print(line)

        # Now increment the time by 1 day
        current_time = current_time + TimeDelta(1 * u.day)
