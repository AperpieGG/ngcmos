#! /usr/bin/env python3

"""
visibility.py - Calculate visibility of targets from TWIST, NGTS or TMO.
copyright: (c) 2024, Morgan Mitchell, University of Warwick
"""

import argparse
import datetime
import json
from math import floor

import astropy.coordinates
import astropy.coordinates.name_resolve as name_resolve
import astropy.units as u
import ephem
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun, get_body
from astropy.time import Time
from astropy.visualization import quantity_support
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from path import Path

import matplotlib

# matplotlib.use('Qt5Agg')


class Observatory:
    def __init__(self, name, lat, lon, height, min_alt, max_hr_angle, max_sun_alt):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.height = height
        self.longitude_offset = self.lon.to(u.hourangle).value
        self.max_hr_angle = max_hr_angle
        self.max_sun_alt = max_sun_alt
        self.min_alt = min_alt

    def __str__(self):
        return f"{self.name} ({self.lat}°, {self.lon}°, {self.height}m)"

    def __repr__(self):
        return self.__str__()

    def is_unobstructed(self, alt, az, times):
        # convert alt, az to hour angle, declination using the observatory's latitude
        altaz = SkyCoord(alt=alt, az=az, unit='deg',
                         frame=AltAz(location=EarthLocation(lat=self.lat, lon=self.lon, height=self.height),
                                     obstime=times))
        hr = altaz.transform_to(astropy.coordinates.HADec).ha.value

        # check if min_alt has len attribute
        if hasattr(self.min_alt.value, '__len__'):
            min_alts = np.array(self.min_alt.to('deg').value)
            min_azs = np.linspace(0, 360, len(min_alts))
            min_alt = np.interp(az, min_azs, min_alts)
            safe_alt = alt > min_alt
            safe_hr = np.abs(hr) < self.max_hr_angle.value
            return safe_alt & safe_hr
        else:
            safe_alt = alt > self.min_alt.to('deg').value
            safe_hr = np.abs(hr) < self.max_hr_angle.value
            return safe_alt & safe_hr


twist_min_alts = np.array(
    [63, 63, 63, 62, 59, 53, 44, 35, 36, 36, 36, 35, 32, 30, 31, 37, 39, 40, 40, 40, 38, 36, 34, 23, 25, 27, 28, 30, 29,
     28, 26, 38, 50, 56, 59, 62, 63]) * u.deg

twist = Observatory("Twenty Inch Survey Telescope", 28.760255 * u.deg, -17.879284 * u.deg, 2348 * u.m, twist_min_alts,
                    6 * u.hourangle, -12 * u.deg)
ngts = Observatory("Next Generation Transit Survey", -24.615662 * u.deg, -70.391809 * u.deg, 2433 * u.m, 30 * u.deg,
                   5.3 * u.hourangle, -15 * u.deg)
tmo = Observatory("Tom Marsh Observatory", 52.376876 * u.deg, -1.567203 * u.deg, 93 * u.m, 30 * u.deg,
                  4.5 * u.hourangle, -12 * u.deg)


def update_handle(handle, orig):
    handle.update_from(orig)
    handle.set_alpha(1)


quantity_support()


def main(args):
    plot = args.plot
    observing_date = datetime.datetime.now().strftime("%d/%m/%Y") if args.date is None else args.date

    # set the observatory object based on the command line argument
    if args.observatory == 'twist':
        obs = twist
    elif args.observatory == 'ngts':
        obs = ngts
    else:
        obs = tmo

    try:
        observing_date = f"{observing_date.split('/')[0].zfill(2)}/{observing_date.split('/')[1].zfill(2)}/{observing_date.split('/')[2] if len(observing_date.split('/')[2]) == 4 else '20' + observing_date.split('/')[2]}"
    except IndexError:
        print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
        exit()
    formatted_date = f"{observing_date.split('/')[2]}-{observing_date.split('/')[1]}-{observing_date.split('/')[0]}"  # YYYY-MM-DD for the date lookup
    reverse_date = formatted_date.replace('-', '')  # YYYYMMDD for the json file name

    obs_site = EarthLocation(lat=obs.lat, lon=obs.lon, height=obs.height)

    obs_site_ephem = ephem.Observer()
    moon = ephem.Moon()
    obs_site_ephem.lat = obs_site.lat.value
    obs_site_ephem.lon = obs_site.lon.value
    obs_site_ephem.elevation = obs_site.height.value
    try:
        obs_site_ephem.date = formatted_date
    except ValueError:
        print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
        exit()
    moon.compute(obs_site_ephem)
    moon_illumination = moon.moon_phase * 100

    print(f"Observing date: {observing_date} (night of)")
    print(f"Moon illumination: {moon_illumination:.0f}%")

    try:
        utc_midnight = Time(f'{formatted_date} 23:59:00') + 1 * u.min
    except ValueError:
        print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
        exit()
    delta_midnight = np.linspace(-12 - obs.longitude_offset, 12 - obs.longitude_offset,
                                 1441) * u.hour  # One minute intervals
    times = utc_midnight + delta_midnight
    frame = AltAz(obstime=times, location=obs_site)

    try:
        if len(args.targets) == 2:
            # check if each of ra and dec strings contain two colons
            if args.targets[0].count(':') == 2:
                ra_sex = [float(num) for num in args.targets[0].split(':')]
                ra = 15 * ra_sex[0] + 15 * 60 ** -1 * ra_sex[1] + 15 * 3600 ** -1 * ra_sex[2]
            else:
                ra = float(args.targets[0])
            if args.targets[1].count(':') == 2:
                dec_sex = [float(num) for num in args.targets[1].split(':')]
                dec = dec_sex[0] + 60 ** -1 * dec_sex[1] + 3600 ** -1 * dec_sex[2]
            else:
                dec = float(args.targets[1])

            sky_coordinates = [SkyCoord(ra=ra, dec=dec, unit='deg')]
            targets = [f'({args.targets[0]}, {args.targets[1]})']
        else:
            raise ValueError
    except ValueError:
        unfiltered_targets = args.targets
        sky_coordinates = []
        targets = []
        for target in unfiltered_targets:
            try:
                sky_coordinates.append(get_body(target, times))
                targets.append(target)
            except KeyError:
                try:
                    sky_coordinates.append(SkyCoord.from_name(target))
                    targets.append(target)
                except name_resolve.NameResolveError:
                    try:
                        sky_coordinates.append(SkyCoord.from_name(target.replace(' ', '')))
                        targets.append(target)
                    except name_resolve.NameResolveError:
                        try:
                            sky_coordinates.append(SkyCoord.from_name(target.replace(' ', '-')))
                            targets.append(target)
                        except name_resolve.NameResolveError:
                            print(f"Target {target} not found")
    if len(targets) == 0:
        print("No targets found")
        exit()

    sunaltaz = get_sun(times).transform_to(frame)
    moonaltaz = get_body("moon", times).transform_to(frame)
    targaltazs = [sky_coordinate.transform_to(frame) for sky_coordinate in sky_coordinates]
    moon_seps = [targaltaz.separation(moonaltaz).value for targaltaz in targaltazs]

    solar_midnight = (times[sunaltaz.alt.argmin()] - utc_midnight).to('hr')
    astro_twilight_start = (times[sunaltaz.alt < -12 * u.deg][0] - utc_midnight).to('hr')
    astro_twilight_end = (times[sunaltaz.alt < -12 * u.deg][-1] - utc_midnight).to('hr')
    flat_time = sunaltaz.alt < -4 * u.deg
    flat_start = times[flat_time][0]
    flat_end = times[flat_time][-1]

    print(
        f"Night + Astronomical Twilight length:     {str(floor((astro_twilight_end - astro_twilight_start).value)).zfill(2)}h {str(floor(((astro_twilight_end - astro_twilight_start).value % 1) * 60)).zfill(2)}m"
        f" - from {(utc_midnight + astro_twilight_start).strftime('%H:%M')} to {(utc_midnight + astro_twilight_end).strftime('%H:%M')}")

    try:
        night_start = (times[sunaltaz.alt < -18 * u.deg][0] - utc_midnight).to('hr')
        night_end = (times[sunaltaz.alt < -18 * u.deg][-1] - utc_midnight).to('hr')
        print(
            f"Night length: {' ' * 27} {str(floor((night_end - night_start).value)).zfill(2)}h {str(floor(((night_end - night_start).value % 1) * 60)).zfill(2)}m"
            f" - from {(utc_midnight + night_start).strftime('%H:%M')} to {(utc_midnight + night_end).strftime('%H:%M')}")
    except IndexError:
        print(f"Night length: {' ' * 27} Not on this date")
    print("Visibility:")

    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams["font.size"] = 18
        plt.rcParams['legend.fontsize'] = 14
        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.title(f"Night of {observing_date}", y=1.05)
        plt.plot(delta_midnight, sunaltaz.alt, color='yellow', ls='--', label='Sun')
        plt.plot(delta_midnight, moonaltaz.alt, color=[0.75] * 3, ls='--', label='Moon')
        plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, color='#87ceeb', zorder=0, alpha=1)
        plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < 0 * u.deg, color='#000000', zorder=0,
                         alpha=0.3)
        plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -6 * u.deg, color='#000000', zorder=0,
                         alpha=0.45)
        plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -12 * u.deg, color='#000000', zorder=0,
                         alpha=0.6)
        plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -18 * u.deg, color='#000000', zorder=0,
                         alpha=0.9)

    obs_starts = []
    obs_ends = []
    for target, targaltaz, moon_sep, sky_coordinate in zip(targets, targaltazs, moon_seps, sky_coordinates):
        alt = targaltaz.alt.value
        az = targaltaz.az.value
        visibility = obs.is_unobstructed(alt, az, times)
        above_min_alt = visibility
        dark_time = sunaltaz.alt < obs.max_sun_alt
        if isinstance(sky_coordinate.dec.value, float):
            dec = sky_coordinate.dec.value
            ra = sky_coordinate.ra.value
            ss = False
        else:
            dec = sky_coordinate.dec.value[0]
            ra = sky_coordinate.ra.value[0]
            ss = True

        if np.any(above_min_alt & dark_time):
            target_start = times[above_min_alt & dark_time][0]
            obs_starts.append(target_start)
            target_end = times[above_min_alt & dark_time][-1]
            obs_ends.append(target_end)
            target_up_moon_sep = moon_sep[above_min_alt & dark_time]
            print(
                f"{f'{target} ({ra:.5f}°, {dec:.5f}°):':<41} {str(floor((target_end - target_start).value * 24)).zfill(2)}h" +
                f" {str(floor(((target_end - target_start).value * 24 % 1) * 60)).zfill(2)}m - from {target_start.strftime('%H:%M')} to {target_end.strftime('%H:%M')} - Mean lunar separation: {target_up_moon_sep.mean():.1f}° ({target_up_moon_sep.min():.1f}° to {target_up_moon_sep.max():.1f}°)")
        else:
            if ss:
                print(f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is a solar system object not visible on this night.')
            else:
                if np.any(above_min_alt):
                    try:
                        target_start = times[np.where(np.diff(above_min_alt.astype(int)) == 1)[0][0]]
                        target_end = times[np.where(np.diff(above_min_alt.astype(int)) == -1)[0][0]]
                        print(
                            f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is not visible on this night - unobstructed between {target_start.strftime("%H:%M")} and {target_end.strftime("%H:%M")}')
                    except IndexError:
                        print(f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is not visible on this date')
                else:
                    print(
                        f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is never visible from the {obs.name}')

        if plot:
            sc = plt.scatter(delta_midnight, targaltaz.alt, label=target, s=1,
                             alpha=[1 if is_v and is_night else 0.2 for is_v, is_night in zip(visibility, dark_time)])
            color = list(sc.get_facecolor()[0])[:3]
            lat_diff = dec - obs_site.lat.value
            for i, vis in zip(range(len(times) - 1), visibility[:-2]):
                alpha = 1 if vis and dark_time[i] else 0.2
                plot_color = color + [alpha]
                if ((az[i]) > 350) and ((az[i + 1]) < 10) or ((az[i]) < 10) and ((az[i + 1]) > 350):
                    plt.annotate("N", (delta_midnight[i], targaltaz.alt[i]),
                                 xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                 color=plot_color)
                    continue
                if (az[i] > 90) and (az[i + 1] < 90) or (az[i] < 90) and (az[i + 1] > 90):
                    plt.annotate("E", (delta_midnight[i], targaltaz.alt[i]),
                                 xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                 color=plot_color)
                if (az[i] > 270) and (az[i + 1] < 270) or (az[i] < 270) and (az[i + 1] > 270):
                    plt.annotate("W", (delta_midnight[i], targaltaz.alt[i]),
                                 xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                 color=plot_color)
                if (az[i] > 180) and (az[i + 1] < 180) or (az[i] < 180) and (az[i + 1] > 180):
                    plt.annotate("S", (delta_midnight[i], targaltaz.alt[i]),
                                 xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                 color=plot_color)
                if not -10 < lat_diff < 0:
                    if (az[i] > 135) and (az[i + 1] < 135) or (az[i] < 135) and (az[i + 1] > 135):
                        plt.annotate("SE", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] > 225) and (az[i + 1] < 225) or (az[i] < 225) and (az[i + 1] > 225):
                        plt.annotate("SW", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] > 247.5) and (az[i + 1] < 247.5) or (az[i] < 247.5) and (
                            az[i + 1] > 247.5):
                        plt.annotate("WSW", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] < 112.5) and (az[i + 1] > 112.5) or (az[i] > 112.5) and (
                            az[i + 1] < 112.5):
                        plt.annotate("ESE", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                if not 0 < lat_diff < 10:
                    if (az[i] > 315) and (az[i + 1] < 315) or (az[i] < 315) and (az[i + 1] > 315):
                        plt.annotate("NW", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] > 67.5) and (az[i + 1] < 67.5) or (az[i] < 67.5) and (az[i + 1] > 67.5):
                        plt.annotate("ENE", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] > 45) and (az[i + 1] < 45) or (az[i] < 45) and (az[i + 1] > 45):
                        plt.annotate("NE", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)
                    if (az[i] > 292.5) and (az[i + 1] < 292.5) or (az[i] < 292.5) and (
                            az[i + 1] > 292.5):
                        plt.annotate("WNW", (delta_midnight[i], targaltaz.alt[i]),
                                     xytext=(delta_midnight[i], targaltaz.alt[i] + 1 * u.deg), ha='center', va='bottom',
                                     color=plot_color)

    if plot:
        leg = plt.legend()
        new_leg = []
        for lh in leg.legend_handles:
            if lh.get_label() != 'Sun' and lh.get_label() != 'Moon':
                lh = Line2D([], [], color=lh.get_facecolor()[0], label=lh.get_label(), linestyle='-')
                new_leg.append(lh)
            else:
                new_leg.append(lh)
            if lh.get_label() == 'Moon':
                lh.set_label(f'{lh.get_label()} ({moon_illumination:.0f}%)')
        if len(new_leg) == 2:
            print("No targets visible - cannot plot")
            exit()

        plt.legend(handles=new_leg, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.,
                   handler_map={Line2D: HandlerLine2D(update_func=update_handle)})

        plt.xlim(solar_midnight - 9 * u.hour, solar_midnight + 9 * u.hour)
        plt.ylim(0 * u.deg, 90 * u.deg)
        labels = plt.gca().get_xticks().tolist()[1:-1]
        labels = [(utc_midnight + int(label) * u.hour).strftime('%H:%M') for label in labels]
        ticks = plt.gca().get_xticks().tolist()[1:-1]
        plt.gca().set_xticks(ticks)
        plt.gca().set_xticklabels(labels)
        plt.xlabel('Time (UTC)')
        plt.ylabel('Altitude')
        alt_tick_labels = plt.gca().get_yticks().tolist()
        ax2 = plt.gca().twinx()
        air_tick_labels = [f'{1 / np.cos(np.radians(90 - alt_tick_label)):.2f}' for alt_tick_label in alt_tick_labels]
        air_tick_labels[0] = ''
        ax2.set_yticks(alt_tick_labels)
        ax2.set_yticklabels(air_tick_labels)
        ax2.set_ylabel('Airmass')

        plt.tight_layout()
        plt.show()

    if args.json is not None:
        # do not allow if ss objects are present
        if np.any([isinstance(sky_coordinate.dec.value, np.ndarray) for sky_coordinate in sky_coordinates]):
            print("Solar system objects present - cannot create json file")
            exit()
        out_path = Path(args.json).parent / f'{reverse_date}.json'

        night = formatted_date
        actions = [
            {
                "type": "SkyFlats",
                "evening": True,
                "pipeline": {
                    "prefix": "evening-flat"
                },
            }
        ]
        for target, sky_coordinate, target_start, target_end in zip(targets, sky_coordinates, obs_starts, obs_ends):
            actions += [
                {
                    "type": "AutoFocus",
                    "start": str((target_start - 5 * u.min).iso).replace(" ", "T").split(".")[0] + "Z",
                    "camera": {
                        "exposure": 10,
                    },
                },
                {
                    "type": "ObserveTimeSeries",
                    "start": str(target_start.iso).replace(" ", "T").split(".")[0] + "Z",
                    "end": str(target_end.iso).replace(" ", "T").split(".")[0] + "Z",
                    "ra": round(sky_coordinate.ra.value, 5),
                    "dec": round(sky_coordinate.dec.value, 5),
                    "pipeline": {
                        "object": target.replace(" ", "-"),  # Replace spaces with hyphens
                        "prefix": target.replace(" ", "-"),  # Replace spaces with hyphens
                    },
                    "camera": {
                        "exposure": 10,
                    },
                }
            ]
        actions += [
            {
                "type": "SkyFlats",
                "evening": False,
                "pipeline": {
                    "prefix": "morning-flat"
                },
            }
        ]
        json_dict = {
            "night": night,
            "actions": actions
        }
        out_path.write_text(json.dumps(json_dict, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate visibility of targets from TWIST, NGTS or TMO.')

    parser.add_argument('observatory', type=str, choices=['twist', 'ngts', 'tmo'],
                        help='(e.g. ngts --date 13/05/2024 TIC 324010229 --json 20240513)')
    parser.add_argument('targets', type=str, nargs='+', help='List of targets to check visibility for')
    parser.add_argument('--plot', action='store_true', help='Plot visibility')
    parser.add_argument('--date', type=str, default=None, metavar='dd/mm/yyyy', help='Observing date (default: today)')
    parser.add_argument('--json', type=str, default=None, help='Directory to save json file to (default: no json file)')

    args = parser.parse_args()
    main(args)
