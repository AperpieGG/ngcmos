#!/usr/bin/env python3

#
#  Copyright 2024 Morgan Mitchell, All Rights Reserved
#

import argparse
import astropy.coordinates.name_resolve as name_resolve
import astropy.units as u
import ephem
import json
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun, get_body
from astropy.time import Time
from astropy.visualization import quantity_support
from matplotlib.legend_handler import HandlerLine2D
from path import Path
from pylab import *
from math import floor

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def is_visible(alt, az):
    min_alt = 30
    return alt >= min_alt


def update_handle(handle, orig):
    handle.update_from(orig)
    handle.set_alpha(1)


quantity_support()

parser = argparse.ArgumentParser(description='Calculate visibility of targets from NGTS on Paranal')

parser.add_argument('targets', type=str, nargs='+', help='List of targets to check visibility for')
parser.add_argument('--plot', action='store_true', help='Plot visibility')
parser.add_argument('--date', type=str, default=None, metavar='dd/mm/yyyy', help='Observing date (default: today)')
parser.add_argument('--json', type=str, default=None, help='Directory to save json file to (default: no json file)')

args = parser.parse_args()
plot = args.plot
observing_date = datetime.datetime.now().strftime("%d/%m/%Y") if args.date is None else args.date
unfiltered_targets = args.targets

try:
    observing_date = f"{observing_date.split('/')[0].zfill(2)}/{observing_date.split('/')[1].zfill(2)}/{observing_date.split('/')[2] if len(observing_date.split('/')[2]) == 4 else '20' + observing_date.split('/')[2]}"
except IndexError:
    print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
    exit()
formatted_date = f"{observing_date.split('/')[2]}-{observing_date.split('/')[1]}-{observing_date.split('/')[0]}"
reverse_date = formatted_date.replace('-', '')  # YYYYMMDD for the json file

paranal = EarthLocation(lat=-24.615744 * u.deg, lon=-70.391032 * u.deg, height=2489 * u.m)

paranal_ephem = ephem.Observer()
moon = ephem.Moon()
paranal_ephem.lat = paranal.lat.value
paranal_ephem.lon = paranal.lon.value
paranal_ephem.elevation = paranal.height.value
try:
    paranal_ephem.date = formatted_date
except ValueError:
    print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
    exit()
moon.compute(paranal_ephem)
moon_illumination = moon.moon_phase * 100

print(f"Observing date: {observing_date} (night of)")
print(f"Moon illumination: {moon_illumination:.0f}%")

try:
    utc_midnight = Time(f'{formatted_date} 23:59:00') + 1 * u.min
except ValueError:
    print(f"Invalid observing date: {observing_date} (usage: --date dd/mm/yyyy)")
    exit()
utc_offset = 3
delta_midnight = np.linspace(-12 + utc_offset, 12 + utc_offset, 1441) * u.hour  # One minute intervals
times = utc_midnight + delta_midnight
frame = AltAz(obstime=times, location=paranal)
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
night_start = (times[sunaltaz.alt < -18 * u.deg][0] - utc_midnight).to('hr')
night_end = (times[sunaltaz.alt < -18 * u.deg][-1] - utc_midnight).to('hr')
flat_time = sunaltaz.alt < -4 * u.deg
flat_start = times[flat_time][0]
flat_end = times[flat_time][-1]

print(
    f"Night + Astronomical Twilight length:     {floor((astro_twilight_end - astro_twilight_start).value):<2}h {floor(((astro_twilight_end - astro_twilight_start).value % 1) * 60):<2}m"
    f" - from {(utc_midnight + astro_twilight_start).strftime('%H:%M')} to {(utc_midnight + astro_twilight_end).strftime('%H:%M')}")
print(
    f"Night length: {' ' * 27} {floor((night_end - night_start).value):<2}h {floor(((night_end - night_start).value % 1) * 60):<2}m"
    f" - from {(utc_midnight + night_start).strftime('%H:%M')} to {(utc_midnight + night_end).strftime('%H:%M')}")
print("Visibility:")

if plot:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams["font.size"] = 8
    plt.rcParams['legend.fontsize'] = 6
    plt.figure(figsize=(6, 4))
    plt.grid()
    plt.title(f"Night of {observing_date}", y=1.05)
    plt.plot(delta_midnight, sunaltaz.alt, color='r', ls='--', label='Sun')
    plt.plot(delta_midnight, moonaltaz.alt, color=[0.75] * 3, ls='--', label='Moon')
    plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -0 * u.deg, color='0.75', zorder=0)
    plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -6 * u.deg, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -12 * u.deg, color='0.25', zorder=0)
    plt.fill_between(delta_midnight, -90 * u.deg, 90 * u.deg, sunaltaz.alt < -18 * u.deg, color='k', zorder=0)
    # plt.tight_layout()

obs_starts = []
obs_ends = []
for target, targaltaz, moon_sep, sky_coordinate in zip(targets, targaltazs, moon_seps, sky_coordinates):
    alt = targaltaz.alt.value
    az = targaltaz.az.value
    visibility = is_visible(alt, az)
    above_min_alt = visibility
    dark_time = sunaltaz.alt < -12 * u.deg
    dec = sky_coordinate.dec.value if isinstance(sky_coordinate.dec.value, float) else sky_coordinate.dec.value[0]
    ra = sky_coordinate.ra.value if isinstance(sky_coordinate.ra.value, float) else sky_coordinate.ra.value[0]
    if dec > 35.4 or dec < -84.6:
        print(f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is never visible from NGTS')
        continue
    if not np.any(above_min_alt & dark_time):
        print(f'{f"{target} ({ra:.5f}°, {dec:.5f}°)":<41} is not visible from NGTS on this date ({observing_date})')
        continue
    target_start = times[above_min_alt & dark_time][0]
    obs_starts.append(target_start)
    target_end = times[above_min_alt & dark_time][-1]
    obs_ends.append(target_end)
    target_up_moon_sep = moon_sep[above_min_alt & dark_time]
    print(f"{f'{target} ({ra:.5f}°, {dec:.5f}°):':<42} {floor((target_end - target_start).value * 24)}h" +
          f" {floor(((target_end - target_start).value * 24 % 1) * 60):<2}m - from {target_start.strftime('%H:%M')} to {target_end.strftime('%H:%M')} - Mean lunar separation: {target_up_moon_sep.mean():.1f}° ({target_up_moon_sep.min():.1f}° to {target_up_moon_sep.max():.1f}°)")
    if plot:
        sc = plt.scatter(delta_midnight, targaltaz.alt, label=target, s=1,
                         alpha=[1 if is_v and is_night else 0.1 for is_v, is_night in zip(visibility, dark_time)])
        color = list(sc.get_facecolor()[0])[:3]
        lat_diff = dec - paranal.lat.value
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
    if len(new_leg) == 2:
        print("No targets visible - cannot plot")
        exit()

    plt.legend(handles=new_leg, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.,
               handler_map={Line2D: HandlerLine2D(update_func=update_handle)})
    ax = plt.gca()
    box = ax.get_position()

    plt.xlim(solar_midnight - 8 * u.hour, solar_midnight + 8 * u.hour)
    plt.ylim(0 * u.deg, 90 * u.deg)
    labels = plt.gca().get_xticks().tolist()
    labels = [(utc_midnight + int(label) * u.hour).strftime('%H:%M') for label in labels]
    ticks = plt.gca().get_xticks().tolist()
    plt.gca().set_xticks(ticks)
    plt.gca().set_xticklabels(labels)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Altitude')

    plt.show()

if args.json is not None:
    out_path = Path(args.json) / f'{reverse_date}.json'
    dome_open = flat_start - 2 * u.min
    dome_close = flat_end + 2 * u.min

    night = formatted_date
    dome = {
        "open": str(dome_open.iso).replace(" ", "T").split(".")[0] + "Z",
        "close": str(dome_close.iso).replace(" ", "T").split(".")[0] + "Z",
    }
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
                "blind_offset_dra": 1,
                "pipeline": {
                    "object": target,
                    "prefix": target
                },
                "camera": {
                    "exposure": 10,
                    "stream": True,
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
    actions += [
        {
            "type": "ShutdownCamera",
            "start": str(dome_close.iso).replace(" ", "T").split(".")[0] + "Z",
        }
    ]
    json_dict = {
        "night": night,
        "dome": dome,
        "actions": actions
    }
    out_path.write_text(json.dumps(json_dict, indent=2))
