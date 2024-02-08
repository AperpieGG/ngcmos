#!/usr/bin/env python3

#  Copyright 2023 James McCormac, All Rights Reserved

"""
Fetch the scheduled observing sequence from a
given NGTS camera and produce a JSON file for
the CMOS to observe with
"""
import argparse as ap
import json
from datetime import datetime
import numpy as np
import pymysql

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

# set some flags to avoid duplicates
FOCUS_SWEEP_SCHEDULED = False
# make blocks for flats and focusing
evening_flats = {"type": "SkyFlats",
                 "evening": True,
                 "pipeline": {"prefix": "evening-flat"}}
morning_flats = {"type": "SkyFlats",
                 "evening": False,
                 "pipeline": {"prefix": "morning-flat"}}


def arg_parse():
    p = ap.ArgumentParser()
    p.add_argument('camera_id',
                   help='camera_id to follow',
                   type=int,
                   choices=np.arange(801, 814))
    p.add_argument('night',
                   help="Night to query from (e.g. 20231008)",
                   type=str)
    return p.parse_args()


def get_action_info(camera_id, night):
    qry = """
        SELECT action_id, action, schedule_start_utc, schedule_end_utc
        FROM action_list
        WHERE camera_id=%s
        AND night=%s
        ORDER BY schedule_start_utc ASC
        """
    qry_args = (camera_id, night)
    conn = pymysql.connect(host='10.2.5.32', db='ngts_ops', user='ops')
    try:
        with conn.cursor() as cur:
            cur.execute(qry, qry_args)
            action_info = cur.fetchall()
    finally:
        conn.close()
    return action_info


def get_action_args(action_id):
    qry = """
        SELECT arg_key, arg_value
        FROM action_args
        WHERE action_id=%s
        AND arg_key IN ('campaign', 'field', 'exposureTime')
        """
    qry_args = (action_id,)
    conn = pymysql.connect(host='10.2.5.32', db='ngts_ops', user='ops')
    try:
        with conn.cursor() as cur:
            cur.execute(qry, qry_args)
            action_args = cur.fetchall()
        act_args = {}
        for aa in action_args:
            act_args[aa[0]] = aa[1]
    finally:
        conn.close()
    return act_args


def get_field_coords(field):
    qry = """
        SELECT ra_centre_deg, dec_centre_deg
        FROM master_field_list
        WHERE field=%s
        """
    qry_args = (field,)
    conn = pymysql.connect(host='10.2.5.32', db='ngts_ops', user='ops')
    try:
        with conn.cursor() as cur:
            cur.execute(qry, qry_args)
            coords = cur.fetchone()
    finally:
        conn.close()
    return coords


if __name__ == "__main__":
    args = arg_parse()

    # fetch all actions for camera we are following
    action_info = get_action_info(args.camera_id, args.night)

    # create empty dict to populate
    OB = {}
    OB['night'] = datetime.strptime(args.night, "%Y%m%d").strftime("%Y-%m-%d")
    OB['actions'] = []
    OB['actions'].append(evening_flats)

    # loop over actions and add to schedule
    for action in action_info:
        action_id, action_type, start, end = action

        if action_type == "focusSweep" and not FOCUS_SWEEP_SCHEDULED:
            evening_focus = {"type": "AutoFocus",
                             "start": start.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                             "camera": {"exposure": 10.0}}
            OB['actions'].append(evening_focus)
            FOCUS_SWEEP_SCHEDULED = True

        elif action_type == "observeField":
            act_args = get_action_args(action_id)
            # build some inputs
            field = act_args['field']
            prefix = f"{act_args['field']}_{act_args['campaign']}"
            exposure = round(float(act_args['exposureTime']), 2)
            # get field coords
            ra_centre_deg, dec_centre_deg = get_field_coords(field)
            # make a science block
            science = {"type": "ObserveTimeSeries",
                       "start": start.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                       "end": end.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                       "ra": ra_centre_deg,
                       "dec": dec_centre_deg,
                       "pipeline": {"prefix": prefix,
                                    "object": prefix},
                       "camera": {"exposure": exposure}
                       }

            # finally plonk the action into the schedule
            OB['actions'].append(science)

    # schedule morning flats
    OB['actions'].append(morning_flats)

    # save the json file
    with open(f'{args.night}.json', 'w') as fp:
        json.dump(OB, fp, indent=4)

    print(f"Saved observing plan to {args.night}.json")
