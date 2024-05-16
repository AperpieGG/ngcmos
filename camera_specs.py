#!/usr/bin/env python

import argparse
import math


def calculate_pixel_scale(pixel_size, focal_length, sensor_resolution):
    """
    Calculate pixel scale, horizontal and vertical FOV, and diagonal FOV of a camera
    """
    # Calculate pixel scale (arcseconds per pixel)
    pixel_scale = pixel_size * 206.265 / focal_length

    # Calculate horizontal and vertical FOV in degrees
    pixel_size_mm = pixel_size / 1000  # Convert microns to millimeters

    fov_x_deg = math.degrees(2 * math.atan((sensor_resolution[0] * pixel_size_mm) / (2 * focal_length)))
    fov_y_deg = math.degrees(2 * math.atan((sensor_resolution[1] * pixel_size_mm) / (2 * focal_length)))

    # Calculate diagonal FOV using Pythagorean theorem
    fov_diagonal_deg = math.sqrt(fov_x_deg ** 2 + fov_y_deg ** 2)

    return fov_x_deg, fov_y_deg, fov_diagonal_deg, pixel_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate pixel scale from focal length, f/ratio, and angular resolution")
    parser.add_argument("pixel_size", type=float, help="Pixel size in microns (i.e. 11 for Marana)")
    parser.add_argument("focal_length", type=float, help="Focal length of the camera in mm (i.e. 560 for NGTS)")
    parser.add_argument("sensor_resolution", type=int, nargs=2, help="Sensor resolution in pixels (width height)")
    args = parser.parse_args()

    fov_x, fov_y, fov_diagonal, pixel_scale = calculate_pixel_scale(args.pixel_size, args.focal_length, args.sensor_resolution)
    print(f"Pixel Scale: {pixel_scale:.2f} arcseconds per pixel")
    print(f"Diagonal FOV: {fov_diagonal:.2f} degrees")
