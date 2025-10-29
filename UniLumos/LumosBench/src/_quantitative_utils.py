# -*- coding:utf-8 -*-

import cv2
import numpy as np
import colour


def calculate_image_cct_and_illuminance(image, rgb_space='sRGB'):
    """
    Calculate the correlated color temperature (CCT) and illuminance of an image.

    Parameters:
    - image: OpenCV image in BGR format, uint8 type.
    - rgb_space: String, the color space of the image, default is 'sRGB'.

    Returns:
    - cct: Float, correlated color temperature in Kelvin.
    - illuminance: Float, estimated illuminance based on the Y component of XYZ.
    - xy: List, chromaticity coordinates of the image.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    image_rgb = image_rgb.astype(np.float32) / 255.0

    # Calculate average RGB values
    avg_rgb = np.mean(image_rgb, axis=(0, 1))

    # Get color space object
    if isinstance(rgb_space, str):
        rgb_space = colour.RGB_COLOURSPACES[rgb_space]

    # Convert average RGB to CIE XYZ
    xyz = colour.RGB_to_XYZ(
        avg_rgb,
        rgb_space.whitepoint,
        rgb_space.whitepoint,
        rgb_space.matrix_RGB_to_XYZ
    )

    # Convert XYZ to chromaticity coordinates xy
    xy = colour.XYZ_to_xy(xyz)

    # Calculate CCT using 'Kang 2002' method
    cct = colour.temperature.xy_to_CCT(xy, method='Kang 2002')

    # Illuminance is the Y component of XYZ
    illuminance = xyz[1]

    # Return CCT, illuminance, and xy coordinates
    return cct, illuminance, xy