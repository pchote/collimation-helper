#!/usr/bin/env python3
#
# This file is part of collimation-helper.
#
# collimation-helper is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# collimation-helper is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with collimation-helper.  If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=invalid-name

import argparse as ap
import numpy as np
from scipy import optimize
import skimage.measure
import skimage.draw
from astropy.io import fits
import pyds9

def process(frame_path, ds9_title):

    # If no regions are available show the frame and ask for one
    p = pyds9.DS9(ds9_title)
    p.set('mode region')
    p.set('regions system physical')
    p.set('regions shape box')
    regions = p.get('regions').split()
    box = None
    for r in regions:
        if r.startswith('box'):
            box = r[4:-1].split(',')[:4]
            break

    p.set('preserve pan yes')
    p.set('preserve regions no')
    p.set('file ' + frame_path)
    if box is not None:
        p.set('regions', 'text {} {}'.format(box[0], box[1]) + ' # text={Processing...}')
        p.set('regions', 'box ' + ' '.join(box))
        x = int(float(box[0]))-1
        y = int(float(box[1]))-1
        w = int(float(box[2]))
        h = int(float(box[3]))

        image = fits.open(frame_path)[0].data.astype(np.float)
        bx1 = max(0, x - int((w + 1)/2))
        bx2 = min(np.shape(image)[0], x + int(w/2))
        by1 = max(0, y - int((h + 1)/2))
        by2 = min(np.shape(image)[1], y + int(h/2))
        image = image[by1:by2, bx1:bx2]

        # Assume that most of the pixels are background, so that a simple median can recover it
        BACKGROUND_MAX_DELTA = 100
        BACKGROUND_SIGMA_THRESHOLD = 5

        rough = np.median(image)
        background = np.median(image[image < rough + BACKGROUND_MAX_DELTA])
        std = np.std(image[image < rough + BACKGROUND_MAX_DELTA])
        image -= background + BACKGROUND_SIGMA_THRESHOLD * std
        image[image < 0] = 0

        mask = image > 0
        regions = skimage.measure.regionprops(mask.astype(int))
        if regions:
            bubble = regions[0]

            y0, x0 = bubble.centroid
            r = bubble.major_axis_length / 2.

            def cost(params):
                x0, y0, r0, x1, y1, r1 = params
                coords = skimage.draw.circle(y0, x0, r0, shape=image.shape)
                template = np.zeros_like(image)
                template[coords] = 1
                coords = skimage.draw.circle(y1, x1, r1, shape=image.shape)
                template[coords] = 0
                return -np.sum(template == mask)

            x0, y0, r0, x1, y1, r1 = optimize.fmin(cost, (x0, y0, r, x0, y0, r / 2))
            p.set('regions delete all')
            p.set('regions', 'box ' + ' '.join(box))
            p.set('regions', 'circle {} {} {} #color=red select=0'.format(x0 + bx1 + 1, y0 + by1 + 1, r0))
            p.set('regions', 'point {} {} #color=red select=0 point=cross'.format(x0 + bx1 + 1, y0 + by1 + 1))
            p.set('regions', 'circle {} {} {} #color=green select=0'.format(x1 + bx1 + 1, y1 + by1 + 1, r1))
            p.set('regions', 'point {} {} #color=green select=0 point=cross'.format(x1 + bx1 + 1, y1 + by1 + 1))
if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Generates a median of all frames in a NGTS action.")
    parser.add_argument('frame_path',
                        type=str,
                        help='Path to the frame to analyse.')
    parser.add_argument('--ds9-title',
                        type=str,
                        default='collimation-helper',
                        help='ds9 window title to use.')
    args = parser.parse_args()
    process(args.frame_path, args.ds9_title)
