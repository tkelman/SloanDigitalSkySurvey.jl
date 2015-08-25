#!/usr/bin/python
#
# Load a processed image from the tractor.  Note that as of writing,
# the masking was incorrect.
# https://github.com/dstndstn/tractor

from tractor.sdss import *
import argparse
import numpy
import copy

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=int, help='The run number.', default=3900)
parser.add_argument('--camcol', type=int, help='The camcol number.', default=6)
parser.add_argument('--field', type=int, help='The field number.', default=269)
parser.add_argument('--band', type=int, help='The band (a number from 0 to 4).', default=3)
parser.add_argument('--destination_base', type=str,
	                help=('Output will be written to files like '
					      '<destination_base>_<run>_<camcol>_<field>_<description>.csv'),
					default="/tmp/test")

args = parser.parse_args()

bands = ['u', 'g', 'r', 'i', 'z']
bandname = bands[args.band]

# Note that get_tractor_image_dr9 just calls get_tractor_image_dr8
img = get_tractor_image_dr8(args.run, args.camcol, args.field, bandname, nanomaggies=True)
sources = get_tractor_sources_dr9(args.run, args.camcol, args.field,
	                              nanomaggies=True, fixedComposites=True, useObjcType=True)

file_base = args.destination_base + ('_%d_%d_%d_' % (args.run, args.camcol, args.field))
band_str = '%d_' % (args.band + 1) # Python uses 0 indexing
numpy.savetxt(file_base + band_str + "img.csv", img[0].data, delimiter=",")
numpy.savetxt(file_base + band_str + "psf.csv", img[0].psf, delimiter=",")

sdss = DR8()

# Mask the image.  Note that as of now, the
masked_img_data = copy.deepcopy(img[0].data)
fpM = sdss.readFpM(args.run, args.camcol, args.field, bandname)
for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
	fpM.setMaskedPixels(plane, masked_img_data, NaN)

print sum(numpy.isnan(masked_img_data))
numpy.savetxt(file_base + band_str + "masked_img.csv", masked_img_data, delimiter=",")

# Debugging the mask:
if False:
	size(fpM.getMaskPlane('INTERP').rmin) # INTERP is element 1 in the 0-indexed python.

	name = 'INTERP'
	masked_img_data = copy.deepcopy(img[0].data)
	val = NaN

	M = fpM.getMaskPlane(name)

	nan_pixels = 0
	for (c0,c1,r0,r1,coff,roff) in zip(M.cmin,M.cmax,M.rmin,M.rmax,
	                                   M.col0, M.row0):
	    assert(coff == 0)
	    assert(roff == 0)
	    nan_pixels = nan_pixels + (r1 - r0 + 1) * (c1 - c0 + 1)
	    masked_img_data[r0:r1, c0:c1] = val

	print nan_pixels
	print sum(numpy.isnan(masked_img_data))
