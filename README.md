SloanDigitalSkySurvey.jl
========

#[![Build Status](https://travis-ci.org/rgiordan/SloanDigitalSkySurvey.jl.svg?branch=master)](https://travis-ci.org/rgiordan/SloanDigitalSkySurvey.jl)
#[![Coverage Status](https://coveralls.io/repos/rgiordan/SloanDigitalSkySurvey.jl/badge.svg?branch=master)](https://coveralls.io/r/rgiordan/SloanDigitalSkySurvey.jl?branch=master)

This package contains helper functions to get images, point spread functions,
and catalog entries from the Sloan Digital Sky Survey.

To get started, take a look at `bin/example_usage.jl`, which plots
the images in `dat/sample_field`.
If you have
the [`tractor`](https://github.com/dstndstn/tractor) installed you can also
run the python scripts in the `bin` directory to get similar images
for comparison or data for new parts of the sky.
(Note that there is currently a bug in the `tractor` image masking function.)
