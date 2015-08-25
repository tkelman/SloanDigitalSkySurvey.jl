#!/usr/bin/env julia

using SloanDigitalSkySurvey
include(joinpath(Pkg.dir("SloanDigitalSkySurvey"), "test", "test_sdss.jl"))
include(joinpath(Pkg.dir("SloanDigitalSkySurvey"), "test", "test_wcs.jl"))
