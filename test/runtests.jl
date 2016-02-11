#!/usr/bin/env julia

using Base.Test
using DataFrames
import SloanDigitalSkySurvey: SDSS, PSF, WCSUtils

include("test_sdss.jl")
include("test_wcs.jl")
