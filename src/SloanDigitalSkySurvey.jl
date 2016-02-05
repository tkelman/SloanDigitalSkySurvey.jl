module SloanDigitalSkySurvey

include("PSF.jl")  # PSF must come first because SDSS imports it.
include("SDSS.jl")
include("WCSUtils.jl")

end # module
