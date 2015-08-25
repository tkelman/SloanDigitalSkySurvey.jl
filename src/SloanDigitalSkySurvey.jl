module SloanDigitalSkySurvey

require(joinpath(Pkg.dir("SloanDigitalSkySurvey"), "src", "WCS.jl"))
require(joinpath(Pkg.dir("SloanDigitalSkySurvey"), "src", "PSF.jl"))
require(joinpath(Pkg.dir("SloanDigitalSkySurvey"), "src", "SDSS.jl"))

import WCS
import PSF
import SDSS

end # module
