module WCS

VERSION < v"0.4.0-dev" && using Docile

import WCSLIB
import DualNumbers

export pixel_deriv_to_world_deriv, world_coordinate_names, wcs_id


@doc """
Convert a world location to a 1-indexed pixel location.

Args:
    - wcs: A world coordinate system object
    - world_loc: Either a 2d vector of world coordinates or a matrix
                 where the world coordinates are rows.

Returns:
    - The 1-indexed pixel locations in the same shape as the input.

The frame files seem to use the order (RA, DEC) for world coordinates,
though you should check the CTYPE1 and CTYPE2 header values if in doubt.
""" ->
function world_to_pixel(wcs::WCSLIB.wcsprm, world_loc::Array{Float64})
    single_row = length(size(world_loc)) == 1
    if single_row
        # Convert to a row vector if it's a single value
        world_loc = world_loc'
    end

    # wcss2p returns 1-indexed pixel locations.
    pix_loc = WCSLIB.wcss2p(wcs, world_loc')

    if single_row
        return pix_loc[:]
    else
        return pix_loc'
    end
end


@doc """
Convert a world location to a 1-indexed pixel location using a linear
approximation to the true transform.

Args:
    - wcs_jacobian: The jacobian of the transform pixel_log = F(world_loc)
    - world_offset: The world location at which the jacobian is evaluated
    - pixel_offset: The pixel location of the world offset
    - world_loc: The world location to be translated to pixel coordinates.

Returns:
    - The 1-indexed pixel locations in the same shape as the input.
""" ->
function world_to_pixel{NumType <: Number}(
  wcs_jacobian::Matrix{Float64}, world_offset::Vector{Float64},
  pixel_offset::Vector{Float64}, world_loc::Array{NumType})
    single_row = length(size(world_loc)) == 1
    if !single_row
        # Convert to a row vector if it's an array
        world_loc = world_loc'
    end

    pix_loc = wcs_jacobian * (world_loc - world_offset) + pixel_offset

    if single_row
        return pix_loc
    else
        return pix_loc'
    end
end



@doc """
Convert a 1-indexed pixel location to a world location.

Args:
    - wcs: A world coordinate system object
    - pix_loc: Either a 2d vector of pixel coordinates or a matrix
                 where the pixel coordinates are rows.

Returns:
    - The world locations in the same shape as the input.

The frame files seem to use the order (RA, DEC) for world coordinates,
though you should check the CTYPE1 and CTYPE2 header values if in doubt.
""" ->
function pixel_to_world{NumType <: Number}(
  wcs::WCSLIB.wcsprm, pix_loc::Array{NumType})
    # C must be passed Float64 types.
    pix_loc = convert(Array{Float64}, pix_loc)
    single_row = length(size(pix_loc)) == 1
    if single_row
        # Convert to a row vector if it's a single value
        pix_loc = pix_loc'
    end

    # wcsp2s uses 1-indexed pixel locations.
    world_loc = WCSLIB.wcsp2s(wcs, pix_loc')

    if single_row
        return world_loc[:]
    else
        return world_loc'
    end
end

@doc """
Extract the real part from a 1d or 2d DualNumbers array.
""" ->
function flatten_dual_number_array{NumType <: Number}(
  world_loc::Array{DualNumbers.Dual{NumType}})
    if length(size(world_loc)) == 1
        float_world_loc = Float64[ DualNumbers.real(loc) for loc in world_loc ]
    else
        @assert length(size(world_loc)) == 2
        float_world_loc = Float64[ DualNumbers.real(world_loc[i, j]) for
                                   i=1:size(world_loc, 1), j=1:size(world_loc, 2)]
    end
    float_world_loc
end

@doc """
Special handling of the DualNumber type which cannot be passed to C.  Note
that forward differentation cannot be applied to the location for this reason.
""" ->
function world_to_pixel{NumType <: Number}(
  wcs::WCSLIB.wcsprm, world_loc::Array{DualNumbers.Dual{NumType}})
    dual_type = typeof(world_loc[1])
    convert(Array{dual_type},
            world_to_pixel(wcs, flatten_dual_number_array(world_loc)))
end

@doc """
Special handling of the DualNumber type which cannot be passed to C.  Note
that forward differentation cannot be applied to the location for this reason.
""" ->
function pixel_to_world{NumType <: Number}(
  wcs::WCSLIB.wcsprm, world_loc::Array{DualNumbers.Dual{NumType}})
    dual_type = typeof(world_loc[1])
    convert(Array{dual_type},
            pixel_to_world(wcs, flatten_dual_number_array(world_loc)))
end


@doc """
The names of the coordinate axes in a WCS coordinate system.
""" ->
function world_coordinate_names(wcs::WCSLIB.wcsprm)
    [ unsafe_load(wcs.ctype, i) for i=1:2 ]
end

@doc """
A finite-differences approximation to the jacobian of the transform
from pixel to world coordinates.

Args:
 - wcs: The world coordinate system
 - pix_loc: The location at which to evaluate the jacobian
 - world_delt: The step size for the finite difference
               approximation in world coordinates

Returns:
 - The jacobian of the transform pixel_coord = F(world_coord).  Following the
   standard definition, the pixel coordinates vary across rows and the world
   coordinates across columns.
""" ->
function pixel_world_jacobian(wcs::WCSLIB.wcsprm, pix_loc::Array{Float64, 1};
	                          world_delt=1e-3)
    world_loc = pixel_to_world(wcs, pix_loc)
    world_delt = Float64[world_delt, world_delt]

    world_loc_1 = world_loc + world_delt[1] * Float64[1, 0]
    world_loc_2 = world_loc + world_delt[2] * Float64[0, 1]

    hcat((world_to_pixel(wcs, world_loc_1) - pix_loc) ./ world_delt[1],
         (world_to_pixel(wcs, world_loc_2) - pix_loc) ./ world_delt[2])
end


@doc """
Transform a derivative of a scalar function with respect to pixel
coordinates into a derivatve with respect to world coordinates.

Args:
 - wcs: The world coordinate system object
 - df_dpix: The derivative of a scalar function with respect to pixel coordinates
 - pix_loc: The pixel location at which the derivative was taken.
 - world_delt: The step size for the finite difference
               approximation in world coordinates

Returns:
 - The derivative with respect to world coordinates.
""" ->
function pixel_deriv_to_world_deriv(
  wcs::WCSLIB.wcsprm, df_dpix::Array{Float64, 1},
  pix_loc::Array{Float64, 1}; world_delt=1e-3)
    @assert length(pix_loc) == length(df_dpix) == 2
    trans = pixel_world_jacobian(wcs, pix_loc, world_delt=world_delt)
    trans' * df_dpix
end


# A world coordinate system where the world and pixel coordinates are the same.
const wcs_id = WCSLIB.wcsprm(2,
                             cd = Float64[1 0; 0 1],
                             ctype = ["none", "none"],
                             crpix = Float64[1, 1],
                             crval = Float64[1, 1]);

end
