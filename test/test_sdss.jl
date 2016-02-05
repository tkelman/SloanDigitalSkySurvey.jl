println("Running sdss tests.")

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"

function test_load_psf()
  raw_psf_comp =
    SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, 1);
  raw_psf = PSF.get_psf_at_point(10.0, 10.0, raw_psf_comp);

  # Check that the raw psf is approximately symmetric
  center_point = (size(raw_psf)[1] + 1) / 2
  @test_approx_eq_eps(sum(collect(1:size(raw_psf)[1]) .* raw_psf) / sum(raw_psf),
                      center_point, 0.5)
  @test_approx_eq_eps(sum(collect(1:size(raw_psf)[1])' .* raw_psf) / sum(raw_psf),
                      center_point, 0.5)

end


function test_load_field()
  b = 1
  band_gain, band_dark_variance =
    SDSS.load_photo_field(field_dir, run_num, camcol_num, field_num)
  nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
    SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num,
                        b, band_gain[b])

  @test size(sky_image) == size(nelec)
  @test length(calib_col) == size(nelec)[1]
  @test sky_image[10, 5] == sky_grid[sky_x[10], sky_y[5]]

  nelec_original = deepcopy(nelec)
  SDSS.mask_image!(nelec, field_dir, run_num, camcol_num, field_num, b);

  # Test that at least some pixels are bad
  @test sum(isnan(nelec)) > 0

  # Test that no more than 1% are bad.  Not sure this is a good test
  # in general but it works for the current files.
  @test sum(isnan(nelec)) / prod(size(nelec)) < 0.01
end


function test_load_catalog()
  # Just check that the catalog loads.
  cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
  @test nrow(cat_df) == 805
end

test_load_psf()
test_load_field()
test_load_catalog()
