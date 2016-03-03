println("Running sdss tests.")


using Base.Test
import SloanDigitalSkySurvey: SDSS, PSF

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"


using ForwardDiff
using Optim
using PyPlot

const field_dir =
  joinpath(Pkg.dir("SloanDigitalSkySurvey"), "dat", "sample_field")
const run_num = "003900"
const camcol_num = "6"
const field_num = "0269"

raw_psf_comp =
  SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, 1);
psf = PSF.get_psf_at_point(10.0, 10.0, raw_psf_comp);
psf_2 = PSF.get_psf_at_point(500.0, 500.0, raw_psf_comp);
x_mat = get_x_matrix_from_psf(psf);

verbose = true

psf_max = maximum(psf)
matshow(psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("PSF")

# Why is NM the best?
# K=3 is much slower and not much better than K=2 by the looks of it.
opt_result = fit_psf_gaussians(psf, K=2,
  optim_method=Optim.NelderMead(), verbose=true);

opt_result_noop = fit_psf_gaussians(psf, K=2, initial_par=opt_result.minimum,
  optim_method=Optim.NelderMead(), verbose=true);

# Why isn't this better?
opt_result_2 = fit_psf_gaussians(psf_2, K=2, initial_par=opt_result.minimum,
  optim_method=Optim.NelderMead(), verbose=true);

# opt_result = fit_psf_gaussians(psf, initial_par=opt_result.minimum, K=2,
#   optim_method=Optim.AcceleratedGradientDescent(), verbose=true, iterations=20);
#

unwrap_parameters(opt_result.minimum)
gmm_psf = render_psf(opt_result.minimum, x_mat);
matshow(gmm_psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("fit1")
psf_residual = psf - gmm_psf;
resid_max = 1.5 * maximum(abs(psf_residual))
matshow(psf_residual, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual")
sum(psf_residual .^ 2)

# Compare to EM
gmm, scale = PSF.fit_psf_gaussians_em(psf);
em_psf = [ PSF.evaluate_gmm(gmm, x_mat[i, j]')[1] for
           i=1:size(x_mat, 1), j=1:size(x_mat, 2)];
em_psf_residual = psf - em_psf;
matshow(em_psf_residual, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("em residual")
sum(em_psf_residual .^ 2)



mu_vec = Array(Vector{T}, K)
sigma_vec = Array(Matrix{T}, K)
weight_vec = zeros(T, K)

for k = 1:K
  mu_vec[k] = gmm.μ[1, :][:]
  sigma_vec[k] = sigma_chol' * sigma_chol + sigma_min
  weight_vec[k] = exp(par[offset + 6]) + weight_min
end


PyPlot.close("all")

psf_max = maximum(psf)
matshow(psf, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("PSF")

opt_result_1 = fit_psf_gaussians(psf);
unwrap_parameters(opt_result_1.minimum)
gmm_psf1 = render_psf(opt_result_1.minimum);
psf_residual1 = psf - gmm_psf1;
matshow(gmm_psf1, vmax=1.2 * psf_max); PyPlot.colorbar(); PyPlot.title("fit1")
resid_max = 1.5 * maximum(abs(psf_residual1))
matshow(psf_residual1, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual1")


opt_result_2 = fit_psf_gaussians(psf_residual1);
unwrap_parameters(opt_result_2.minimum)
gmm_psf2 = gmm_psf1 + render_psf(opt_result_2.minimum);
psf_residual2 = psf - gmm_psf2;
#matshow(gmm_psf2); PyPlot.colorbar(); PyPlot.title("fit2")
matshow(psf_residual2, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual2")

opt_result_3 = fit_psf_gaussians(psf_residual2);
unwrap_parameters(opt_result_3.minimum)
gmm_psf3 = gmm_psf2 + render_psf(opt_result_3.minimum);
psf_residual3 = psf - gmm_psf3;
matshow(psf_residual3, vmax=resid_max, vmin=-resid_max)
PyPlot.colorbar(); PyPlot.title("residual2")




function test_psf_transforms()

  mu_vec = Vector{Float64}[ Float64[1, 2], Float64[-1, -2], Float64[1, -1] ]
  sigma_vec = Array(Matrix{Float64}, 3)
  sigma_vec[1] = Float64[ 1 0.1; 0.1 1]
  sigma_vec[2] = Float64[ 1 0.3; 0.3 2]
  sigma_vec[3] = Float64[ 0.5 0.2; 0.2 0.5]
  weight_vec = Float64[0.4, 0.6, 0.1]

  par = wrap_parameters(mu_vec, sigma_vec, weight_vec)
  mu_vec_test, sigma_vec_test, weight_vec_test = unwrap_parameters(par)

  using Base.Test

  for k=1:3
    @test_approx_eq mu_vec_test[k] mu_vec[k]
    @test_approx_eq sigma_vec_test[k] sigma_vec[k]
    @test_approx_eq weight_vec_test[k] weight_vec[k]
  end

end



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
