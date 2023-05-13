# Example for reconstruction using ADMM solving an TV regularization for X-ray attenuation CT
add_example(admml2_tv_tomo admml2_tv_tomo.cpp)

# Example for reconstruction using ADMM solving an TV regularization for X-ray attenuation CT
add_example(admml2_tv_tomo_alloc_bench admml2_tv_tomo_alloc_bench.cpp)

# Example for reconstruction using CGLS solving the least squares problem for limited angle X-ray attenuation CT
add_example(cgls_ls_limitedtomo cgls_leastsquares_limitedtomo.cpp)

# Example for reconstruction using CGLS solving the least squares problem for X-ray attenuation CT
add_example(cgls_ls_tomo cgls_leastsquares_tomo.cpp)

# Example for reconstruction using CGLS solving the Tikhonov problem (or L2-Regularization) problem for X-ray
# attenuation CT
add_example(cgls_tikhonov_tomo cgls_tikhonov_tomo.cpp)

# Example for reconstruction using Landweber solving the least squares problem for X-ray attenuation CT
add_example(landweber_ls_tomo landweber_ls_tomo.cpp)

# Example for reconstruction using PDG solving the least squares problem with a non-negativity constraint for X-ray
# attenuation CT
add_example(pgd_ls-nonneg_tomo pgd_ls-nonneg_tomo.cpp)

# Example for reconstruction using APDG solving the least squares problem with a non-negativity constraint for X-ray
# attenuation CT
add_example(apgd_ls-nonneg_tomo apgd_ls-nonneg_tomo.cpp)

# Example for reconstruction using OGM solving the least squares problem with a Huber-TV regularization for X-ray
# attenuation CT
add_example(fgm_tikhonov_tomo fgm_tikhonov_tomo.cpp)

# Example for reconstruction using OGM solving the least squares problem with a Huber-TV regularization for X-ray
# attenuation CT
add_example(fgm_gentikhonov_tomo fgm_gentikhonov_tomo.cpp)

# Example for reconstruction using OGM solving the least squares problem with a Huber-TV regularization for X-ray
# attenuation CT
add_example(ogm_tikhonov_tomo ogm_tikhonov_tomo.cpp)
