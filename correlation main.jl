using JuMP
using SCS
using Random
using LinearAlgebra
using Statistics
using StatsBase
using DataFrames, XLSX
using Tables
using CSV
# include("coorelation.jl")
# include("NEW CORREATION.jl")
include("correlation functions after edit-1.jl")

solver = optimizer_with_attributes(SCS.Optimizer, "max_iters" => 1e+6, "eps_abs" => 1e-8, "eps_rel" => 1e-8)



#expermint approx-completion"

# n = 10 #(result is a matrix of size 5)
# n=36 #(result is a matrix of size 9)
n = 45 #(result is a matrix of size 10)
n = 105 #(result is a matrix of size 15)
n = 300 #(result is a matrix of size 25)
n = 1225 #(result is a matrix of size 50)
n = 4950 #(result is a matrix of size 100)
n = 19900 #(result is a matrix of size 200)

rng1 = MersenneTwister(56308)
f1 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f1[omega] .= 0
MS1 = vector_to_symmetric_matrix(f1)
result1, etc1 = @timed correlation_projection_approximation(MS1; iteration=1000_000, norm_tolerance=1e-8)
Proj_0, Proj_0_iter = result1
SQV_0 = solveitcorrelationSQV_approx_nodiag(f1, solver)
SQC_0 = solveitcorrelationSQH_approx_nodiag(f1, solver)
# SDC_0 = solveitcorrelationSDH_approx_nodiag(f1, solver)

rng1 = MersenneTwister(56308)
f2 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f2[omega] .= sum(f2) / length(f2)
MS2 = vector_to_symmetric_matrix(f2)
result2, etc2 = @timed correlation_projection_approximation(MS2; iteration=1000_000, norm_tolerance=1e-8)
Proj_MEAN, Proj_MEAN_iter = result2
SQV_MEAN = solveitcorrelationSQV_approx_nodiag(f2, solver)
SQC_MEAN = solveitcorrelationSQH_approx_nodiag(f2, solver)
# SDC_MEAN = solveitcorrelationSDH_approx_nodiag(f2, solver)

rng1 = MersenneTwister(56308)
f3 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f3[omega] .= rand(minimum(f3):maximum(f3))
MS3 = vector_to_symmetric_matrix(f3)
result3, etc3 = @timed correlation_projection_approximation(MS3; iteration=1000_000, norm_tolerance=1e-8)
Proj_Brand, Proj_Brand_iter = result3
SQV_Brand = solveitcorrelationSQV_approx_nodiag(f3, solver)
SQC_Brand = solveitcorrelationSQH_approx_nodiag(f3, solver)
# SDC_Brand = solveitcorrelationSDH_approx_nodiag(f3, solver)

rng1 = MersenneTwister(56308)
f4 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f4[omega] .= median(f4)
MS4 = vector_to_symmetric_matrix(f4)
result4, etc4 = @timed correlation_projection_approximation(MS4; iteration=1000_000, norm_tolerance=1e-8)
Proj_m, Proj_m_iter = result4
SQV_m = solveitcorrelationSQV_approx_nodiag(f4, solver)
SQC_m = solveitcorrelationSQH_approx_nodiag(f4, solver)
# SDC_m = solveitcorrelationSDH_approx_nodiag(f4, solver)

# # Create a DataFrame to store the results
# results_df = DataFrame(
#     Metric = ["Proj_m", "SQV_m","SQC_m", "SDC_m"],
#     Time = [etc4[1], retc4[1], retc44[1], retc444[1]],
#     Iterations = [Proj_m_iter, SQV_m_iter, SQC_m_iter, SDC_m_iter]
# )

# CSV.write("output.csv", results_df)

rng1 = MersenneTwister(56308)
f5 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f5[omega] .= StatsBase.mode(f5[:])
MS5 = vector_to_symmetric_matrix(f5)
result5, etc5 = @timed correlation_projection_approximation(MS5; iteration=1000_000, norm_tolerance=1e-8)
Proj_mode, Proj_mode_iter = result5
SQV_mode = solveitcorrelationSQV_approx_nodiag(f5, solver)
SQC_mode = solveitcorrelationSQH_approx_nodiag(f5, solver)
# SDC_mode = solveitcorrelationSDH_approx_nodiag(f5, solver)

rng1 = MersenneTwister(56308)
f6 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f6[omega] .= maximum(f6)
MS6 = vector_to_symmetric_matrix(f6)
result6, etc6 = @timed correlation_projection_approximation(MS6; iteration=1000_000, norm_tolerance=1e-8)
Proj_max, Proj_max_iter = result6
SQV_max = solveitcorrelationSQV_approx_nodiag(f6, solver)
SQC_max = solveitcorrelationSQH_approx_nodiag(f6, solver)
# SDC_max = solveitcorrelationSDH_approx_nodiag(f6, solver)

rng1 = MersenneTwister(56308)
f7 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f7[omega] .= minimum(f7)
MS7 = vector_to_symmetric_matrix(f7)
result7, etc7 = @timed correlation_projection_approximation(MS7; iteration=1000_000, norm_tolerance=1e-8)
Proj_min, Proj_min_iter = result7
SQV_min = solveitcorrelationSQV_approx_nodiag(f7, solver)
SQC_min = solveitcorrelationSQH_approx_nodiag(f7, solver)
# SDC_min = solveitcorrelationSDH_approx_nodiag(f7, solver)

rng1 = MersenneTwister(56308)
f8 = 2 * (rand(rng1, n) .- 0.5)
omega = collect(1:2:n)
f8[omega] .= rand()
MS8 = vector_to_symmetric_matrix(f8)
result8, etc8 = @timed correlation_projection_approximation(MS8; iteration=1000_000, norm_tolerance=1e-8)
Proj_r, Proj_r_iter = result8
SQV_r = solveitcorrelationSQV_approx_nodiag(f8, solver)
SQC_r = solveitcorrelationSQH_approx_nodiag(f8, solver)
# SDC_r = solveitcorrelationSDH_approx_nodiag(f8, solver)

#Calculate the error

norm(Proj_0 - MS1) / norm(MS1)
norm(SQV_0 - MS1) / norm(MS1)
norm(SQC_0 - MS1) / norm(MS1)
norm(SDC_0 - MS1) / norm(MS1)


norm(Proj_MEAN - MS2) / norm(MS2)
norm(SQV_MEAN - MS2) / norm(MS2)
norm(SQC_MEAN - MS2) / norm(MS2)
norm(SDC_MEAN - MS2) / norm(MS2)

norm(Proj_Brand - MS3) / norm(MS3)
norm(SQV_Brand - MS3) / norm(MS3)
norm(SQC_Brand - MS3) / norm(MS3)
norm(SDC_Brand - MS3) / norm(MS3)

norm(Proj_m - MS4) / norm(MS4)
norm(SQV_m - MS4) / norm(MS4)
norm(SQC_m - MS4) / norm(MS4)
norm(SDC_m - MS4) / norm(MS4)

norm(Proj_mode - MS5) / norm(MS5)
norm(SQV_mode - MS5) / norm(MS5)
norm(SQC_mode - MS5) / norm(MS5)
norm(SDC_mode - MS5) / norm(MS5)

norm(Proj_max - MS6) / norm(MS6)
norm(SQV_max - MS6) / norm(MS6)
norm(SQC_max - MS6) / norm(MS6)
norm(SDC_max - MS6) / norm(MS6)

norm(Proj_min - MS7) / norm(MS7)
norm(SQV_min - MS7) / norm(MS7)
norm(SQC_min - MS7) / norm(MS7)
norm(SDC_min - MS7) / norm(MS7)

norm(Proj_r - MS8) / norm(MS8)
norm(SQV_r - MS8) / norm(MS8)
norm(SQC_r - MS8) / norm(MS8)
norm(SDC_r - MS8) / norm(MS8)

#expermint completion"

n = 10 #(result is a matrix of size 5)
#it works till size n=6
n = 36 #(result is a matrix of size 9)
n = 45 #(result is a matrix of size 10)
n = 105 #(result is a matrix of size 15)
n = 300 #(result is a matrix of size 25)
n = 1225 #(result is a matrix of size 50)
n = 4950 #(result is a matrix of size 100)
n = 19900 #(result is a matrix of size 200)


rng1 = MersenneTwister(56308)
f1 = 2 * (rand(rng1, n) .- 0.5)
MS1 = vector_to_symmetric_matrix(f1)
W1 = cor(MS1)
m = size(W1, 1)
#################################
#the following steps to indicate the position of the missing entries to be a propriate with what we described in our function
s1 = display_upper_diagonal_positions(m)
u_size = size(s1, 1)
s2 = collect(1:2:u_size) # select the even indecies in the strictly upper diagonal entries
s3 = s1[s2]  # find the positions of these even indices in a matrix A
s4 = positions_to_indices(s3, m) # find the indices in a matrix A that corresponding to the positions s3
s5 = symmetric_position(s4, m)# find the symmetric positions of s4 indices
s6 = positions_to_indices(s5, m)# find the corresponding indices of the symmetric positions s5
Omega = vcat(s4, s6)
w1 = matrix_to_vector(W1)
######################################
W1[Omega] .= 0
eigen(W1)
w1 = matrix_to_vector(W1)
W1
result1, etc1 = @timed correlation_projection_approx_comp(W1; iteration=1000_000, norm_tolerance=1e-8)
ProjC_0, ProjC_0_iter = result1
SQVC_0 = solveitcorrelationSQVC_completion_nodiag(w1, solver)
SQHC_0 = solveitcorrelationSQHC_completion_nodiag(w1, solver)
# SDHC_0 = solveitcorrelationSDHC_completion_nodiag(w1, solver)



rng1 = MersenneTwister(56308)
f2 = 2 * (rand(rng1, n) .- 0.5)
MS2 = vector_to_symmetric_matrix(f2)
W2 = cor(MS2)
m = size(W2, 1)
w2 = matrix_to_vector(W2)
W2[Omega] .= sum(w2) / length(w2)
eigen(W2)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w2 = matrix_to_vector(W2)
result2, etc2 = @timed correlation_projection_approx_comp(W2; iteration=1000_000, norm_tolerance=1e-8)
ProjC_MEAN, ProjC_MEAN_iter = result2
SQVC_MEAN = solveitcorrelationSQVC_completion_nodiag(w2, solver)
SQHC_MEAN = solveitcorrelationSQHC_completion_nodiag(w2, solver)
# SDHC_MEAN = solveitcorrelationSDHC_completion_nodiag(w2, solver)

rng1 = MersenneTwister(56308)
f3 = 2 * (rand(rng1, n) .- 0.5)
MS3 = vector_to_symmetric_matrix(f3)
W3 = cor(MS3)
m = size(W3, 1)
w3 = matrix_to_vector(W3)
W3[Omega] .= rand(minimum(w3):maximum(w3))
eigen(W3)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w3 = matrix_to_vector(W3)
result3, etc3 = @timed correlation_projection_approx_comp(W3; iteration=1000_000, norm_tolerance=1e-8)
ProjC_Brand, ProjC_Brand_iter = result3
SQVC_Brand = solveitcorrelationSQVC_completion_nodiag(w3, solver)
SQHC_Brand = solveitcorrelationSQHC_completion_nodiag(w3, solver)
# SDHC_Brand = solveitcorrelationSDHC_completion_nodiag(w3, solver)

rng1 = MersenneTwister(56308)
f4 = 2 * (rand(rng1, n) .- 0.5)
MS4 = vector_to_symmetric_matrix(f4)
W4 = cor(MS4)
m = size(W4, 1)
w4 = matrix_to_vector(W4)
W4[Omega] .= median(w4)
eigen(W4)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w4 = matrix_to_vector(W4)
result4, etc4 = @timed correlation_projection_approx_comp(W4; iteration=1000_000, norm_tolerance=1e-8)
ProjC_m, ProjC_m_iter = result4
SQVC_m = solveitcorrelationSQVC_completion_nodiag(w4, solver)
SQHC_m = solveitcorrelationSQHC_completion_nodiag(w4, solver)
# SDHC_m = solveitcorrelationSDHC_completion_nodiag(w4, solver)

rng1 = MersenneTwister(56308)
f5 = 2 * (rand(rng1, n) .- 0.5)
MS5 = vector_to_symmetric_matrix(f5)
W5 = cor(MS5)
m = size(W5, 1)
w5 = matrix_to_vector(W5)
W5[Omega] .= StatsBase.mode(w5[:])
eigen(W5)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w5 = matrix_to_vector(W5)
result5, etc5 = @timed correlation_projection_approx_comp(W5; iteration=1000_000, norm_tolerance=1e-8)
ProjC_mode, ProjC_mode_iter = result5
SQVC_mode = solveitcorrelationSQVC_completion_nodiag(w5, solver)
SQHC_mode = solveitcorrelationSQHC_completion_nodiag(w5, solver)
# SDHC_mode = solveitcorrelationSDHC_completion_nodiag(w5, solver)

rng1 = MersenneTwister(56308)
f6 = 2 * (rand(rng1, n) .- 0.5)
MS6 = vector_to_symmetric_matrix(f6)
W6 = cor(MS6)
m = size(W6, 1)
w6 = matrix_to_vector(W6)
W6[Omega] .= maximum(w6)
eigen(W6)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w6 = matrix_to_vector(W6)
result6, etc6 = @timed correlation_projection_approx_comp(W6; iteration=1000_000, norm_tolerance=1e-8)
ProjC_max, ProjC_max_iter = result6
SQVC_max = solveitcorrelationSQVC_completion_nodiag(w6, solver)
SQHC_max = solveitcorrelationSQHC_completion_nodiag(w6, solver)
# SDHC_max = solveitcorrelationSDHC_completion_nodiag(w6, solver)

rng1 = MersenneTwister(56308)
f7 = 2 * (rand(rng1, n) .- 0.5)
MS7 = vector_to_symmetric_matrix(f7)
W7 = cor(MS7)
m = size(W7, 1)
w7 = matrix_to_vector(W7)
W7[Omega] .= minimum(w7)
eigen(W7)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w7 = matrix_to_vector(W7)
result7, etc7 = @timed correlation_projection_approx_comp(W7; iteration=1000_000, norm_tolerance=1e-8)
ProjC_min, ProjC_min_iter = result7
SQVC_min = solveitcorrelationSQVC_completion_nodiag(w7, solver)
SQHC_min = solveitcorrelationSQHC_completion_nodiag(w7, solver)
# SDHC_min = solveitcorrelationSDHC_completion_nodiag(w7, solver)

rng1 = MersenneTwister(56308)
f8 = 2 * (rand(rng1, n) .- 0.5)
MS8 = vector_to_symmetric_matrix(f8)
W8 = cor(MS8)
m = size(W8, 1)
w8 = matrix_to_vector(W8)
W8[Omega] .= rand()
eigen(W8)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w8 = matrix_to_vector(W8)
result8, etc8 = @timed correlation_projection_approx_comp(W8; iteration=1000_000, norm_tolerance=1e-8)
ProjC_r, ProjC_r_iter = result8
SQVC_r = solveitcorrelationSQVC_completion_nodiag(w8, solver)
SQHC_r = solveitcorrelationSQHC_completion_nodiag(w8, solver)
# SDHC_r = solveitcorrelationSDHC_completion_nodiag(w8, solver)


#Calculate the error 
norm(ProjC_0 - W1) / norm(W1)
norm(SQVC_0 - W1) / norm(W1)
norm(SQHC_0 - W1) / norm(W1)
norm(SDHC_0 - W1) / norm(W1)

norm(ProjC_MEAN - W2) / norm(W2)
norm(SQVC_MEAN - W2) / norm(W2)
norm(SQHC_MEAN - W2) / norm(W2)
norm(SDHC_MEAN - W2) / norm(W2)

norm(ProjC_Brand - W3) / norm(W3)
norm(SQVC_Brand - W3) / norm(W3)
norm(SQHC_Brand - W3) / norm(W3)
norm(SDHC_Brand - W3) / norm(W3)

norm(ProjC_m - W4) / norm(W4)
norm(SQVC_m - W4) / norm(W4)
norm(SQHC_m - W4) / norm(W4)
norm(SDHC_m - W4) / norm(W4)

norm(ProjC_mode - W5) / norm(W5)
norm(SQVC_mode - W5) / norm(W5)
norm(SQHC_mode - W5) / norm(W5)
norm(SDHC_mode - W5) / norm(W5)

norm(ProjC_max - W6) / norm(W6)
norm(SQVC_max - W6) / norm(W6)
norm(SQHC_max - W6) / norm(W6)
norm(SDHC_max - W6) / norm(W6)

norm(ProjC_min - W7) / norm(W7)
norm(SQVC_min - W7) / norm(W7)
norm(SQHC_min - W7) / norm(W7)
norm(SDHC_min - W7) / norm(W7)

norm(ProjC_r - W8) / norm(W8)
norm(SQVC_r - W8) / norm(W8)
norm(SQHC_r - W8) / norm(W8)
norm(SDHC_r - W8) / norm(W8)


