using LinearAlgebra
using Base
using DataFrames
import Plots as plt
using Statistics
using CSV
using StatsBase
using Random
using SCS
using JuMP
using StatsPlots
using PGFPlots
using Distributions

# include("coorelation.jl")
# include("NEW CORREATION.jl")
include("correlation functions after edit-1.jl")

df = CSV.read("train_split.csv", DataFrame)

solver = optimizer_with_attributes(SCS.Optimizer, "max_iters" => 1e+6, "eps_abs" => 1e-8, "eps_rel" => 1e-8)
color_palette = cgrad(:matter, 10, categorical=true)

# Numeric data types
numerics = ["Int16", "Int32", "Int64", "Float16", "Float32", "Float64"]
num = [Symbol("member_id"), Symbol("loan_amnt"), Symbol("funded_amnt"), Symbol("funded_amnt_inv"), Symbol("int_rate"),
    Symbol("annual_inc"), Symbol("dti"), Symbol("delinq_2yrs"), Symbol("inq_last_6mths"), Symbol("open_acc"),
    Symbol("pub_rec"), Symbol("revol_bal"), Symbol("revol_util"), Symbol("total_acc"), Symbol("total_rec_int"),
    Symbol("total_rec_late_fee"), Symbol("recoveries"), Symbol("collection_recovery_fee"),
    Symbol("collections_12_mths_ex_med"), Symbol("acc_now_delinq"), Symbol("tot_coll_amt"),
    Symbol("tot_cur_bal"), Symbol("total_rev_hi_lim"), Symbol("loan_status")]
numeric_df = select(df, num)
numeric_df = coalesce.(numeric_df, 0)  # Replace missing values with 0
numeric_df = Matrix{Float64}(numeric_df)
correlation_matrix = cor(numeric_df)
#to check the psd of a matrix
function is_positive_semidefinite(A)
    eigenvals = eigen(A).values
    return all(eigenvals .>= 0)
end
# Get the remaining categorical columns
m = size(correlation_matrix, 1)

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
W1 = correlation_matrix
w1 = matrix_to_vector(W1)
W1[Omega] .= 0
eigen(W1)
w1 = matrix_to_vector(W1)
result1, etc1 = @timed correlation_projection_approx_comp(W1; iteration=1000_000, norm_tolerance=1e-8)
ProjC_0, ProjC_0_iter = result1
SQVC_0 = solveitcorrelationSQVC_completion_nodiag(w1, solver)
SQHC_0 = solveitcorrelationSQHC_completion_nodiag(w1, solver)
# SDHC_0 = solveitcorrelationSDHC_completion_nodiag(w1, solver)
# Plot the correlation matrix as a heatmap
# r = length(num)
Actual_C = plt.heatmap(correlation_matrix, title="Actual correlation matrix", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
ProjC_p0 = plt.heatmap(ProjC_0, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_p0 = plt.heatmap(SDHC_0, title="Completion by SDC", xticks=(1:r, r), yticks=(1:r, r), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_p0 = plt.heatmap(SQVC_0, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_p0 = plt.heatmap(SQHC_0, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()

f1 = plt.plot(Actual_C, ProjC_p0, SQVC_p0, SQHC_p0, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_0 = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])
plt.savefig(completion_0, "completion_0.png")

W2 = correlation_matrix
w2 = matrix_to_vector(W2)
W2[Omega] .= sum(w2) / length(w2)
eigen(W2)
w2 = matrix_to_vector(W2)
result2, etc2 = @timed correlation_projection_approx_comp(W2; iteration=1000_000, norm_tolerance=1e-8)
ProjC_MEAN, ProjC_MEAN_iter = result2
SQVC_MEAN = solveitcorrelationSQVC_completion_nodiag(w2, solver)
SQHC_MEAN = solveitcorrelationSQHC_completion_nodiag(w2, solver)
# SDHC_MEAN = solveitcorrelationSDHC_completion_nodiag(w2, solver)

ProjC_pMEAN = plt.heatmap(ProjC_MEAN, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pMEAN = plt.heatmap(SDHC_MEAN, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pMEAN = plt.heatmap(SQVC_MEAN, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pMEAN = plt.heatmap(SQHC_MEAN, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f2 = plt.plot!(Actual_C, ProjC_pMEAN, SQVC_pMEAN, SQHC_pMEAN, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_MEAN = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])

plt.savefig("completion_MEAN.png")

# # plt.savefig(f1, "Cor_Gau.tex")
W3 = correlation_matrix
w3 = matrix_to_vector(W3)
W3[Omega] .= rand(minimum(w3):maximum(w3))
eigen(W3)
w3 = matrix_to_vector(W3)
result3, etc3 = @timed correlation_projection_approx_comp(W3; iteration=1000_000, norm_tolerance=1e-8)
ProjC_Brand, ProjC_Brand_iter = result3
SQVC_Brand = solveitcorrelationSQVC_completion_nodiag(w3, solver)
SQHC_Brand = solveitcorrelationSQHC_completion_nodiag(w3, solver)
# SDHC_Brand = solveitcorrelationSDHC_completion_nodiag(w3, solver)

ProjC_pBrand = plt.heatmap(ProjC_Brand, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pBrand = plt.heatmap(SDHC_Brand, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pBrand = plt.heatmap(SQVC_Brand, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pBrand = plt.heatmap(SQHC_Brand, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f3 = plt.plot!(Actual_C, ProjC_pBrand, SQVC_pBrand, SQHC_pBrand, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_Brand = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])
plt.savefig("completion_Brand.png")

W4 = correlation_matrix
w4 = matrix_to_vector(W4)
W4[Omega] .= median(w4)
eigen(W4)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w4 = matrix_to_vector(W4)
result4, etc4 = @timed correlation_projection_approx_comp(W4; iteration=1000_000, norm_tolerance=1e-8)
ProjC_m, ProjC_m_iter = result4
SQVC_m = solveitcorrelationSQVC_completion_nodiag(w4, solver)
SQHC_m = solveitcorrelationSQHC_completion_nodiag(w4, solver)
# SDHC_m = solveitcorrelationSDHC_completion_nodiag(w4, solver)

ProjC_pm = plt.heatmap(ProjC_m, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pm = plt.heatmap(SDHC_m, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pm = plt.heatmap(SQVC_m, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pm = plt.heatmap(SQHC_m, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f4 = plt.plot!(Actual_C, ProjC_pm, SQVC_pm, SQHC_pm, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_m = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])

plt.savefig("completion_m.png")

W5 = correlation_matrix
w5 = matrix_to_vector(W5)
W5[Omega] .= StatsBase.mode(w5[:])
eigen(W5)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w5 = matrix_to_vector(W5)
result5, etc5 = @timed correlation_projection_approx_comp(W5; iteration=1000_000, norm_tolerance=1e-8)
ProjC_mode, ProjC_mode_iter = result5
SQVC_mode = solveitcorrelationSQVC_completion_nodiag(w5, solver)
SQHC_mode = solveitcorrelationSQHC_completion_nodiag(w5, solver)
# SDHC_mode = solveitcorrelationSDHC_completion_nodiag(w5, solver)


ProjC_pmode = plt.heatmap(ProjC_mode, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pmode = plt.heatmap(SDHC_mode, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pmode = plt.heatmap(SQVC_mode, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pmode = plt.heatmap(SQHC_mode, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f5 = plt.plot!(Actual_C, ProjC_pmode, SQVC_pmode, SQHC_pmode, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_mode = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])
plt.savefig("completion_mode.png")

W6 = correlation_matrix
w6 = matrix_to_vector(W6)
W6[Omega] .= maximum(w6)
eigen(W6)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w6 = matrix_to_vector(W6)
result6, etc6 = @timed correlation_projection_approx_comp(W6; iteration=1000_000, norm_tolerance=1e-8)
ProjC_max, ProjC_max_iter = result6
SQVC_max = solveitcorrelationSQVC_completion_nodiag(w6, solver)
SQHC_max = solveitcorrelationSQHC_completion_nodiag(w6, solver)
# SDHC_max = solveitcorrelationSDHC_completion_nodiag(w6, solver)

ProjC_pmax = plt.heatmap(ProjC_max, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pmax = plt.heatmap(SDHC_max, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pmax = plt.heatmap(SQVC_max, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pmax = plt.heatmap(SQHC_max, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f6 = plt.plot!(Actual_C, ProjC_pmax, SQVC_pmax, SQHC_pmax, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_max = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])
plt.savefig("completion_max.png")

W7 = correlation_matrix
w7 = matrix_to_vector(W7)
W7[Omega] .= minimum(w7)
eigen(W7)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w7 = matrix_to_vector(W7)
result7, etc7 = @timed correlation_projection_approx_comp(W7; iteration=1000_000, norm_tolerance=1e-8)
ProjC_min, ProjC_min_iter = result7
SQVC_min = solveitcorrelationSQVC_completion_nodiag(w7, solver)
SQHC_min = solveitcorrelationSQHC_completion_nodiag(w7, solver)
# SDHC_min = solveitcorrelationSDHC_completion_nodiag(w7, solver)

ProjC_pmin = plt.heatmap(ProjC_min, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pmin = plt.heatmap(SDHC_min, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pmin = plt.heatmap(SQVC_min, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pmin = plt.heatmap(SQHC_min, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f7 = plt.plot!(Actual_C, ProjC_pmode, SQVC_pmode, SQHC_pmode, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_min = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])

plt.savefig("completion_min.png")

W8 = correlation_matrix
w8 = matrix_to_vector(W8)
w8 = matrix_to_vector(W8)
W8[Omega] .= rand()
eigen(W8)# check is the matrix is PSD, if it is then we dont need to do any completion to find the PSD
w8 = matrix_to_vector(W8)
result8, etc8 = @timed correlation_projection_approx_comp(W8; iteration=1000_000, norm_tolerance=1e-8)
ProjC_r, ProjC_r_iter = result8
SQVC_r = solveitcorrelationSQVC_completion_nodiag(w8, solver)
SQHC_r = solveitcorrelationSQHC_completion_nodiag(w8, solver)
# SDHC_r = solveitcorrelationSDHC_completion_nodiag(w8, solver)

ProjC_pr = plt.heatmap(ProjC_r, title="Completion by Proj.", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# SDHC_pr = plt.heatmap(SDHC_r, title="Completion by SDC", xticks=(1:length(num), num), yticks=(1:length(num), num), xrotation=45, color=:vikO10, clims=(-1, 1))
SQVC_pr = plt.heatmap(SQVC_r, title="Completion by SQV", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
SQHC_pr = plt.heatmap(SQHC_r, title="Completion by SQC", xticks=false, yticks=false, xrotation=45, color=:vikO10, clims=(-1, 1))
# plt.pgfplotsx()
f8 = plt.plot!(Actual_C, ProjC_pr, SQVC_pr, SQHC_pr, titlefont=("Arial", 6, :bold), layout=(2, 2), title=["" "" "" "" ""])
completion_r = plt.plot!(title=["Actual correlation matrix " "Correlation matrix by Projection method" "Correlation matrix by SQV method" "Correlation matrix by SQC method"])
plt.savefig("completion_r.png")
# #error
norm(ProjC_0 - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_0 - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_0 - correlation_matrix) / norm(correlation_matrix)
# norm(SDHC_0 - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_MEAN - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_MEAN - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_MEAN - correlation_matrix) / norm(correlation_matrix)
# norm(SDHC_MEAN - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_Brand - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_Brand - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_Brand - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_Brand - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_m - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_m - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_m - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_m - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_mode - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_mode - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_mode - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_mode - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_max - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_max - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_max - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_max - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_min - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_min - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_min - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_min - correlation_matrix) / norm(correlation_matrix)

norm(ProjC_r - correlation_matrix) / norm(correlation_matrix)
norm(SQVC_r - correlation_matrix) / norm(correlation_matrix)
norm(SQHC_r - correlation_matrix) / norm(correlation_matrix)
norm(SDHC_r - correlation_matrix) / norm(correlation_matrix)
