
function symmetric_matrix(X, n)
    for j in 1:n
        for i in 1:n
            X[i, j] = X[j, i]
        end
    end
    return value.(X)
end

function matrix_to_vector(A::Array{T,2} where {T<:Real})

    n = size(A, 1)
    v = Vector{Float64}(undef, n * (n - 1) ÷ 2)  # Preallocate vector

    k = 1
    for j in 1:n
        for i in 1:j-1
            v[k] = A[i, j]
            k += 1
        end
    end

    return v
end


function vector_to_symmetric_matrix(v::Vector{<:Real})
    m = length(v)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))

    if m != n * (n - 1) / 2
        throw(ArgumentError("The length of the input vector is not compatible with creating a symmetric matrix."))
    end

    A = Matrix{Float64}(undef, n, n)
    k = 1
    for j in 2:n
        for i in 1:j-1
            A[i, j] = v[k]
            A[j, i] = v[k]
            k += 1
        end
    end

    return A
end
function display_upper_diagonal_positions(m)
    positions = []

    for j in (2:m)
        for i in 1:(j-1)
            push!(positions, (i, j))
        end
    end

    return positions
end
function positions_to_indices(positions, n)
    indices = Vector{Int}(undef, length(positions))
    for (i, (row, col)) in enumerate(positions)
        index = (col - 1) * n + row
        indices[i] = index
    end
    return indices
end
function project_to_PSD(f)
    n = size(f, 1)
    λ, U = eigen(f)
    λ = real(λ)
    eigenvecs = real.(U)
    λ[λ.<0] .= 0
    S = U * Diagonal(λ) * U'
    return S

end
function symmetric_position(omega, n)
    positions = []
    for i in omega
        row = ceil(Int, i / n)
        col = i - (row - 1) * n
        if row != col
            push!(positions, (row, col))
        end
    end
    return positions
end

function correlation_projection_approximation(A; iteration=1000_000, norm_tolerance=1e-8)
    n = size(A, 1)
    ONES = ones(n)
    P_K(X, k) = X - Diagonal(diag(X)) + Diagonal(ONES)
    P_S(X) = project_to_PSD(X)
    A_k = copy(A)
    for k in 1:iteration
        A_prev = A_k
        A_k = A_k + P_K(P_S(A_k), k) - P_S(A_k)
        A_k = A_k - Diagonal(diag(A_k)) + Diagonal(ONES)
        λ, U = eigen(A_k)
        λ = real(λ)
        A_k = whatever(A_k, U, λ, ONES)

        # Check for convergence based on norm difference
        norm_diff = norm(A_k - A_prev)
        if norm_diff ≤ norm_tolerance
            return A_k, k
        end
    end

    return A_k, iteration
end

function whatever(A_k, U, λ, ONES)
    U = real(U)
    λ[λ.<0] .= 0
    new_A = U * Diagonal(λ) * U'
    new_A = new_A - Diagonal(diag(new_A)) + Diagonal(ONES)
    return new_A
end

# ╔═╡ dd2b49f2-1e8e-4e12-8152-957cf83c956b
function correlation_projection_approx_comp(A; iteration=1000_000, norm_tolerance=1e-8)
    n = size(A, 1)
    ONES = ones(n)
    s1 = display_upper_diagonal_positions(n)
    u_size = size(s1, 1)
    s2 = collect(2:2:u_size) # select the even indecies in the strictly upper diagonal entries
    s3 = s1[s2]  # find the positions of these even indices in a matrix A
    s4 = positions_to_indices(s3, n) # find the indices in a matrix A that corresponding to the positions s3
    s5 = symmetric_position(s4, n)# find the symmetric positions of s4 indices
    s6 = positions_to_indices(s5, n)# find the corresponding indices of the symmetric positions s5
    Omega = vcat(s4, s6)

    # Omega = collect(2:2:n^2)

    P_K(X, k) = X - Diagonal(diag(X)) + Diagonal(ONES)
    P_S(X) = project_to_PSD(X)
    A_k = copy(A)
    for k in 1:iteration
        A_prev = A_k
        A_k = A_k + P_K(P_S(A_k), k) - P_S(A_k)
        A_k = A_k - Diagonal(diag(A_k)) + Diagonal(ONES)
        A_k[Omega] = A[Omega]
        λ, U = eigen(A_k)
        λ = real(λ)
        A_k = whatever(A_k, U, λ, ONES)

        # Check for convergence based on norm difference
        norm_diff = norm(A_k - A_prev)
        if norm_diff ≤ norm_tolerance
            return A_k, k
        end
    end
    return A_k, iteration

end

function solveitcorrelationSQVC_completion_nodiag(f::Vector{<:Real}, solver)
    # Finding the approximation correlation matrix by using the m by 1 vector f through SQV method, where m=n(n-1)/2(f does not include the diagonal)
    m, = size(f)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))
    model = Model(solver)

    # model = Model(SCS.Optimizer)

    # set_silent(model)
    @variable(model, t >= 0)
    @variable(model, v[1:m])
    @variable(model, w[1:m])
    @variable(model, X[1:n, 1:n], PSD)


    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    @constraint(model, [i in 1:m], w[i] == triu_elements[i])
    @constraint(model, [i in 2:2:m], w[i] == f[i])

    @constraint(model, [t; f .- w] in SecondOrderCone())

    @objective(model, Min, t)

    println(model)
    optimize!(model)
    return value.(X)
    println(solution_summary(model))


end

function solveitcorrelationSQHC_completion_nodiag(f::Vector{<:Real}, solver)
    # Finding the COMPLETION correlation matrix by using the m by 1 vector f through SQH method, where m=n(n-1)/2
    m, = size(f)

    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))

    # model = Model(SCS.Optimizer)

    model = Model(solver)
    # set_silent(model)
    @variable(model, t >= 0)
    @variable(model, v[1:m+1])
    @variable(model, X[1:n, 1:n], PSD)
    @variable(model, w[1:m+1])

    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    triu_elements1 = vcat(triu_elements, sqrt(n) / sqrt(2))
    f1 = vcat(f, sqrt(n) / sqrt(2))
    @constraint(model, [i in 1:m+1], w[i] == triu_elements1[i])
    # diagonal_indices = cumsum(1:n)
    # filtered_indices = filter(i -> !(i in diagonal_indices), 2:2:m)
    @constraint(model, [i in 2:2:m], w[i] == f1[i])
    @constraint(model, svecC, v == sqrt(2) .* (f1 - w))

    @constraint(model, [t; v] in SecondOrderCone())

    @objective(model, Min, t)

    optimize!(model)

    return value.(X)
    println(solution_summary(model))

end


function solveitcorrelationSDHC_completion_nodiag(f::Vector{<:Real}, solver)
    # Finding the approximation correlation matrix by using the m by 1 vector f through SDH method, where m=n(n-1)/2(f does not include the diagonal)
    m, = size(f)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))
    model = Model(solver)

    # model = Model(SCS.Optimizer)
    @variable(model, t >= 0)
    @variable(model, v[1:m+1])
    @variable(model, w[1:m+1])
    @variable(model, X[1:n, 1:n], PSD)

    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    triu_elements1 = vcat(triu_elements, sqrt(n) / sqrt(2))
    f1 = vcat(f, sqrt(n) / sqrt(2))
    @constraint(model, [i in 1:m+1], w[i] == triu_elements1[i])
    @constraint(model, [i in 2:2:m], w[i] == f1[i])

    # diagonal_indices = cumsum(1:n)
    @constraint(model, svecC, v == sqrt(2) .* (f1 - w))
    @constraint(model, [I(m + 1) v; v' t] in PSDCone())
    @objective(model, Min, t)
    optimize!(model)
    return value.(X)
    println(solution_summary(model))
end


function solveitcorrelationSDH_approx_nodiag(f::Vector{<:Real}, solver)
    # Finding the approximation correlation matrix by using the m by 1 vector f through SDH method, where m=n(n-1)/2(f does not include the diagonal)
    m, = size(f)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))
    model = Model(solver)

    # model = Model(SCS.Optimizer)
    @variable(model, t >= 0)
    @variable(model, v[1:m+1])
    @variable(model, X[1:n, 1:n], PSD)

    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    triu_elements1 = vcat(triu_elements, sqrt(n) / sqrt(2))
    f1 = vcat(f, sqrt(n) / sqrt(2))
    # diagonal_indices = cumsum(1:n)
    @constraint(model, svecC, v == sqrt(2) .* (f1 .- triu_elements1))
    @constraint(model, [I(m + 1) v; v' t] in PSDCone())
    @objective(model, Min, t)
    optimize!(model)
    return value.(X)
    println(solution_summary(model))
end

function solveitcorrelationSQH_approx_nodiag(f::Vector{<:Real}, solver)
    # Finding the approximation correlation matrix by using the m by 1 vector f through SQH method, where m=n(n-1)/2(f does not include the diagonal)
    m, = size(f)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))
    model = Model(solver)

    # model = Model(SCS.Optimizer)
    @variable(model, t >= 0)
    @variable(model, v[1:m+1])
    @variable(model, X[1:n, 1:n], PSD)

    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    triu_elements1 = vcat(triu_elements, sqrt(n) / sqrt(2))
    f1 = vcat(f, sqrt(n) / sqrt(2))
    @constraint(model, svecC, v == sqrt(2) .* (f1 .- triu_elements1))
    @constraint(model, [t; v] in SecondOrderCone())
    @objective(model, Min, t)
    optimize!(model)
    return value.(X)
    println(solution_summary(model))
end

function solveitcorrelationSQV_approx_nodiag(f::Vector{<:Real}, solver)
    # Finding the approximation correlation matrix by using the m by 1 vector f through SQV method, where m=n(n-1)/2(f does not include the diagonal)
    m, = size(f)
    n = Int(round((0.5 + sqrt(1 + 8 * m)) / 2))
    model = Model(solver)

    # model = Model(SCS.Optimizer)

    # set_silent(model)
    @variable(model, t >= 0)
    @variable(model, X[1:n, 1:n], PSD)
    @constraint(model, [i in 1:n], X[i, i] == 1)
    triu_elements = [X[i, j] for j in 2:n for i in 1:(j-1)]
    triu_elements1 = vcat(triu_elements, sqrt(n) / sqrt(2))
    f1 = vcat(f, sqrt(n) / sqrt(2))
    @constraint(model, [t; f1 .- triu_elements1] in SecondOrderCone())
    @objective(model, Min, t)
    println(model)
    optimize!(model)
    return value.(X)
    println(solution_summary(model))


end