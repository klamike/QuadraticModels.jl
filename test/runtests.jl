# stdlib
using LinearAlgebra, Printf, SparseArrays, Test

# our packages
using ADNLPModels,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  NLPModelsTest,
  QPSReader,
  QuadraticModels,
  SparseConnectivityTracer,
  SparseMatricesCOO

@testset "test utils" begin
  A = rand(10, 10)
  @test nnz(A) == 100
  @test nnz(Diagonal(A)) == 10
  @test nnz(Symmetric(A)) == 55
  v1, v2 = rand(10), rand(9)
  @test nnz(SymTridiagonal(v1, v2)) == 19
  Asp = sparse([1.0 0.0 0.0; 1.0 0.0 0.0; 0.0 3.0 2.0])
  @test nnz(Symmetric(Asp, :L)) == 4
end

# Definition of quadratic problems
qp_problems_Matrix = ["bndqp", "eqconqp"]
qp_problems_COO = ["uncqp", "ineqconqp"]
for qp in [qp_problems_Matrix; qp_problems_COO]
  include(joinpath("problems", "$qp.jl"))
end

include("test_consistency.jl")

function testSM(sm) # test function for a specific problem
  @test (sm.meta.ncon == 2 && sm.meta.nvar == 5)
  @test sm.meta.lcon == sm.meta.ucon
  lvar_sm_true = [0.0; 0.0; 0.0; -4.0; -3.0]
  uvar_sm_true = [Inf; Inf; Inf; Inf; -2.0]
  H_sm_true = sparse(
    [
      6.0 0.0 0.0 0.0 0.0
      2.0 5.0 0.0 0.0 0.0
      1.0 2.0 4.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0
      0.0 0.0 0.0 0.0 0.0
    ],
  )
  A_sm_true = sparse([
    1.0 0.0 1.0 0.0 -1.0
    0.0 2.0 1.0 -1.0 0.0
  ])

  @test all(lvar_sm_true .== sm.meta.lvar)
  @test all(uvar_sm_true .== sm.meta.uvar)
  @test all(sparse(sm.data.A.rows, sm.data.A.cols, sm.data.A.vals, 2, 5) .≈ A_sm_true)
  @test all(sparse(sm.data.H.rows, sm.data.H.cols, sm.data.H.vals, 5, 5) .≈ H_sm_true)
end

@testset "SlackModel" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 2.0 1.0
  ]
  b = [0.0; 3]
  l = [0.0; 0; 0]
  u = [Inf; Inf; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(tril(H)),
    A = SparseMatrixCOO(A),
    lcon = [-3.0; -4.0],
    ucon = [-2.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )
  sm = SlackModel(qp)
  testSM(sm)

  SlackModel!(qp)
  testSM(qp)
end

@testset "dense QP, convert" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 2.0 1.0
  ]
  b = [0.0; 3]
  l = [0.0; 0; 0]
  u = [Inf; Inf; Inf]
  T = eltype(c)

  qpdense = QuadraticModel(
    c,
    tril(H),
    A = A,
    lcon = [-3.0; -4.0],
    ucon = [-2.0; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
  )

  smdense = SlackModel(qpdense)
  testSM(smdense)

  qpcoo =
    convert(QuadraticModel{T, Vector{T}, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}}, qpdense)
  @test typeof(qpcoo.data.H) <: SparseMatrixCOO
  @test typeof(qpcoo.data.A) <: SparseMatrixCOO
end

@testset "sort cols COO" begin
  Hrows = [1; 2; 1; 3]
  Hcols = [2; 1; 4; 4]
  Hvals = [1.0; 2.0; 3.0; 4.0]
  Arows = [2; 2; 1; 4]
  Acols = [4; 3; 1; 2]
  Avals = [-1.0; -2.0; -3.0; -4.0]
  c = [-8.0; -3; -3; 2.0]
  l = [0.0; 0; 0; 0]
  u = [Inf; Inf; Inf; Inf]
  qp = QuadraticModel(
    c,
    Hrows,
    Hcols,
    Hvals,
    Arows = Arows,
    Acols = Acols,
    Avals = Avals,
    lcon = [-3.0; -4.0; 2.0; 1.0],
    ucon = [-2.0; Inf; Inf; Inf],
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM1",
    sortcols = true,
  )
  @test issorted(Hcols)
  @test issorted(Acols)
end

@testset "LinearOperators" begin
  nvar, ncon = 10, 7
  T = Float64
  H = Symmetric(tril(sprand(T, nvar, nvar, 0.3)), :L)
  A = sprand(T, ncon, nvar, 0.4)
  c = rand(nvar)
  lvar = fill(-Inf, nvar)
  uvar = fill(0.0, nvar)
  lcon = rand(ncon)
  ucon = lcon .+ 100.0
  qp = QuadraticModel(
    c,
    SparseMatrixCOO(H.data),
    A = SparseMatrixCOO(A),
    lcon = lcon,
    ucon = ucon,
    lvar = lvar,
    uvar = uvar,
    c0 = 0.0,
    name = "QM",
  )
  qpLO = QuadraticModel(
    c,
    LinearOperator(H),
    A = LinearOperator(A),
    lcon = lcon,
    ucon = ucon,
    lvar = lvar,
    uvar = uvar,
    c0 = 0.0,
    name = "QMLO",
  )
  x = ones(10)
  @test obj(qp, x) ≈ obj(qpLO, x)
  @test grad(qp, x) ≈ grad(qpLO, x)
  @test cons(qp, x) ≈ cons(qpLO, x)

  SM = SlackModel(qp)
  SMLO = SlackModel(qpLO)
  nfix = length(qp.meta.jfix)
  ns = qp.meta.ncon - nfix
  x = rand(SM.meta.nvar)
  y = rand(SM.meta.ncon)
  @test SM.meta.nvar == qp.meta.nvar + ns
  @test obj(SM, x) ≈ obj(SMLO, x)
  @test grad(SM, x) ≈ grad(SMLO, x)
  @test cons(SM, x) ≈ cons(SMLO, x)
  @test hprod(SMLO, x, x) ≈ hprod(SMLO, x, x)
  @test jtprod(SMLO, x, y) ≈ jtprod(SM, x, y)
  @test objgrad(SMLO, x)[1] ≈ objgrad(SM, x)[1]
  @test objgrad(SMLO, x)[2] ≈ objgrad(SM, x)[2]

  # tests default A value
  qp = QuadraticModel(c, SparseMatrixCOO(H.data), lvar = lvar, uvar = uvar, c0 = 0.0, name = "QM")
  qpLO = QuadraticModel(c, LinearOperator(H), lvar = lvar, uvar = uvar, c0 = 0.0, name = "QMLO")
  @test qp.data.A isa SparseMatrixCOO
  @test qpLO.data.A isa AbstractLinearOperator
end

@testset "struct and coord CSC" begin
  H = [
    6.0 2.0 1.0
    2.0 5.0 2.0
    1.0 2.0 4.0
  ]
  c = [-8.0; -3; -3]
  A = [
    1.0 0.0 1.0
    0.0 2.0 1.0
  ]
  b = [0.0; 3]
  l = [0.0; 0; 0]
  u = [Inf; Inf; Inf]
  T = eltype(c)
  qp = QuadraticModel(
    c,
    tril(sparse(H)),
    A = sparse(A),
    lcon = b,
    ucon = b,
    lvar = l,
    uvar = u,
    c0 = 0.0,
    name = "QM",
  )

  x = zeros(3)
  rowsH, colsH = hess_structure(qp)
  valsH = hess_coord(qp, x)
  rowsHtrue, colsHtrue, valsHtrue = findnz(tril(sparse(H)))
  @test rowsH == rowsHtrue
  @test colsH == colsHtrue
  @test valsH == valsHtrue
  rowsA, colsA = jac_structure(qp)
  valsA = jac_coord(qp, x)
  rowsAtrue, colsAtrue, valsAtrue = findnz(sparse(A))
  @test rowsA == rowsAtrue
  @test colsA == colsAtrue
  @test valsA == valsAtrue
end

include("test_presolve.jl")
include("test_allocations.jl")

@testset "ParametricQuadraticModels" begin
    @testset "Basic construction" begin
        # Test basic construction
        n = 3
        m = 2
        p = 2
        pcon = 1
        
        c = [1.0, 2.0, 3.0]
        F = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        H = [2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        A = [1.0 1.0 0.0; 0.0 1.0 1.0]
        B = [1.0 0.0; 0.0 1.0]
        P = [1.0 0.0]
        lcon = [0.0, 0.0]
        ucon = [5.0, 5.0]
        lparam = [-1.0]
        uparam = [1.0]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, P=P, lcon=lcon, ucon=ucon, lparam=lparam, uparam=uparam)
        
        @test pqp isa ParametricQuadraticModel
        @test pqp.meta.nvar == n
        @test pqp.meta.ncon == m
        @test size(pqp.data.F) == (n, p)
        @test size(pqp.data.B) == (m, p)
        @test size(pqp.data.P) == (pcon, p)
    end
    
    @testset "Parameter evaluation" begin
        n = 2
        m = 1
        p = 2
        
        c = [1.0, 2.0]
        F = [1.0 0.0; 0.0 0.5]
        H = [2.0 0.0; 0.0 1.0]
        A = [1.0 1.0]
        B = [1.0 0.0]
        lcon = [0.0]
        ucon = [5.0]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, lcon=lcon, ucon=ucon)
        
        θ = [1.0, 2.0]
        qp = evaluate_at_parameter(pqp, θ)
        
        @test qp isa QuadraticModel
        @test qp.meta.nvar == n
        @test qp.meta.ncon == m
        
        # Check that the linear term is c + F*θ
        c_expected = c + F * θ
        @test qp.data.c ≈ c_expected
        
        # Check that constraint bounds are lcon - B*θ, ucon - B*θ
        Bθ = B * θ
        @test qp.meta.lcon ≈ lcon - Bθ
        @test qp.meta.ucon ≈ ucon - Bθ
    end
    
    @testset "NLPModels interface" begin
        n = 2
        m = 1
        p = 2
        
        c = [1.0, 2.0]
        F = [1.0 0.0; 0.0 0.5]
        H = [2.0 0.0; 0.0 1.0]
        A = [1.0 1.0]
        B = [1.0 0.0]
        lcon = [0.0]
        ucon = [5.0]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, lcon=lcon, ucon=ucon)
        x = [1.0, 2.0]
        
        # Test objective function
        obj_val = obj(pqp, x)
        expected_obj = 0.5 * dot(x, H * x) + dot(c, x)
        @test obj_val ≈ expected_obj
        
        # Test gradient
        g = grad(pqp, x)
        expected_g = c + H * x
        @test g ≈ expected_g
        
        # Test constraints
        cons_val = cons(pqp, x)
        expected_cons = A * x
        @test cons_val ≈ expected_cons
    end
    
    @testset "Sparse matrices" begin
        n = 3
        m = 2
        p = 2
        
        c = [1.0, 2.0, 3.0]
        F = sparse([1.0 0.0; 0.0 0.5; 0.0 0.0])
        H = sparse([2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
        A = sparse([1.0 1.0 0.0; 0.0 1.0 1.0])
        B = sparse([1.0 0.0; 0.0 1.0])
        lcon = [-Inf, -Inf]
        ucon = [Inf, Inf]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, lcon=lcon, ucon=ucon)
        
        @test pqp isa ParametricQuadraticModel
        @test issparse(pqp.data.F)
        @test issparse(pqp.data.H)
        @test issparse(pqp.data.A)
        @test issparse(pqp.data.B)
    end
    
    @testset "Parameter sensitivity functions" begin
        n = 2
        m = 1
        p = 2
        
        c = [1.0, 2.0]
        F = sparse([1.0 0.0; 0.0 0.5])
        H = sparse([2.0 0.0; 0.0 1.0])
        A = sparse([1.0 1.0])
        B = sparse([1.0 0.0])
        lcon = [0.0]
        ucon = [5.0]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, lcon=lcon, ucon=ucon)
        x = [1.0, 2.0]
        θ = [1.0, 2.0]
        y = [1.0]
        
        # Test jac_param
        B_result = jac_param(pqp, x, θ)
        @test B_result == B
        
        # Test jac_param!
        B_copy = similar(B)
        jac_param!(pqp, x, θ, B_copy)
        @test B_copy == B
        
        # Test hess_param
        F_result = hess_param(pqp, x, θ, y)
        @test F_result == F
        
        # Test hess_param!
        F_copy = similar(F)
        hess_param!(pqp, x, θ, y, F_copy)
        @test F_copy == F
        
        # Test structure functions
        rows = Vector{Int}(undef, nnz(B))
        cols = Vector{Int}(undef, nnz(B))
        jac_param_structure!(pqp, rows, cols)
        @test length(rows) == nnz(B)
        @test length(cols) == nnz(B)
        
        rows = Vector{Int}(undef, nnz(F))
        cols = Vector{Int}(undef, nnz(F))
        hess_param_structure!(pqp, rows, cols)
        @test length(rows) == nnz(F)
        @test length(cols) == nnz(F)
        
        # Test coordinate functions
        vals_jac = zeros(nnz(B))
        jac_param_coord!(pqp, x, θ, vals_jac)
        @test vals_jac == B.nzval
        
        vals_hess_param = zeros(nnz(F))
        hess_param_coord!(pqp, x, θ, y, vals_hess_param)
        @test vals_hess_param == F.nzval
    end
    
    @testset "Parameter feasibility check" begin
        n = 2
        m = 1
        p = 2
        pcon = 1
        
        c = [1.0, 2.0]
        F = [1.0 0.0; 0.0 0.5]
        H = [2.0 0.0; 0.0 1.0]
        A = [1.0 1.0]
        B = [1.0 0.0]
        P_matrix = [1.0 0.0]  # 1x2 matrix: 1 parameter constraint, 2 parameters
        lcon = [0.0]
        ucon = [5.0]
        lparam_bounds = [-1.0]
        uparam_bounds = [1.0]
        
        pqp = ParametricQuadraticModel(c, F, H; A=A, B=B, P=P_matrix, lcon=lcon, ucon=ucon, lparam=lparam_bounds, uparam=uparam_bounds)
        
        # Test feasible parameter
        θ_feasible = [0.5, 0.0]  # P*θ = [0.5] which is in [-1.0, 1.0]
        qp_feasible = evaluate_at_parameter(pqp, θ_feasible)
        @test qp_feasible isa QuadraticModel
        
        # Test infeasible parameter (violates upper bound)
        θ_infeasible_upper = [2.0, 0.0]  # P*θ = [2.0] which is > 1.0
        @test_throws ArgumentError evaluate_at_parameter(pqp, θ_infeasible_upper)
        
        # Test infeasible parameter (violates lower bound)
        θ_infeasible_lower = [-2.0, 0.0]  # P*θ = [-2.0] which is < -1.0
        @test_throws ArgumentError evaluate_at_parameter(pqp, θ_infeasible_lower)
        
        # Test with check_feasibility=false (should not throw)
        qp_no_check = evaluate_at_parameter(pqp, θ_infeasible_upper; check_feasibility=false)
        @test qp_no_check isa QuadraticModel
        
        # Test with no parameter constraints (should not throw)
        pqp_no_param_con = ParametricQuadraticModel(c, F, H; A=A, B=B, lcon=lcon, ucon=ucon)
        qp_no_param_con = evaluate_at_parameter(pqp_no_param_con, θ_infeasible_upper)
        @test qp_no_param_con isa QuadraticModel
    end
end
