@testset "ParametricQuadraticModel" begin
  T = Float64
  nvar = 3
  ncon = 2
  nparam = 2

  # H structure: one nonzero at (2,2) — lower triangle COO
  Hrows = [2]
  Hcols = [2]
  # A structure: sparse([1 2 0; 0 1 3]) in COO
  Arows = [1, 1, 2, 2]
  Acols = [1, 2, 2, 3]

  # data_fn: H_nzvals = [θ₁²], A_nzvals = [1+θ₁, 2, θ₂, 3],
  #          c = [1+θ₁, -2+θ₂, 3], c0 = 0.5 + θ₁θ₂
  #          lcon = [-5+θ₁, -5], ucon = [10, 10+θ₂]
  #          lvar = [-10+θ₁, -10, -10], uvar = [10, 10, 10+θ₂]
  data_fn(θ) = (
    [θ[1]^2],
    [1 + θ[1], 2.0, θ[2], 3.0],
    [1 + θ[1], -2 + θ[2], 3.0],
    0.5 + θ[1] * θ[2],
    [-5.0 + θ[1], -5.0], [10.0, 10.0 + θ[2]],
    [-10.0 + θ[1], -10.0, -10.0], [10.0, 10.0, 10.0 + θ[2]],
  )

  # dH: 1 nonzero at (1,1) → ∂H_nzvals[1]/∂θ₁ = 2θ₁
  # dA: 2 nonzeros: (1,1) → ∂A[1]/∂θ₁=1, (3,2) → ∂A[3]/∂θ₂=1
  # dc: 2 nonzeros: (1,1) → ∂c[1]/∂θ₁=1, (2,2) → ∂c[2]/∂θ₂=1
  # dlcon: 1 nonzero: (1,1) → ∂lcon[1]/∂θ₁=1
  # ducon: 1 nonzero: (2,2) → ∂ucon[2]/∂θ₂=1
  # dlvar: 1 nonzero: (1,1) → ∂lvar[1]/∂θ₁=1
  # duvar: 1 nonzero: (3,2) → ∂uvar[3]/∂θ₂=1
  data_jac_fn(θ) = (
    [2θ[1]],                          # dH_nzvals (1 value)
    [1.0, 1.0],                        # dA_nzvals (2 values)
    [1.0, 1.0],                        # dc_nzvals (2 values)
    [θ[2], θ[1]],                      # dc0 (2-vector)
    [1.0],                             # dlcon_nzvals
    [1.0],                             # ducon_nzvals
    [1.0],                             # dlvar_nzvals
    [1.0],                             # duvar_nzvals
  )

  # SparseMatrixCOO structures for Jacobians
  dH_structure = SparseMatrixCOO(1, 2, [1], [1], [0.0])
  dA_structure = SparseMatrixCOO(4, 2, [1, 3], [1, 2], [0.0, 0.0])
  dc_structure = SparseMatrixCOO(3, 2, [1, 2], [1, 2], [0.0, 0.0])
  dlcon_structure = SparseMatrixCOO(2, 2, [1], [1], [0.0])
  ducon_structure = SparseMatrixCOO(2, 2, [2], [2], [0.0])
  dlvar_structure = SparseMatrixCOO(3, 2, [1], [1], [0.0])
  duvar_structure = SparseMatrixCOO(3, 2, [3], [2], [0.0])

  θ = [1.0, 2.0]

  pqp = QuadraticModels.ParametricQuadraticModel(
    Hrows, Hcols, Arows, Acols, nvar, ncon,
    data_fn, data_jac_fn, θ;
    dH_structure = dH_structure,
    dA_structure = dA_structure,
    dc_structure = dc_structure,
    dlcon_structure = dlcon_structure,
    ducon_structure = ducon_structure,
    dlvar_structure = dlvar_structure,
    duvar_structure = duvar_structure,
  )

  @test pqp.meta.nvar == nvar
  @test pqp.meta.ncon == ncon
  @test NLPModels.get_nparam(pqp) == nparam
  @test pqp.meta.nnzh == 1

  # Build equivalent QuadraticModel for same θ
  H_nz, A_nz, c_v, c0_v, lc, uc, lv, uv = data_fn(θ)
  qp_ref = QuadraticModel(
    c_v, copy(Hrows), copy(Hcols), H_nz;
    Arows = copy(Arows), Acols = copy(Acols), Avals = A_nz,
    lcon = lc, ucon = uc, lvar = lv, uvar = uv, c0 = c0_v,
  )

  x = [1.0, 2.0, 3.0]
  y = [1.0, -1.0]

  # Standard QP operations match reference
  @test obj(pqp, x) ≈ obj(qp_ref, x)
  @test grad(pqp, x) ≈ grad(qp_ref, x)
  @test cons(pqp, x) ≈ cons(qp_ref, x)

  v_x = [0.5, -1.0, 2.0]
  @test hprod(pqp, x, v_x) ≈ hprod(qp_ref, x, v_x)

  jrows, jcols = jac_structure(pqp)
  jvals = jac_coord(pqp, x)
  jrows2, jcols2 = jac_structure(qp_ref)
  jvals2 = jac_coord(qp_ref, x)
  @test jrows == jrows2
  @test jcols == jcols2
  @test jvals ≈ jvals2

  hrows, hcols = hess_structure(pqp)
  hvals = hess_coord(pqp, x)
  hrows2, hcols2 = hess_structure(qp_ref)
  hvals2 = hess_coord(qp_ref, x)
  @test hrows == hrows2
  @test hcols == hcols2
  @test hvals ≈ hvals2

  # ── Parametric API ──

  @test NLPModels.get_param_values(pqp) ≈ θ

  # grad_param: ∇_θ f = dc0 + dc'x + ½ dH'u_h
  # f = 0.5+θ₁θ₂ + (1+θ₁)x₁+(-2+θ₂)x₂+3x₃ + ½θ₁²x₂²
  # ∂f/∂θ₁ = θ₂+x₁+θ₁x₂² = 2+1+1*4 = 7
  # ∂f/∂θ₂ = θ₁+x₂ = 1+2 = 3
  gp = NLPModels.grad_param(pqp, x)
  @test gp ≈ [7.0, 3.0]

  # jac_param: ∂cons/∂θ
  # cons₁ = (1+θ₁)x₁+2x₂ → ∂/∂θ₁=x₁=1, ∂/∂θ₂=0
  # cons₂ = θ₂x₂+3x₃ → ∂/∂θ₁=0, ∂/∂θ₂=x₂=2
  jp_rows, jp_cols = NLPModels.jac_param_structure(pqp)
  jp_vals = NLPModels.jac_param_coord(pqp, x)
  # column-major: [J[1,1], J[2,1], J[1,2], J[2,2]] = [1, 0, 0, 2]
  @test jp_vals ≈ [1.0, 0.0, 0.0, 2.0]

  # jpprod: J * v_param
  v_param = [0.5, -0.5]
  Jv = NLPModels.jpprod(pqp, x, v_param)
  @test Jv ≈ [1.0 0.0; 0.0 2.0] * v_param

  # jptprod: J' * y
  Jtv = NLPModels.jptprod(pqp, x, y)
  @test Jtv ≈ [1.0 0.0; 0.0 2.0]' * y

  # hess_param: ∇²_{x,θ} L with obj_weight=2
  # L = 2f + y₁cons₁ + y₂cons₂
  # ∂²L/∂x₁∂θ₁ = 2+y₁ = 3, ∂²L/∂x₁∂θ₂ = 0
  # ∂²L/∂x₂∂θ₁ = 2·2θ₁x₂ = 8, ∂²L/∂x₂∂θ₂ = 2+y₂ = 1
  # ∂²L/∂x₃∂θ₁ = 0, ∂²L/∂x₃∂θ₂ = 0
  hp_rows, hp_cols = NLPModels.hess_param_structure(pqp)
  hp_vals = NLPModels.hess_param_coord(pqp, x, y; obj_weight = 2.0)
  # column-major: [col1; col2] = [3, 8, 0, 0, 1, 0]
  @test hp_vals ≈ [3.0, 8.0, 0.0, 0.0, 1.0, 0.0]

  # hpprod: (∇²_{x,θ} L) v with obj_weight=2
  Hv = NLPModels.hpprod(pqp, x, y, v_param; obj_weight = 2.0)
  @test Hv ≈ [3.0 0.0; 8.0 1.0; 0.0 0.0] * v_param

  # hptprod: (∇²_{x,θ} L)' v with obj_weight=2
  Htv = NLPModels.hptprod(pqp, x, y, v_x; obj_weight = 2.0)
  @test Htv ≈ [3.0 8.0 0.0; 0.0 1.0 0.0] * v_x

  # ── Bounds parametric API ──

  @test pqp.meta.nnzjplcon == 1
  @test pqp.meta.nnzjpucon == 1
  @test pqp.meta.nnzjplvar == 1
  @test pqp.meta.nnzjpuvar == 1

  # lcon_jpprod: dlcon * v
  # dlcon = [1 0; 0 0], v = [0.5, -0.5] → [0.5, 0.0]
  lcon_Jv = NLPModels.lcon_jpprod(pqp, v_param)
  @test lcon_Jv ≈ [0.5, 0.0]

  # lcon_jptprod: dlcon' * y
  lcon_Jtv = NLPModels.lcon_jptprod(pqp, y)
  @test lcon_Jtv ≈ [1.0, 0.0]

  # ucon_jpprod: ducon * v
  # ducon = [0 0; 0 1], v = [0.5, -0.5] → [0.0, -0.5]
  ucon_Jv = NLPModels.ucon_jpprod(pqp, v_param)
  @test ucon_Jv ≈ [0.0, -0.5]

  # lvar_jpprod: dlvar * v
  # dlvar = [1 0; 0 0; 0 0], v = [0.5, -0.5] → [0.5, 0.0, 0.0]
  lvar_Jv = NLPModels.lvar_jpprod(pqp, v_param)
  @test lvar_Jv ≈ [0.5, 0.0, 0.0]

  # uvar_jpprod: duvar * v
  # duvar = [0 0; 0 0; 0 1], v = [0.5, -0.5] → [0.0, 0.0, -0.5]
  uvar_Jv = NLPModels.uvar_jpprod(pqp, v_param)
  @test uvar_Jv ≈ [0.0, 0.0, -0.5]

  # ── set_param_values! ──

  θ2 = [-1.0, 0.5]
  NLPModels.set_param_values!(pqp, θ2)
  @test NLPModels.get_param_values(pqp) ≈ θ2

  H_nz2, A_nz2, c_v2, c0_v2, lc2, uc2, lv2, uv2 = data_fn(θ2)
  qp_ref2 = QuadraticModel(
    c_v2, copy(Hrows), copy(Hcols), H_nz2;
    Arows = copy(Arows), Acols = copy(Acols), Avals = A_nz2,
    lcon = lc2, ucon = uc2, lvar = lv2, uvar = uv2, c0 = c0_v2,
  )

  @test obj(pqp, x) ≈ obj(qp_ref2, x)
  @test grad(pqp, x) ≈ grad(qp_ref2, x)
  @test cons(pqp, x) ≈ cons(qp_ref2, x)
  @test hprod(pqp, x, v_x) ≈ hprod(qp_ref2, x, v_x)

  # Parametric derivatives at new θ
  # ∂f/∂θ₁ = θ₂+x₁+θ₁x₂² = 0.5+1+(-1)*4 = -2.5
  # ∂f/∂θ₂ = θ₁+x₂ = -1+2 = 1
  gp2 = NLPModels.grad_param(pqp, x)
  @test gp2 ≈ [-2.5, 1.0]

  # ── LP case (H = 0) ──

  data_fn_lp(θ) = (
    Float64[],
    [1 + θ[1], 2.0, θ[2], 3.0],
    [1 + θ[1], -2 + θ[2], 3.0],
    0.5 + θ[1] * θ[2],
    [-5.0 + θ[1], -5.0], [10.0, 10.0 + θ[2]],
    [-10.0 + θ[1], -10.0, -10.0], [10.0, 10.0, 10.0 + θ[2]],
  )

  data_jac_fn_lp(θ) = (
    Float64[],                         # dH (empty)
    [1.0, 1.0],                        # dA
    [1.0, 1.0],                        # dc
    [θ[2], θ[1]],                      # dc0
    [1.0],                             # dlcon
    [1.0],                             # ducon
    [1.0],                             # dlvar
    [1.0],                             # duvar
  )

  pqp_lp = QuadraticModels.ParametricQuadraticModel(
    Int[], Int[], Arows, Acols, nvar, ncon,
    data_fn_lp, data_jac_fn_lp, [1.0, 2.0];
    dA_structure = dA_structure,
    dc_structure = dc_structure,
    dlcon_structure = dlcon_structure,
    ducon_structure = ducon_structure,
    dlvar_structure = dlvar_structure,
    duvar_structure = duvar_structure,
  )
  @test pqp_lp.meta.islp == true
  @test pqp_lp.meta.nnzh == 0

  c_eff_lp = [1 + 1.0, -2 + 2.0, 3.0]
  @test obj(pqp_lp, x) ≈ (0.5 + 1.0 * 2.0) + dot(c_eff_lp, x)
  @test grad(pqp_lp, x) ≈ c_eff_lp
  @test cons(pqp_lp, x) ≈ [1 + 1.0 2.0 0.0; 0.0 2.0 3.0] * x

  # grad_param for LP (no H term)
  gp_lp = NLPModels.grad_param(pqp_lp, x)
  # dc0=[2,1], dc'x=[1,2], no H term → [3, 3]
  @test gp_lp ≈ [3.0, 3.0]
end
