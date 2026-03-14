@testset "LinearParametricQuadraticModel" begin
  T = Float64
  nvar = 3
  ncon = 2
  nparam = 2

  c = [1.0, -2.0, 3.0]
  c0 = 0.5
  H_dense = [0.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 0.0]
  A = sparse([1.0 2.0 0.0; 0.0 1.0 3.0])
  F = [1.0 0.5; 0.0 1.0; -1.0 0.0]
  B = [0.5 0.0; 0.0 1.0]
  θ = [1.0, 2.0]

  lcon = [-5.0, -5.0]
  ucon = [10.0, 10.0]
  lvar = [-10.0, -10.0, -10.0]
  uvar = [10.0, 10.0, 10.0]

  H_sp = tril(sparse(H_dense))

  lp = QuadraticModels.LinearParametricQuadraticModel(
    c, H_sp, A, F, B, θ;
    c0 = c0, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar,
  )

  @test lp.meta.nvar == nvar
  @test lp.meta.ncon == ncon
  @test NLPModels.get_nparam(lp) == nparam
  @test lp.meta.nnzh == nnz(H_sp)

  # Effective c = c + F*θ
  c_eff = c + F * θ
  Bθ_expected = B * θ

  # Build equivalent QuadraticModel:
  #   same H, A, c_eff; cons = Ax (no Bθ in QM), so shift bounds
  qp_equiv = QuadraticModel(
    c_eff, tril(sparse(H_dense));
    A = A,
    lcon = lcon - Bθ_expected, ucon = ucon - Bθ_expected,
    lvar = lvar, uvar = uvar, c0 = c0,
  )

  x = [1.0, 2.0, 3.0]
  y = [1.0, -1.0]

  # obj and grad only depend on c_eff and H, same in both
  @test obj(lp, x) ≈ obj(qp_equiv, x)
  @test grad(lp, x) ≈ grad(qp_equiv, x)

  # cons: our model returns Ax + Bθ, QM returns Ax
  @test cons(lp, x) ≈ A * x + Bθ_expected
  @test cons(lp, x) ≈ cons(qp_equiv, x) + Bθ_expected

  # bounds are fixed at construction values
  @test lp.meta.lcon ≈ lcon
  @test lp.meta.ucon ≈ ucon

  # hprod (standard Hessian-vector product)
  v_x = [0.5, -1.0, 2.0]
  @test hprod(lp, x, v_x) ≈ hprod(qp_equiv, x, v_x)

  # jac structure & coord (Jacobian wrt x is just A, same in both)
  jrows, jcols = jac_structure(lp)
  jvals = jac_coord(lp, x)
  jrows2, jcols2 = jac_structure(qp_equiv)
  jvals2 = jac_coord(qp_equiv, x)
  @test jrows == jrows2
  @test jcols == jcols2
  @test jvals ≈ jvals2

  # hess structure & coord
  hrows, hcols = hess_structure(lp)
  hvals = hess_coord(lp, x)
  hrows2, hcols2 = hess_structure(qp_equiv)
  hvals2 = hess_coord(qp_equiv, x)
  @test hrows == hrows2
  @test hcols == hcols2
  @test hvals ≈ hvals2

  # ── Parametric API ──

  @test NLPModels.get_param_values(lp) ≈ θ

  # grad_param: ∇_θ f = F'x
  gp = NLPModels.grad_param(lp, x)
  @test gp ≈ F' * x

  # jac_param: ∂c/∂θ = B
  jp_rows, jp_cols = NLPModels.jac_param_structure(lp)
  jp_vals = NLPModels.jac_param_coord(lp, x)
  @test jp_vals ≈ vec(B)

  # jpprod: B * v
  v_param = [0.5, -0.5]
  Jv = NLPModels.jpprod(lp, x, v_param)
  @test Jv ≈ B * v_param

  # jptprod: B' * v
  Jtv = NLPModels.jptprod(lp, x, y)
  @test Jtv ≈ B' * y

  # hess_param: ∇²_{x,θ} L = obj_weight * F
  hp_rows, hp_cols = NLPModels.hess_param_structure(lp)
  hp_vals = NLPModels.hess_param_coord(lp, x, y; obj_weight = 2.0)
  @test hp_vals ≈ 2.0 * vec(F)

  # hpprod: obj_weight * F * v
  Hv = NLPModels.hpprod(lp, x, y, v_param; obj_weight = 2.0)
  @test Hv ≈ 2.0 * F * v_param

  # hptprod: obj_weight * F' * v
  Htv = NLPModels.hptprod(lp, x, y, v_x; obj_weight = 2.0)
  @test Htv ≈ 2.0 * F' * v_x

  # ── set_param_values! ──

  θ2 = [-1.0, 0.5]
  NLPModels.set_param_values!(lp, θ2)
  @test NLPModels.get_param_values(lp) ≈ θ2

  c_eff2 = c + F * θ2
  Bθ2 = B * θ2

  # obj/grad use updated c
  qp_equiv2 = QuadraticModel(
    c_eff2, tril(sparse(H_dense));
    A = A, lcon = lcon - Bθ2, ucon = ucon - Bθ2,
    lvar = lvar, uvar = uvar, c0 = c0,
  )
  @test obj(lp, x) ≈ obj(qp_equiv2, x)
  @test grad(lp, x) ≈ grad(qp_equiv2, x)

  # cons uses updated Bθ
  @test cons(lp, x) ≈ A * x + Bθ2

  # bounds unchanged
  @test lp.meta.lcon ≈ lcon
  @test lp.meta.ucon ≈ ucon

  # ── Test with COO H ──

  H_coo = SparseMatrixCOO(3, 3, [2], [2], [2.0])
  lp_coo = QuadraticModels.LinearParametricQuadraticModel(
    c, H_coo, A, F, B, θ2;
    c0 = c0, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar,
  )

  @test obj(lp_coo, x) ≈ obj(qp_equiv2, x)
  @test grad(lp_coo, x) ≈ grad(qp_equiv2, x)
  @test hprod(lp_coo, x, v_x) ≈ hprod(qp_equiv2, x, v_x)

  hrows_coo, hcols_coo = hess_structure(lp_coo)
  hvals_coo = hess_coord(lp_coo, x)
  @test hrows_coo == [2]
  @test hcols_coo == [2]
  @test hvals_coo ≈ [2.0]

  # ── Test LP case (H = 0) ──

  H_zero = SparseMatrixCOO(3, 3, Int[], Int[], Float64[])
  lp_zero = QuadraticModels.LinearParametricQuadraticModel(
    c, H_zero, A, F, B, θ2;
    c0 = c0, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar,
  )
  @test lp_zero.meta.islp == true
  @test lp_zero.meta.nnzh == 0

  c_eff_lp = c + F * θ2
  @test obj(lp_zero, x) ≈ c0 + dot(c_eff_lp, x)
  @test grad(lp_zero, x) ≈ c_eff_lp
  @test cons(lp_zero, x) ≈ A * x + Bθ2
end
