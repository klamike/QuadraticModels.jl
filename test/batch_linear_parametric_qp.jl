@testset "BatchLinearParametricQuadraticModel" begin
  T = Float64
  nvar = 3
  ncon = 2
  nparam = 2
  nbatch = 3

  c = [1.0, -2.0, 3.0]
  c0 = 0.5
  H_dense = [0.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 0.0]
  A = sparse([1.0 2.0 0.0; 0.0 1.0 3.0])
  F = [1.0 0.5; 0.0 1.0; -1.0 0.0]
  B = [0.5 0.0; 0.0 1.0]
  H_sp = tril(sparse(H_dense))

  θ_vals = [[1.0, 2.0], [0.5, -1.0], [-1.0, 0.5]]
  lcon = [-5.0, -5.0]
  ucon = [10.0, 10.0]
  lvar = [-10.0, -10.0, -10.0]
  uvar = [10.0, 10.0, 10.0]

  # Build sequential models
  lpqps = [QuadraticModels.LinearParametricQuadraticModel(
    c, H_sp, A, F, B, θ;
    c0 = c0, lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar,
  ) for θ in θ_vals]

  # Build batch from vector of LPQPs
  bqp = BatchLinearParametricQuadraticModel(lpqps)

  @testset "Metadata" begin
    @test bqp.meta.nvar == nvar
    @test bqp.meta.ncon == ncon
    @test bqp.meta.nbatch == nbatch
    @test bqp.meta.nparam == nparam
    @test bqp.meta.nnzh == nnz(H_sp)
    @test bqp.meta.nnzjp == length(vec(B))
    @test bqp.meta.nnzhp == length(vec(F))
    @test bqp.meta.islp == false
  end

  bx = reduce(hcat, [[1.0, 2.0, 3.0] for _ in 1:nbatch])
  by = reduce(hcat, [[1.0, -1.0] for _ in 1:nbatch])
  bobj_weight = ones(T, nbatch)

  @testset "obj!" begin
    bf = zeros(T, nbatch)
    NLPModels.obj!(bqp, bx, bf)
    for j in 1:nbatch
      @test bf[j] ≈ obj(lpqps[j], bx[:, j])
    end
  end

  @testset "grad!" begin
    bg = zeros(T, nvar, nbatch)
    NLPModels.grad!(bqp, bx, bg)
    for j in 1:nbatch
      @test bg[:, j] ≈ grad(lpqps[j], bx[:, j])
    end
  end

  @testset "cons!" begin
    bc = zeros(T, ncon, nbatch)
    NLPModels.cons!(bqp, bx, bc)
    for j in 1:nbatch
      @test bc[:, j] ≈ cons(lpqps[j], bx[:, j])
    end
  end

  @testset "jac_structure! and jac_coord!" begin
    jrows = zeros(Int, bqp.meta.nnzj)
    jcols = zeros(Int, bqp.meta.nnzj)
    NLPModels.jac_structure!(bqp, jrows, jcols)

    jrows_seq = zeros(Int, lpqps[1].meta.nnzj)
    jcols_seq = zeros(Int, lpqps[1].meta.nnzj)
    jac_structure(lpqps[1])
    NLPModels.jac_lin_structure!(lpqps[1], jrows_seq, jcols_seq)
    @test jrows == jrows_seq
    @test jcols == jcols_seq

    bjvals = zeros(T, bqp.meta.nnzj, nbatch)
    NLPModels.jac_coord!(bqp, bx, bjvals)
    jvals_seq = jac_coord(lpqps[1], bx[:, 1])
    for j in 1:nbatch
      @test bjvals[:, j] ≈ jvals_seq
    end
  end

  @testset "hess_structure! and hess_coord!" begin
    hrows = zeros(Int, bqp.meta.nnzh)
    hcols = zeros(Int, bqp.meta.nnzh)
    NLPModels.hess_structure!(bqp, hrows, hcols)

    hrows_seq, hcols_seq = hess_structure(lpqps[1])
    @test hrows == hrows_seq
    @test hcols == hcols_seq

    bhvals = zeros(T, bqp.meta.nnzh, nbatch)
    NLPModels.hess_coord!(bqp, bx, by, bobj_weight, bhvals)
    hvals_seq = hess_coord(lpqps[1], bx[:, 1])
    for j in 1:nbatch
      @test bhvals[:, j] ≈ hvals_seq
    end
  end

  @testset "grad_param!" begin
    bg = zeros(T, nparam, nbatch)
    NLPModels.grad_param!(bqp, bx, bg)
    for j in 1:nbatch
      gp_seq = NLPModels.grad_param(lpqps[j], bx[:, j])
      @test bg[:, j] ≈ gp_seq
    end
  end

  @testset "jpprod!" begin
    bv = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])
    bJv = zeros(T, ncon, nbatch)
    NLPModels.jpprod!(bqp, bx, bv, bJv)
    for j in 1:nbatch
      @test bJv[:, j] ≈ NLPModels.jpprod(lpqps[j], bx[:, j], bv[:, j])
    end
  end

  @testset "jptprod!" begin
    bJtv = zeros(T, nparam, nbatch)
    NLPModels.jptprod!(bqp, bx, by, bJtv)
    for j in 1:nbatch
      @test bJtv[:, j] ≈ NLPModels.jptprod(lpqps[j], bx[:, j], by[:, j])
    end
  end

  @testset "hpprod!" begin
    bv = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])
    bHv = zeros(T, nvar, nbatch)
    bw = fill(2.0, nbatch)
    NLPModels.hpprod!(bqp, bx, by, bv, bw, bHv)
    for j in 1:nbatch
      @test bHv[:, j] ≈ NLPModels.hpprod(lpqps[j], bx[:, j], by[:, j], bv[:, j]; obj_weight = 2.0)
    end
  end

  @testset "hptprod!" begin
    bv = reduce(hcat, [[1.0, 2.0, 3.0] for _ in 1:nbatch])
    bHtv = zeros(T, nparam, nbatch)
    bw = fill(2.0, nbatch)
    NLPModels.hptprod!(bqp, bx, by, bv, bw, bHtv)
    for j in 1:nbatch
      @test bHtv[:, j] ≈ NLPModels.hptprod(lpqps[j], bx[:, j], by[:, j], bv[:, j]; obj_weight = 2.0)
    end
  end

  @testset "LP case (H = 0)" begin
    H_zero = SparseMatrixCOO(3, 3, Int[], Int[], Float64[])
    lpqps_lp = [QuadraticModels.LinearParametricQuadraticModel(
      c, H_zero, A, F, B, θ;
      lcon = lcon, ucon = ucon, lvar = lvar, uvar = uvar,
    ) for θ in θ_vals]
    bqp_lp = BatchLinearParametricQuadraticModel(lpqps_lp)

    @test bqp_lp.meta.islp == true
    @test bqp_lp.meta.nnzh == 0

    bf = zeros(T, nbatch)
    NLPModels.obj!(bqp_lp, bx, bf)
    for j in 1:nbatch
      @test bf[j] ≈ obj(lpqps_lp[j], bx[:, j])
    end
  end

  @testset "Constructor from single LPQP + nbatch" begin
    bqp2 = BatchLinearParametricQuadraticModel(lpqps[1], nbatch;
      lvar = reduce(hcat, [lvar for _ in 1:nbatch]),
      uvar = reduce(hcat, [uvar for _ in 1:nbatch]),
      lcon = reduce(hcat, [lcon for _ in 1:nbatch]),
      ucon = reduce(hcat, [ucon for _ in 1:nbatch]),
    )
    @test bqp2.meta.nbatch == nbatch
    @test bqp2.meta.nvar == nvar

    bf2 = zeros(T, nbatch)
    NLPModels.obj!(bqp2, bx, bf2)
    # All instances use the same θ from lpqps[1]
    for j in 1:nbatch
      @test bf2[j] ≈ obj(lpqps[1], bx[:, j])
    end
  end
end
