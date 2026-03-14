@testset "BatchParametricQuadraticModel" begin
  T = Float64
  nvar = 3
  ncon = 2
  nparam = 2
  nbatch = 3

  Hrows = [2]
  Hcols = [2]
  Arows = [1, 1, 2, 2]
  Acols = [1, 2, 2, 3]

  data_fn(θ) = (
    [θ[1]^2],
    [1 + θ[1], 2.0, θ[2], 3.0],
    [1 + θ[1], -2 + θ[2], 3.0],
    0.5 + θ[1] * θ[2],
    [-5.0 + θ[1], -5.0], [10.0, 10.0 + θ[2]],
    [-10.0 + θ[1], -10.0, -10.0], [10.0, 10.0, 10.0 + θ[2]],
  )

  data_jac_fn(θ) = (
    [2θ[1]],
    [1.0, 1.0],
    [1.0, 1.0],
    [θ[2], θ[1]],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
  )

  dH_structure = SparseMatrixCOO(1, 2, [1], [1], [0.0])
  dA_structure = SparseMatrixCOO(4, 2, [1, 3], [1, 2], [0.0, 0.0])
  dc_structure = SparseMatrixCOO(3, 2, [1, 2], [1, 2], [0.0, 0.0])
  dlcon_structure = SparseMatrixCOO(2, 2, [1], [1], [0.0])
  ducon_structure = SparseMatrixCOO(2, 2, [2], [2], [0.0])
  dlvar_structure = SparseMatrixCOO(3, 2, [1], [1], [0.0])
  duvar_structure = SparseMatrixCOO(3, 2, [3], [2], [0.0])

  θ_vals = [[1.0, 2.0], [0.5, -1.0], [-1.0, 0.5]]

  # Build individual models
  pqps = [QuadraticModels.ParametricQuadraticModel(
    Hrows, Hcols, Arows, Acols, nvar, ncon,
    data_fn, data_jac_fn, θ;
    dH_structure = dH_structure,
    dA_structure = dA_structure,
    dc_structure = dc_structure,
    dlcon_structure = dlcon_structure,
    ducon_structure = ducon_structure,
    dlvar_structure = dlvar_structure,
    duvar_structure = duvar_structure,
  ) for θ in θ_vals]

  # Build batch from vector
  bpqp = QuadraticModels.BatchParametricQuadraticModel(pqps)

  @testset "Metadata" begin
    @test bpqp.meta.nvar == nvar
    @test bpqp.meta.ncon == ncon
    @test bpqp.meta.nbatch == nbatch
    @test bpqp.meta.nparam == nparam
    @test bpqp.meta.nnzh == 1
    @test bpqp.meta.islp == false
    @test bpqp.meta.nnzjplcon == 1
    @test bpqp.meta.nnzjpucon == 1
    @test bpqp.meta.nnzjplvar == 1
    @test bpqp.meta.nnzjpuvar == 1
  end

  bx = reduce(hcat, [[1.0, 2.0, 3.0] for _ in 1:nbatch])
  by = reduce(hcat, [[1.0, -1.0] for _ in 1:nbatch])
  bobj_weight = ones(T, nbatch)

  @testset "obj!" begin
    bf = zeros(T, nbatch)
    NLPModels.obj!(bpqp, bx, bf)
    for j in 1:nbatch
      @test bf[j] ≈ obj(pqps[j], bx[:, j])
    end
  end

  @testset "grad!" begin
    bg = zeros(T, nvar, nbatch)
    NLPModels.grad!(bpqp, bx, bg)
    for j in 1:nbatch
      @test bg[:, j] ≈ grad(pqps[j], bx[:, j])
    end
  end

  @testset "cons!" begin
    bc = zeros(T, ncon, nbatch)
    NLPModels.cons!(bpqp, bx, bc)
    for j in 1:nbatch
      @test bc[:, j] ≈ cons(pqps[j], bx[:, j])
    end
  end

  @testset "jac_structure! and jac_coord!" begin
    jrows = zeros(Int, bpqp.meta.nnzj)
    jcols = zeros(Int, bpqp.meta.nnzj)
    NLPModels.jac_structure!(bpqp, jrows, jcols)

    jrows_seq = zeros(Int, pqps[1].meta.nnzj)
    jcols_seq = zeros(Int, pqps[1].meta.nnzj)
    NLPModels.jac_lin_structure!(pqps[1], jrows_seq, jcols_seq)
    @test jrows == jrows_seq
    @test jcols == jcols_seq

    bjvals = zeros(T, bpqp.meta.nnzj, nbatch)
    NLPModels.jac_coord!(bpqp, bx, bjvals)
    for j in 1:nbatch
      jvals_seq = jac_coord(pqps[j], bx[:, j])
      @test bjvals[:, j] ≈ jvals_seq
    end
  end

  @testset "hess_structure! and hess_coord!" begin
    hrows = zeros(Int, bpqp.meta.nnzh)
    hcols = zeros(Int, bpqp.meta.nnzh)
    NLPModels.hess_structure!(bpqp, hrows, hcols)

    hrows_seq, hcols_seq = hess_structure(pqps[1])
    @test hrows == hrows_seq
    @test hcols == hcols_seq

    bhvals = zeros(T, bpqp.meta.nnzh, nbatch)
    NLPModels.hess_coord!(bpqp, bx, by, bobj_weight, bhvals)
    for j in 1:nbatch
      hvals_seq = hess_coord(pqps[j], bx[:, j])
      @test bhvals[:, j] ≈ hvals_seq
    end
  end

  @testset "grad_param!" begin
    bg = zeros(T, nparam, nbatch)
    NLPModels.grad_param!(bpqp, bx, bg)
    for j in 1:nbatch
      gp_seq = NLPModels.grad_param(pqps[j], bx[:, j])
      @test bg[:, j] ≈ gp_seq
    end
  end

  @testset "jpprod!" begin
    bv = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])
    bJv = zeros(T, ncon, nbatch)
    NLPModels.jpprod!(bpqp, bx, bv, bJv)
    for j in 1:nbatch
      @test bJv[:, j] ≈ NLPModels.jpprod(pqps[j], bx[:, j], bv[:, j])
    end
  end

  @testset "jptprod!" begin
    bJtv = zeros(T, nparam, nbatch)
    NLPModels.jptprod!(bpqp, bx, by, bJtv)
    for j in 1:nbatch
      @test bJtv[:, j] ≈ NLPModels.jptprod(pqps[j], bx[:, j], by[:, j])
    end
  end

  @testset "hpprod!" begin
    bv = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])
    bHv = zeros(T, nvar, nbatch)
    bw = fill(2.0, nbatch)
    NLPModels.hpprod!(bpqp, bx, by, bv, bw, bHv)
    for j in 1:nbatch
      @test bHv[:, j] ≈ NLPModels.hpprod(pqps[j], bx[:, j], by[:, j], bv[:, j]; obj_weight = 2.0)
    end
  end

  @testset "hptprod!" begin
    bv = reduce(hcat, [[1.0, 2.0, 3.0] for _ in 1:nbatch])
    bHtv = zeros(T, nparam, nbatch)
    bw = fill(2.0, nbatch)
    NLPModels.hptprod!(bpqp, bx, by, bv, bw, bHtv)
    for j in 1:nbatch
      @test bHtv[:, j] ≈ NLPModels.hptprod(pqps[j], bx[:, j], by[:, j], bv[:, j]; obj_weight = 2.0)
    end
  end

  @testset "Bounds parametric API" begin
    bv_param = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])

    # lcon_jpprod
    bJv = zeros(T, ncon, nbatch)
    NLPModels.lcon_jpprod!(bpqp, bv_param, bJv)
    for j in 1:nbatch
      @test bJv[:, j] ≈ NLPModels.lcon_jpprod(pqps[j], bv_param[:, j])
    end

    # lcon_jptprod
    bJtv = zeros(T, nparam, nbatch)
    NLPModels.lcon_jptprod!(bpqp, by, bJtv)
    for j in 1:nbatch
      @test bJtv[:, j] ≈ NLPModels.lcon_jptprod(pqps[j], by[:, j])
    end

    # ucon_jpprod
    bJv2 = zeros(T, ncon, nbatch)
    NLPModels.ucon_jpprod!(bpqp, bv_param, bJv2)
    for j in 1:nbatch
      @test bJv2[:, j] ≈ NLPModels.ucon_jpprod(pqps[j], bv_param[:, j])
    end

    # lvar_jpprod
    bJv3 = zeros(T, nvar, nbatch)
    NLPModels.lvar_jpprod!(bpqp, bv_param, bJv3)
    for j in 1:nbatch
      @test bJv3[:, j] ≈ NLPModels.lvar_jpprod(pqps[j], bv_param[:, j])
    end

    # uvar_jpprod
    bJv4 = zeros(T, nvar, nbatch)
    NLPModels.uvar_jpprod!(bpqp, bv_param, bJv4)
    for j in 1:nbatch
      @test bJv4[:, j] ≈ NLPModels.uvar_jpprod(pqps[j], bv_param[:, j])
    end

    # lvar_jptprod
    bv_x = reduce(hcat, [[1.0, 2.0, 3.0] for _ in 1:nbatch])
    bJtv2 = zeros(T, nparam, nbatch)
    NLPModels.lvar_jptprod!(bpqp, bv_x, bJtv2)
    for j in 1:nbatch
      @test bJtv2[:, j] ≈ NLPModels.lvar_jptprod(pqps[j], bv_x[:, j])
    end

    # uvar_jptprod
    bJtv3 = zeros(T, nparam, nbatch)
    NLPModels.uvar_jptprod!(bpqp, bv_x, bJtv3)
    for j in 1:nbatch
      @test bJtv3[:, j] ≈ NLPModels.uvar_jptprod(pqps[j], bv_x[:, j])
    end
  end

  @testset "LP case (H = 0)" begin
    data_fn_lp(θ) = (
      Float64[],
      [1 + θ[1], 2.0, θ[2], 3.0],
      [1 + θ[1], -2 + θ[2], 3.0],
      0.5 + θ[1] * θ[2],
      [-5.0 + θ[1], -5.0], [10.0, 10.0 + θ[2]],
      [-10.0 + θ[1], -10.0, -10.0], [10.0, 10.0, 10.0 + θ[2]],
    )

    data_jac_fn_lp(θ) = (
      Float64[],
      [1.0, 1.0],
      [1.0, 1.0],
      [θ[2], θ[1]],
      [1.0],
      [1.0],
      [1.0],
      [1.0],
    )

    pqps_lp = [QuadraticModels.ParametricQuadraticModel(
      Int[], Int[], Arows, Acols, nvar, ncon,
      data_fn_lp, data_jac_fn_lp, θ;
      dA_structure = dA_structure,
      dc_structure = dc_structure,
      dlcon_structure = dlcon_structure,
      ducon_structure = ducon_structure,
      dlvar_structure = dlvar_structure,
      duvar_structure = duvar_structure,
    ) for θ in θ_vals]
    bpqp_lp = QuadraticModels.BatchParametricQuadraticModel(pqps_lp)

    @test bpqp_lp.meta.islp == true
    @test bpqp_lp.meta.nnzh == 0

    bf = zeros(T, nbatch)
    NLPModels.obj!(bpqp_lp, bx, bf)
    for j in 1:nbatch
      @test bf[j] ≈ obj(pqps_lp[j], bx[:, j])
    end
  end

  @testset "Constructor from single PQP + nbatch" begin
    bpqp2 = QuadraticModels.BatchParametricQuadraticModel(pqps[1], nbatch;
      lvar = reduce(hcat, [[-10.0 + 1.0, -10.0, -10.0] for _ in 1:nbatch]),
      uvar = reduce(hcat, [[10.0, 10.0, 10.0 + 2.0] for _ in 1:nbatch]),
      lcon = reduce(hcat, [[-5.0 + 1.0, -5.0] for _ in 1:nbatch]),
      ucon = reduce(hcat, [[10.0, 10.0 + 2.0] for _ in 1:nbatch]),
    )
    @test bpqp2.meta.nbatch == nbatch
    @test bpqp2.meta.nvar == nvar

    bf2 = zeros(T, nbatch)
    NLPModels.obj!(bpqp2, bx, bf2)
    for j in 1:nbatch
      @test bf2[j] ≈ obj(pqps[1], bx[:, j])
    end
  end

  @testset "set_param_values!" begin
    θ_new = [[2.0, -1.0], [-0.5, 1.5], [0.0, 0.0]]
    θ_batch = reduce(hcat, θ_new)

    # Update individual models
    for (j, pqp) in enumerate(pqps)
      NLPModels.set_param_values!(pqp, θ_new[j])
    end

    # Update batch model
    NLPModels.set_param_values!(bpqp, θ_batch)

    # Verify consistency
    bf = zeros(T, nbatch)
    NLPModels.obj!(bpqp, bx, bf)
    for j in 1:nbatch
      @test bf[j] ≈ obj(pqps[j], bx[:, j])
    end

    bg = zeros(T, nparam, nbatch)
    NLPModels.grad_param!(bpqp, bx, bg)
    for j in 1:nbatch
      @test bg[:, j] ≈ NLPModels.grad_param(pqps[j], bx[:, j])
    end

    # Bounds consistency after set_param_values!
    bv_param = reduce(hcat, [[0.5, -0.5] for _ in 1:nbatch])
    bJv = zeros(T, ncon, nbatch)
    NLPModels.lcon_jpprod!(bpqp, bv_param, bJv)
    for j in 1:nbatch
      @test bJv[:, j] ≈ NLPModels.lcon_jpprod(pqps[j], bv_param[:, j])
    end
  end
end
