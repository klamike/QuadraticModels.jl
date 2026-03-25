"""
    BatchedSparseJac{T, MT, VI}

A batched sparse Jacobian: same sparsity pattern, per-batch nzvals.
Supports natively batched prod/tprod via two-step BatchSparseOp operations.
"""
struct BatchedSparseJac{T, MT <: AbstractMatrix{T}, VI <: AbstractVector{Int}}
  nrow::Int
  ncol::Int
  nnz_jac::Int
  nzvals::MT              # nnz_jac × nbatch
  prod_op::BatchSparseOp  # M * V → nrow × nbatch
  tprod_op::BatchSparseOp # M' * V → ncol × nbatch
end

function BatchedSparseJac(
  coo::SparseMatrixCOO{T},
  nbatch::Int;
  MT = Matrix{T},
) where {T}
  nnz_jac = length(coo.vals)
  nrow = coo.m
  ncol = coo.n

  if nnz_jac == 0
    empty_vi = Int[]
    nzvals = MT(undef, 0, nbatch)
    prod_rowptr = ones(Int, nrow + 1)
    prod_op = _build_op(nzvals, prod_rowptr, empty_vi, empty_vi, empty_vi)
    tprod_rowptr = ones(Int, ncol + 1)
    tprod_op = _build_op(nzvals, tprod_rowptr, empty_vi, empty_vi, empty_vi)
    return BatchedSparseJac{T, MT, typeof(empty_vi)}(
      nrow, ncol, 0, nzvals, prod_op, tprod_op,
    )
  end

  nzvals = fill!(MT(undef, nnz_jac, nbatch), zero(T))
  identity_map = collect(1:nnz_jac)

  prod_rowptr, prod_colidx = _coo_to_csr(coo.rows, nrow)
  prod_val_map = copy(coo.cols)
  prod_op = _build_op(nzvals, prod_rowptr, identity_map, prod_val_map, prod_colidx)

  tprod_rowptr, tprod_colidx = _coo_to_csr(coo.cols, ncol)
  tprod_val_map = copy(coo.rows)
  tprod_op = _build_op(nzvals, tprod_rowptr, copy(identity_map), tprod_val_map, tprod_colidx)

  VI = Vector{Int}
  return BatchedSparseJac{T, MT, VI}(
    nrow, ncol, nnz_jac, nzvals, prod_op, tprod_op,
  )
end

# result = M * V  (nrow × nbatch)
function bsj_prod!(result::AbstractMatrix, bsj::BatchedSparseJac, V::AbstractMatrix)
  if bsj.nnz_jac == 0
    fill!(result, zero(eltype(result)))
    return result
  end
  batch_spmv!(result, bsj.prod_op, V)
  return result
end

# result = M' * V  (ncol × nbatch)
function bsj_tprod!(result::AbstractMatrix, bsj::BatchedSparseJac, V::AbstractMatrix)
  if bsj.nnz_jac == 0
    fill!(result, zero(eltype(result)))
    return result
  end
  batch_spmv!(result, bsj.tprod_op, V)
  return result
end

# result += α * M * V
function bsj_prod_add!(result::AbstractMatrix, bsj::BatchedSparseJac, V::AbstractMatrix, α = one(eltype(result)))
  if bsj.nnz_jac == 0
    return result
  end
  batch_spmv!(result, bsj.prod_op, V, α, one(α))
  return result
end

# result += α * M' * V
function bsj_tprod_add!(result::AbstractMatrix, bsj::BatchedSparseJac, V::AbstractMatrix, α = one(eltype(result)))
  if bsj.nnz_jac == 0
    return result
  end
  batch_spmv!(result, bsj.tprod_op, V, α, one(α))
  return result
end

"""
    BatchParametricQuadraticModel

Batch version of `ParametricQuadraticModel` with natively batched parametric ops
via two-step BatchSparseOp operations.
"""
struct BatchParametricQuadraticModel{T, MT, VT<:AbstractVector{T}, VI<:AbstractVector{Int}, BSJ <: BatchedSparseJac{T}, DF, DJF} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}

  # Shared callbacks
  data_fn::DF
  data_jac_fn::DJF

  # Parameters
  θ::MT                       # nparam × nbatch

  # Per-batch QP data
  c_batch::MT                 # nvar × nbatch
  c0_batch::VT                # nbatch
  H_nzvals::MT                # nnzh × nbatch
  A_nzvals::MT                # nnzj × nbatch

  # Per-batch scalar dc0
  dc0::MT                     # nparam × nbatch

  # Shared structure
  hess_rows::VI
  hess_cols::VI
  H_sym::VT                   # symmetry factors
  A_rows::VI
  A_cols::VI

  # BatchSparseOps for jac, jact, hess
  jac_op::BatchSparseOp
  jact_op::BatchSparseOp
  hess_op::BatchSparseOp

  # BatchedSparseJac for each parametric Jacobian
  dc_jac::BSJ
  dH_jac::BSJ
  dA_jac::BSJ
  dlcon_jac::BSJ
  ducon_jac::BSJ
  dlvar_jac::BSJ
  duvar_jac::BSJ

  # Work buffers
  _HX::MT                     # nvar × nbatch
  _u_h_batch::MT               # nnzh × nbatch
  _u_a_batch::MT               # nnzj × nbatch
end

function BatchParametricQuadraticModel(
  pqps::Vector{<:ParametricQuadraticModel{T}};
  name::String = "BatchParametricQP",
  MT = typeof(similar(first(pqps).data.c, T, 0, 0)),
) where {T}
  nbatch = length(pqps)
  @assert nbatch > 0
  pqp1 = first(pqps)
  nvar = pqp1.meta.nvar
  ncon = pqp1.meta.ncon
  nnzj = pqp1.meta.nnzj
  nnzh = pqp1.meta.nnzh
  nparam = length(pqp1.θ)

  lvar = MT(reduce(hcat, [pqp.meta.lvar for pqp in pqps]))
  uvar = MT(reduce(hcat, [pqp.meta.uvar for pqp in pqps]))
  lcon = MT(reduce(hcat, [pqp.meta.lcon for pqp in pqps]))
  ucon = MT(reduce(hcat, [pqp.meta.ucon for pqp in pqps]))

  has_param = nparam > 0
  has_con = ncon > 0

  nnzjplcon = pqp1.meta.nnzjplcon
  nnzjpucon = pqp1.meta.nnzjpucon
  nnzjplvar = pqp1.meta.nnzjplvar
  nnzjpuvar = pqp1.meta.nnzjpuvar

  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch, nvar;
    lvar = lvar, uvar = uvar,
    ncon = ncon, lcon = lcon, ucon = ucon,
    nnzj = nnzj, nnzh = nnzh,
    islp = (nnzh == 0),
    name = name,
    nparam = nparam,
    nnzgp = nparam,
    nnzjp = ncon * nparam,
    nnzhp = nvar * nparam,
    grad_param_available = has_param,
    jac_param_available = has_param && has_con,
    hess_param_available = has_param,
    jpprod_available = has_param && has_con,
    jptprod_available = has_param && has_con,
    hpprod_available = has_param,
    hptprod_available = has_param,
    nnzjplcon = nnzjplcon,
    nnzjpucon = nnzjpucon,
    nnzjplvar = nnzjplvar,
    nnzjpuvar = nnzjpuvar,
    lcon_jac_available = pqp1.meta.lcon_jac_available,
    ucon_jac_available = pqp1.meta.ucon_jac_available,
    lvar_jac_available = pqp1.meta.lvar_jac_available,
    uvar_jac_available = pqp1.meta.uvar_jac_available,
    lcon_jpprod_available = pqp1.meta.lcon_jpprod_available,
    lcon_jptprod_available = pqp1.meta.lcon_jptprod_available,
    ucon_jpprod_available = pqp1.meta.ucon_jpprod_available,
    ucon_jptprod_available = pqp1.meta.ucon_jptprod_available,
    lvar_jpprod_available = pqp1.meta.lvar_jpprod_available,
    lvar_jptprod_available = pqp1.meta.lvar_jptprod_available,
    uvar_jpprod_available = pqp1.meta.uvar_jpprod_available,
    uvar_jptprod_available = pqp1.meta.uvar_jptprod_available,
  )

  θ = MT(reduce(hcat, [pqp.θ for pqp in pqps]))

  c_batch = MT(undef, nvar, nbatch)
  for (i, pqp) in enumerate(pqps)
    copyto!(view(c_batch, :, i), pqp.data.c)
  end
  c0_batch = similar(c_batch, T, nbatch)
  copyto!(c0_batch, T[pqp.data.c0 for pqp in pqps])

  H_nzvals_mat = MT(undef, nnzh, nbatch)
  for (i, pqp) in enumerate(pqps)
    copyto!(view(H_nzvals_mat, :, i), pqp.data.H.vals)
  end
  A_nzvals_mat = MT(undef, nnzj, nbatch)
  for (i, pqp) in enumerate(pqps)
    copyto!(view(A_nzvals_mat, :, i), pqp.data.A.vals)
  end

  # dc0
  dc0_mat = MT(undef, nparam, nbatch)
  for (i, pqp) in enumerate(pqps)
    copyto!(view(dc0_mat, :, i), pqp.dc0)
  end

  # Structure from first element (shared)
  hess_rows = similar(c_batch, Int, nnzh)
  hess_cols = similar(c_batch, Int, nnzh)
  copyto!(hess_rows, pqp1.H_rows)
  copyto!(hess_cols, pqp1.H_cols)

  H_sym_vec = similar(c0_batch, nnzh)
  copyto!(H_sym_vec, pqp1.H_sym)

  A_rows_vec = similar(c_batch, Int, nnzj)
  A_cols_vec = similar(c_batch, Int, nnzj)
  copyto!(A_rows_vec, pqp1.A_rows)
  copyto!(A_cols_vec, pqp1.A_cols)

  # Build jac op
  jac_identity = collect(1:nnzj)
  jac_rowptr, jac_colidx = _coo_to_csr(Vector{Int}(A_rows_vec), ncon)
  jac_val_map = Vector{Int}(A_cols_vec)
  jac_op = _build_op(A_nzvals, jac_rowptr, jac_identity, jac_val_map, jac_colidx)

  # Build jact op
  jact_rowptr, jact_colidx = _coo_to_csr(Vector{Int}(A_cols_vec), nvar)
  jact_val_map = Vector{Int}(A_rows_vec)
  jact_op = _build_op(A_nzvals, jact_rowptr, copy(jac_identity), jact_val_map, jact_colidx)

  # Build hess symmetric op
  off_diag = findall(hess_rows .!= hess_cols)
  sym_scatter_rows = vcat(Vector{Int}(hess_rows), Vector{Int}(hess_cols[off_diag]))
  base_idx = collect(1:nnzh)
  sym_nz_idx = vcat(base_idx, Vector{Int}(off_diag))
  sym_gather_cols = vcat(Vector{Int}(hess_cols), Vector{Int}(hess_rows[off_diag]))

  hess_rowptr, hess_colidx = _coo_to_csr(sym_scatter_rows, nvar)
  hess_op = _build_op(H_nzvals, hess_rowptr, sym_nz_idx, sym_gather_cols, hess_colidx)

  # Build BatchedSparseJac for each parametric Jacobian
  dc_jac = BatchedSparseJac(pqp1.dc, nbatch; MT = MT)
  dH_jac = BatchedSparseJac(pqp1.dH, nbatch; MT = MT)
  dA_jac = BatchedSparseJac(pqp1.dA, nbatch; MT = MT)
  dlcon_jac = BatchedSparseJac(pqp1.dlcon, nbatch; MT = MT)
  ducon_jac = BatchedSparseJac(pqp1.ducon, nbatch; MT = MT)
  dlvar_jac = BatchedSparseJac(pqp1.dlvar, nbatch; MT = MT)
  duvar_jac = BatchedSparseJac(pqp1.duvar, nbatch; MT = MT)

  # Fill nzvals from individual models
  for (i, pqp) in enumerate(pqps)
    if dc_jac.nnz_jac > 0
      copyto!(view(dc_jac.nzvals, :, i), pqp.dc.vals)
    end
    if dH_jac.nnz_jac > 0
      copyto!(view(dH_jac.nzvals, :, i), pqp.dH.vals)
    end
    if dA_jac.nnz_jac > 0
      copyto!(view(dA_jac.nzvals, :, i), pqp.dA.vals)
    end
    if dlcon_jac.nnz_jac > 0
      copyto!(view(dlcon_jac.nzvals, :, i), pqp.dlcon.vals)
    end
    if ducon_jac.nnz_jac > 0
      copyto!(view(ducon_jac.nzvals, :, i), pqp.ducon.vals)
    end
    if dlvar_jac.nnz_jac > 0
      copyto!(view(dlvar_jac.nzvals, :, i), pqp.dlvar.vals)
    end
    if duvar_jac.nnz_jac > 0
      copyto!(view(duvar_jac.nzvals, :, i), pqp.duvar.vals)
    end
  end

  VT = typeof(c0_batch)
  VI = typeof(hess_rows)
  BSJ = typeof(dc_jac)

  _HX = fill!(MT(undef, nvar, nbatch), zero(T))
  _u_h_batch = fill!(MT(undef, nnzh, nbatch), zero(T))
  _u_a_batch = fill!(MT(undef, nnzj, nbatch), zero(T))

  return BatchParametricQuadraticModel{T, MT, VT, VI, BSJ, typeof(pqp1.data_fn), typeof(pqp1.data_jac_fn)}(
    meta, pqp1.data_fn, pqp1.data_jac_fn,
    θ, c_batch, c0_batch, H_nzvals_mat, A_nzvals_mat,
    dc0_mat,
    hess_rows, hess_cols, H_sym_vec, A_rows_vec, A_cols_vec,
    jac_op, jact_op, hess_op,
    dc_jac, dH_jac, dA_jac,
    dlcon_jac, ducon_jac, dlvar_jac, duvar_jac,
    _HX, _u_h_batch, _u_a_batch,
  )
end

# Constructor from single PQP + nbatch
function BatchParametricQuadraticModel(
  pqp::ParametricQuadraticModel{T},
  nbatch::Int;
  MT = typeof(similar(pqp.data.c, T, 0, 0)),
  θ = copyto!(MT(undef, length(pqp.θ), nbatch), repeat(pqp.θ, 1, nbatch)),
  lvar = fill!(MT(undef, pqp.meta.nvar, nbatch), T(-Inf)),
  uvar = fill!(MT(undef, pqp.meta.nvar, nbatch), T(Inf)),
  lcon = fill!(MT(undef, pqp.meta.ncon, nbatch), T(-Inf)),
  ucon = fill!(MT(undef, pqp.meta.ncon, nbatch), T(Inf)),
  name::String = "BatchParametricQP",
) where {T}
  nparam = length(pqp.θ)
  pqps = Vector{typeof(pqp)}(undef, nbatch)
  for i in 1:nbatch
    θ_i = θ[:, i]
    pqp_i = ParametricQuadraticModel(
      pqp.H_rows, pqp.H_cols, pqp.A_rows, pqp.A_cols,
      pqp.meta.nvar, pqp.meta.ncon,
      pqp.data_fn, pqp.data_jac_fn, θ_i;
      name = name,
      dH_structure = _coo_structure(pqp.dH),
      dA_structure = _coo_structure(pqp.dA),
      dc_structure = _coo_structure(pqp.dc),
      dlcon_structure = _coo_structure(pqp.dlcon),
      ducon_structure = _coo_structure(pqp.ducon),
      dlvar_structure = _coo_structure(pqp.dlvar),
      duvar_structure = _coo_structure(pqp.duvar),
    )
    copyto!(pqp_i.meta.lvar, view(lvar, :, i))
    copyto!(pqp_i.meta.uvar, view(uvar, :, i))
    copyto!(pqp_i.meta.lcon, view(lcon, :, i))
    copyto!(pqp_i.meta.ucon, view(ucon, :, i))
    pqps[i] = pqp_i
  end
  return BatchParametricQuadraticModel(pqps; name = name, MT = MT)
end

# Extract structure-only COO (zero values, same rows/cols/dims)
function _coo_structure(coo::SparseMatrixCOO{T}) where {T}
  SparseMatrixCOO(coo.m, coo.n, copy(coo.rows), copy(coo.cols), zeros(T, length(coo.vals)))
end

# ── set_param_values! ──

function NLPModels.set_param_values!(bpqp::BatchParametricQuadraticModel, θ_batch::AbstractMatrix)
  copyto!(bpqp.θ, θ_batch)
  for i in 1:bpqp.meta.nbatch
    θ_i = view(θ_batch, :, i)
    H_nzvals, A_nzvals, c, c0, lcon, ucon, lvar, uvar = bpqp.data_fn(θ_i)
    copyto!(view(bpqp.H_nzvals, :, i), H_nzvals)
    copyto!(view(bpqp.A_nzvals, :, i), A_nzvals)
    copyto!(view(bpqp.c_batch, :, i), c)
    bpqp.c0_batch[i] = c0
    copyto!(view(bpqp.meta.lvar, :, i), lvar)
    copyto!(view(bpqp.meta.uvar, :, i), uvar)
    copyto!(view(bpqp.meta.lcon, :, i), lcon)
    copyto!(view(bpqp.meta.ucon, :, i), ucon)

    jac_result = bpqp.data_jac_fn(θ_i)
    dH_nz, dA_nz, dc_nz, dc0_v, dlcon_nz, ducon_nz, dlvar_nz, duvar_nz = jac_result
    copyto!(view(bpqp.dc0, :, i), dc0_v)
    if bpqp.dH_jac.nnz_jac > 0
      copyto!(view(bpqp.dH_jac.nzvals, :, i), dH_nz)
    end
    if bpqp.dA_jac.nnz_jac > 0
      copyto!(view(bpqp.dA_jac.nzvals, :, i), dA_nz)
    end
    if bpqp.dc_jac.nnz_jac > 0
      copyto!(view(bpqp.dc_jac.nzvals, :, i), dc_nz)
    end
    if bpqp.dlcon_jac.nnz_jac > 0
      copyto!(view(bpqp.dlcon_jac.nzvals, :, i), dlcon_nz)
    end
    if bpqp.ducon_jac.nnz_jac > 0
      copyto!(view(bpqp.ducon_jac.nzvals, :, i), ducon_nz)
    end
    if bpqp.dlvar_jac.nnz_jac > 0
      copyto!(view(bpqp.dlvar_jac.nzvals, :, i), dlvar_nz)
    end
    if bpqp.duvar_jac.nnz_jac > 0
      copyto!(view(bpqp.duvar_jac.nzvals, :, i), duvar_nz)
    end
  end
  return bpqp
end

# ── Standard NLP API (using BatchSparseOp) ──

function NLPModels.obj!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where T
  batch_spmv!(bpqp._HX, bpqp.hess_op, bx)
  bf .= bpqp.c0_batch .+ vec(sum(bpqp.c_batch .* bx, dims=1)) .+ T(0.5) .* vec(sum(bx .* bpqp._HX, dims=1))
  return bf
end

function NLPModels.grad!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where T
  batch_spmv!(bg, bpqp.hess_op, bx)
  bg .+= bpqp.c_batch
  return bg
end

function NLPModels.cons!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where T
  batch_spmv!(bc, bpqp.jac_op, bx)
  return bc
end

function NLPModels.jac_structure!(
  bpqp::BatchParametricQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  copyto!(jrows, bpqp.A_rows)
  copyto!(jcols, bpqp.A_cols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bpqp::BatchParametricQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= bpqp.A_nzvals
  return bjvals
end

function NLPModels.jprod!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix) where T
  batch_spmv!(bJv, bpqp.jac_op, bv)
  return bJv
end

function NLPModels.jtprod!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where T
  batch_spmv!(bJtv, bpqp.jact_op, bv)
  return bJtv
end

function NLPModels.hess_structure!(
  bpqp::BatchParametricQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  copyto!(hrows, bpqp.hess_rows)
  copyto!(hcols, bpqp.hess_cols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bpqp::BatchParametricQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where T
  bhvals .= bpqp.H_nzvals .* bobj_weight'
  return bhvals
end

function NLPModels.hprod!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where T
  batch_spmv!(bHv, bpqp.hess_op, bv)
  bHv .*= bobj_weight'
  return bHv
end

# ── Natively Batched Parametric API ──

# grad_param!: bg = dc'*bx + dc0 + 0.5 * dH' * u_h
function NLPModels.grad_param!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where T
  nnzh = bpqp.meta.nnzh

  # bg = dc' * bx
  bsj_tprod!(bg, bpqp.dc_jac, bx)
  # bg += dc0
  bg .+= bpqp.dc0
  # H contribution
  if nnzh > 0
    _gather_mul!(bpqp._u_h_batch, bx, bpqp.hess_rows, bx, bpqp.hess_cols)
    bpqp._u_h_batch .*= bpqp.H_sym
    bsj_tprod_add!(bg, bpqp.dH_jac, bpqp._u_h_batch, one(T) / 2)
  end
  return bg
end

# jpprod!: compute dA*bv per nonzero, multiply by x[A_cols], scatter by A_rows
function NLPModels.jpprod!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix) where T
  nnzj = bpqp.meta.nnzj
  if nnzj == 0
    fill!(bJv, zero(T))
    return bJv
  end
  # _u_a_batch = dA * bv (nnzj × nbatch)
  bsj_prod!(bpqp._u_a_batch, bpqp.dA_jac, bv)
  # bJv[r,j] = Σ u_a[k,j] * bx[A_cols[k],j] for A_rows[k]==r
  batch_spmv!(bJv, bpqp.jac_op, bx)
  return bJv
end

# jptprod!: bJtv = dA' * (bx[A_cols,:] .* bv[A_rows,:])
function NLPModels.jptprod!(bpqp::BatchParametricQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where T
  nnzj = bpqp.meta.nnzj
  if nnzj == 0
    fill!(bJtv, zero(T))
    return bJtv
  end
  # _u_a_batch[k,j] = bx[A_cols[k],j] * bv[A_rows[k],j]
  _gather_mul!(bpqp._u_a_batch, bx, bpqp.A_cols, bv, bpqp.A_rows)
  bsj_tprod!(bJtv, bpqp.dA_jac, bpqp._u_a_batch)
  return bJtv
end

# hpprod!: (∇²_{x,θ} L) v
function NLPModels.hpprod!(
  bpqp::BatchParametricQuadraticModel{T},
  bx::AbstractMatrix, by::AbstractMatrix,
  bv::AbstractMatrix, bobj_weight::AbstractVector,
  bHv::AbstractMatrix,
) where T
  nnzh = bpqp.meta.nnzh
  nnzj = bpqp.meta.nnzj

  # bHv = dc * bv
  bsj_prod!(bHv, bpqp.dc_jac, bv)
  bHv .*= bobj_weight'

  if nnzh > 0
    # _u_h_batch = dH * bv
    bsj_prod!(bpqp._u_h_batch, bpqp.dH_jac, bv)
    # Symmetric H*x scatter
    batch_spmv!(bpqp._HX, bpqp.hess_op, bx)
    bHv .+= bpqp._HX .* bobj_weight'
  end

  if nnzj > 0
    # _u_a_batch = dA * bv
    bsj_prod!(bpqp._u_a_batch, bpqp.dA_jac, bv)
    # (∂A/∂θ·v)' * y
    batch_spmv!(bpqp._HX, bpqp.jact_op, by)
    bHv .+= bpqp._HX
  end

  return bHv
end

# hptprod!: (∇²_{x,θ} L)' v
function NLPModels.hptprod!(
  bpqp::BatchParametricQuadraticModel{T},
  bx::AbstractMatrix, by::AbstractMatrix,
  bv::AbstractMatrix, bobj_weight::AbstractVector,
  bHtv::AbstractMatrix,
) where T
  nnzh = bpqp.meta.nnzh
  nnzj = bpqp.meta.nnzj

  # bHtv = dc' * bv
  bsj_tprod!(bHtv, bpqp.dc_jac, bv)
  bHtv .*= bobj_weight'

  if nnzh > 0
    # u_h[k] = x[row]*v[col] + (sym-1)*x[col]*v[row]
    _gather_mul!(bpqp._u_h_batch, bx, bpqp.hess_rows, bv, bpqp.hess_cols)
    # Add off-diagonal symmetric contribution
    @views bpqp._u_h_batch .+= (bpqp.H_sym .- 1) .* bx[bpqp.hess_cols, :] .* bv[bpqp.hess_rows, :]
    bpqp._u_h_batch .*= bobj_weight'
    bsj_tprod_add!(bHtv, bpqp.dH_jac, bpqp._u_h_batch)
  end

  if nnzj > 0
    # _u_a_batch[k,j] = by[A_rows[k],j] * bv[A_cols[k],j]
    _gather_mul!(bpqp._u_a_batch, by, bpqp.A_rows, bv, bpqp.A_cols)
    bsj_tprod_add!(bHtv, bpqp.dA_jac, bpqp._u_a_batch)
  end

  return bHtv
end

# ── Bounds parametric API (batched) ──

function NLPModels.lcon_jpprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJv::AbstractMatrix)
  bsj_prod!(bJv, bpqp.dlcon_jac, bv)
  return bJv
end

function NLPModels.lcon_jptprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJtv::AbstractMatrix)
  bsj_tprod!(bJtv, bpqp.dlcon_jac, bv)
  return bJtv
end

function NLPModels.ucon_jpprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJv::AbstractMatrix)
  bsj_prod!(bJv, bpqp.ducon_jac, bv)
  return bJv
end

function NLPModels.ucon_jptprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJtv::AbstractMatrix)
  bsj_tprod!(bJtv, bpqp.ducon_jac, bv)
  return bJtv
end

function NLPModels.lvar_jpprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJv::AbstractMatrix)
  bsj_prod!(bJv, bpqp.dlvar_jac, bv)
  return bJv
end

function NLPModels.lvar_jptprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJtv::AbstractMatrix)
  bsj_tprod!(bJtv, bpqp.dlvar_jac, bv)
  return bJtv
end

function NLPModels.uvar_jpprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJv::AbstractMatrix)
  bsj_prod!(bJv, bpqp.duvar_jac, bv)
  return bJv
end

function NLPModels.uvar_jptprod!(bpqp::BatchParametricQuadraticModel, bv::AbstractMatrix, bJtv::AbstractMatrix)
  bsj_tprod!(bJtv, bpqp.duvar_jac, bv)
  return bJtv
end
