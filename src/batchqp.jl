"""
    BatchQuadraticModel{T, MT, VT, VI}

Batch quadratic model where all instances share the same sparsity structure
but may have different QP data (linear cost `c`, constraint matrix `A` values,
Hessian `H` values, constant `c0`, and bounds).

Uses two-step gather-scatter operations via BatchSparseOp.
"""
struct BatchQuadraticModel{T, MT, VT<:AbstractVector{T}, VI<:AbstractVector{Int}} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::MT           # nvar × nbatch
  c0_batch::VT          # nbatch
  H_nzvals::MT          # nnzh × nbatch
  A_nzvals::MT          # nnzj × nbatch

  hess_rows::VI
  hess_cols::VI
  A_rows::VI
  A_cols::VI

  # BatchSparseOps
  jac_op::BatchSparseOp    # A * v → ncon (group by A_rows)
  jact_op::BatchSparseOp   # A' * v → nvar (group by A_cols)
  hess_op::BatchSparseOp   # symmetric H * v → nvar

  # Workspace
  _HX::MT               # nvar × nbatch
end

"""
    BatchQuadraticModel(qps::Vector{QuadraticModel}; kwargs...)

Construct from a vector of `QuadraticModel`s that share the same sparsity structure.
"""
function BatchQuadraticModel(
  qps::Vector{QP};
  name::String = "SameStructBatchQP",
  MT = typeof(similar(first(qps).data.c, T, 0, 0)),
) where {QP <: QuadraticModel{T}} where {T}
  nbatch = length(qps)
  @assert nbatch > 0 "Need at least one model"
  qp1 = first(qps)
  nvar = qp1.meta.nvar
  ncon = qp1.meta.ncon
  nnzj = qp1.meta.nnzj
  nnzh = qp1.meta.nnzh

  for qp in qps[2:end]
    @assert qp.meta.nvar == nvar "All models must have same nvar"
    @assert qp.meta.ncon == ncon "All models must have same ncon"
    @assert qp.meta.nnzj == nnzj "All models must have same nnzj"
    @assert qp.meta.nnzh == nnzh "All models must have same nnzh"
  end

  x0   = MT(reduce(hcat, [qp.meta.x0   for qp in qps]))
  lvar = MT(reduce(hcat, [qp.meta.lvar for qp in qps]))
  uvar = MT(reduce(hcat, [qp.meta.uvar for qp in qps]))
  lcon = MT(reduce(hcat, [qp.meta.lcon for qp in qps]))
  ucon = MT(reduce(hcat, [qp.meta.ucon for qp in qps]))

  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch, nvar;
    x0 = x0, lvar = lvar, uvar = uvar,
    ncon = ncon, lcon = lcon, ucon = ucon,
    nnzj = nnzj, nnzh = nnzh,
    islp = qp1.meta.islp,
    name = name,
  )

  c_batch = MT(undef, nvar, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(c_batch, :, i), qp.data.c)
  end
  c0_batch = similar(c_batch, T, nbatch)
  copyto!(c0_batch, T[qp.data.c0 for qp in qps])

  hess_rows = similar(c_batch, Int, nnzh)
  hess_cols = similar(c_batch, Int, nnzh)
  fill_structure!(qp1.data.H, hess_rows, hess_cols)

  A_nzvals = MT(undef, nnzj, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(A_nzvals, :, i), nonzeros(qp.data.A))
  end
  H_nzvals = MT(undef, nnzh, nbatch)
  for (i, qp) in enumerate(qps)
    copyto!(view(H_nzvals, :, i), nonzeros(qp.data.H))
  end

  # Extract A structure
  A_rows_vec = similar(c_batch, Int, nnzj)
  A_cols_vec = similar(c_batch, Int, nnzj)
  fill_structure!(qp1.data.A, A_rows_vec, A_cols_vec)

  # Jac op (A * v → ncon): group by A_rows
  jac_identity = collect(1:nnzj)
  jac_rowptr, jac_colidx = _coo_to_csr(Vector{Int}(A_rows_vec), ncon)
  jac_val_map = Vector{Int}(A_cols_vec)
  jac_scatter = _coo_to_scatter(Vector{Int}(A_rows_vec), ncon, nnzj)
  jac_buffer = fill!(MT(undef, nnzj, nbatch), zero(T))
  jac_op = _build_op(jac_rowptr, jac_identity, jac_val_map, jac_colidx, jac_scatter, jac_buffer)

  # Jac transpose op (A' * v → nvar): group by A_cols
  jact_rowptr, jact_colidx = _coo_to_csr(Vector{Int}(A_cols_vec), nvar)
  jact_val_map = Vector{Int}(A_rows_vec)
  jact_scatter = _coo_to_scatter(Vector{Int}(A_cols_vec), nvar, nnzj)
  jact_buffer = fill!(MT(undef, nnzj, nbatch), zero(T))
  jact_op = _build_op(jact_rowptr, copy(jac_identity), jact_val_map, jact_colidx, jact_scatter, jact_buffer)

  # Hess symmetric op
  off_diag = findall(hess_rows .!= hess_cols)
  sym_scatter_rows = vcat(Vector{Int}(hess_rows), Vector{Int}(hess_cols[off_diag]))
  base_idx = collect(1:nnzh)
  sym_nz_idx = vcat(base_idx, Vector{Int}(off_diag))
  sym_gather_cols = vcat(Vector{Int}(hess_cols), Vector{Int}(hess_rows[off_diag]))
  sym_nnz = nnzh + length(off_diag)

  hess_rowptr, hess_colidx = _coo_to_csr(sym_scatter_rows, nvar)
  hess_scatter = _coo_to_scatter(sym_scatter_rows, nvar, sym_nnz)
  hess_buffer = fill!(MT(undef, sym_nnz, nbatch), zero(T))
  hess_op = _build_op(hess_rowptr, sym_nz_idx, sym_gather_cols, hess_colidx, hess_scatter, hess_buffer)

  VT = typeof(c0_batch)
  VI = typeof(hess_rows)
  _HX = fill!(MT(undef, nvar, nbatch), zero(T))

  return BatchQuadraticModel{T, MT, VT, VI}(
    meta,
    c_batch, c0_batch, H_nzvals, A_nzvals,
    hess_rows, hess_cols, A_rows_vec, A_cols_vec,
    jac_op, jact_op, hess_op,
    _HX,
  )
end

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where T
  batch_spmv!(bqp._HX, bqp.H_nzvals, bx, bqp.hess_op)
  bf .= bqp.c0_batch .+ vec(sum(bqp.c_batch .* bx, dims=1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims=1))
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where T
  batch_spmv!(bg, bqp.H_nzvals, bx, bqp.hess_op)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where T
  batch_spmv!(bc, bqp.A_nzvals, bx, bqp.jac_op)
  return bc
end

function NLPModels.jac_structure!(
  bqp::BatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  copyto!(jrows, bqp.A_rows)
  copyto!(jcols, bqp.A_cols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::BatchQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= bqp.A_nzvals
  return bjvals
end

function NLPModels.jprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix) where T
  batch_spmv!(bJv, bqp.A_nzvals, bv, bqp.jac_op)
  return bJv
end

function NLPModels.jtprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where T
  batch_spmv!(bJtv, bqp.A_nzvals, bv, bqp.jact_op)
  return bJtv
end

function NLPModels.hess_structure!(
  bqp::BatchQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  copyto!(hrows, bqp.hess_rows)
  copyto!(hcols, bqp.hess_cols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::BatchQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where T
  bhvals .= bqp.H_nzvals .* bobj_weight'
  return bhvals
end

function NLPModels.hprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, by::AbstractMatrix, bv::AbstractMatrix, bobj_weight::AbstractVector, bHv::AbstractMatrix) where T
  batch_spmv!(bHv, bqp.H_nzvals, bv, bqp.hess_op)
  bHv .*= bobj_weight'
  return bHv
end
