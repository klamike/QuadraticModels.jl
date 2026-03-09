"""
    BatchQuadraticModel{T, MT, SpMT_J, SpMT_H, VT, VI}

Batch quadratic model where all instances share the same sparsity structure
but may have different QP data (linear cost `c`, constraint matrix `A` values,
Hessian `H` values, constant `c0`, and bounds).
"""
struct BatchQuadraticModel{T, MT, SpMT_J, SpMT_H, VT<:AbstractVector{T}, VI<:AbstractVector{Int}} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::MT           # nvar × nbatch
  c0_batch::VT          # nbatch
  H_nzvals::MT          # nnzh × nbatch
  A_nzvals::MT          # nnzj × nbatch

  hess_rows::VI
  hess_cols::VI

  _jac_scatter::SpMT_J    # ncon × nnzj:  S[rows[k], k] = 1
  _jact_scatter::SpMT_J   # nvar × nnzj:  S[cols[k], k] = 1
  _hess_scatter::SpMT_H   # nvar × sym_nnzh: handles lower-tri symmetrization

  _hess_sym_gather_cols::VI   # column indices for V gather
  _hess_sym_nzidx::VI         # indices into H_nzvals rows

  # Workspace
  _HX::MT               # nvar × nbatch (for Hx in obj!/grad!)
  _jac_buffer::MT        # nnzj × nbatch
  _hess_buffer::MT       # sym_nnzh × nbatch
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

  jac_rows_dev = similar(c_batch, Int, nnzj)
  jac_cols_dev = similar(c_batch, Int, nnzj)
  fill_structure!(qp1.data.A, jac_rows_dev, jac_cols_dev)
  scat_vals = fill!(similar(c_batch, T, nnzj), one(T))
  scat_idx  = similar(c_batch, Int, nnzj); scat_idx .= 1:nnzj
  _jac_scatter  = sparse(jac_rows_dev, scat_idx, scat_vals, ncon, nnzj)
  _jact_scatter = sparse(jac_cols_dev, scat_idx, scat_vals, nvar, nnzj)

  off_diag = findall(hess_rows .!= hess_cols)
  sym_scatter_rows = vcat(hess_rows, hess_cols[off_diag])
  sym_gather_cols  = vcat(hess_cols, hess_rows[off_diag])
  base_idx = similar(c_batch, Int, nnzh); base_idx .= 1:nnzh
  sym_nz_idx = vcat(base_idx, off_diag)
  sym_nnzh = nnzh + length(off_diag)

  sym_idx  = similar(c_batch, Int, sym_nnzh); sym_idx .= 1:sym_nnzh
  sym_vals = fill!(similar(c_batch, T, sym_nnzh), one(T))
  _hess_scatter = sparse(sym_scatter_rows, sym_idx, sym_vals, nvar, sym_nnzh)

  SpMT_J = typeof(_jac_scatter)
  SpMT_H = typeof(_hess_scatter)
  VT = typeof(c0_batch)
  VI = typeof(hess_rows)
  _HX = fill!(MT(undef, nvar, nbatch), zero(T))
  _jac_buffer = fill!(MT(undef, nnzj, nbatch), zero(T))
  _hess_buffer = fill!(MT(undef, sym_nnzh, nbatch), zero(T))

  return BatchQuadraticModel{T, MT, SpMT_J, SpMT_H, VT, VI}(
    meta,
    c_batch, c0_batch, H_nzvals, A_nzvals,
    hess_rows, hess_cols,
    _jac_scatter, _jact_scatter, _hess_scatter,
    sym_gather_cols, sym_nz_idx,
    _HX, _jac_buffer, _hess_buffer,
  )
end

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where T
  @views bqp._hess_buffer .= bqp.H_nzvals[bqp._hess_sym_nzidx, :] .* bx[bqp._hess_sym_gather_cols, :]
  mul!(bqp._HX, bqp._hess_scatter, bqp._hess_buffer)
  bf .= bqp.c0_batch .+ vec(sum(bqp.c_batch .* bx, dims=1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims=1))
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where T
  @views bqp._hess_buffer .= bqp.H_nzvals[bqp._hess_sym_nzidx, :] .* bx[bqp._hess_sym_gather_cols, :]
  mul!(bg, bqp._hess_scatter, bqp._hess_buffer)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where T
  @views bqp._jac_buffer .= bqp.A_nzvals .* bx[rowvals(bqp._jact_scatter), :]
  mul!(bc, bqp._jac_scatter, bqp._jac_buffer)
  return bc
end

function NLPModels.jac_structure!(
  bqp::BatchQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  @lencheck bqp.meta.nnzj jrows jcols
  copyto!(jrows, rowvals(bqp._jac_scatter))
  copyto!(jcols, rowvals(bqp._jact_scatter))
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
  @views bqp._jac_buffer .= bqp.A_nzvals .* bv[rowvals(bqp._jact_scatter), :]
  mul!(bJv, bqp._jac_scatter, bqp._jac_buffer)
  return bJv
end

function NLPModels.jtprod!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix) where T
  @views bqp._jac_buffer .= bqp.A_nzvals .* bv[rowvals(bqp._jac_scatter), :]
  mul!(bJtv, bqp._jact_scatter, bqp._jac_buffer)
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
  @views bqp._hess_buffer .= bqp.H_nzvals[bqp._hess_sym_nzidx, :] .* bobj_weight' .* bv[bqp._hess_sym_gather_cols, :]
  mul!(bHv, bqp._hess_scatter, bqp._hess_buffer)
  return bHv
end
