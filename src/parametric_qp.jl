"""
    ParametricQuadraticModel

A fully parametric quadratic program:

    min  c0(θ) + c(θ)ᵀx + ½xᵀH(θ)x
    s.t. lcon(θ) ≤ A(θ)x ≤ ucon(θ)
         lvar(θ) ≤ x ≤ uvar(θ)

The sparsity structure of H and A is fixed; only values change with θ.
User provides:
- `data_fn(θ)` → `(H_nzvals, A_nzvals, c, c0, lcon, ucon, lvar, uvar)`
- `data_jac_fn(θ)` → `(dH_nzvals, dA_nzvals, dc_nzvals, dc0, dlcon_nzvals, ducon_nzvals, dlvar_nzvals, duvar_nzvals)`

All Jacobians are stored as SparseMatrixCOO. Sparsity structure is provided via
keyword arguments (defaults to empty COO matrices).

Both callbacks are invoked once on `set_param_values!` and results are cached.
"""
mutable struct ParametricQuadraticModel{T, S, M1, M2, DF, DJF} <: AbstractQuadraticModel{T, S}
  qp::QuadraticModel{T, S, M1, M2}
  θ::S                # nparam
  data_fn::DF         # θ → (H_nzvals, A_nzvals, c, c0, lcon, ucon, lvar, uvar)
  data_jac_fn::DJF    # θ → (dH_nzvals, dA_nzvals, dc_nzvals, dc0, dlcon_nzvals, ducon_nzvals, dlvar_nzvals, duvar_nzvals)
  # Cached Jacobians as SparseMatrixCOO
  dH::SparseMatrixCOO{T, Int}     # nnzh × nparam
  dA::SparseMatrixCOO{T, Int}     # nnzj × nparam
  dc::SparseMatrixCOO{T, Int}     # nvar × nparam
  dc0::S                           # nparam
  dlcon::SparseMatrixCOO{T, Int}  # ncon × nparam
  ducon::SparseMatrixCOO{T, Int}  # ncon × nparam
  dlvar::SparseMatrixCOO{T, Int}  # nvar × nparam
  duvar::SparseMatrixCOO{T, Int}  # nvar × nparam
  # Cached sparsity structure
  H_rows::Vector{Int}
  H_cols::Vector{Int}
  H_sym::S            # 1 for diagonal, 2 for off-diagonal
  A_rows::Vector{Int}
  A_cols::Vector{Int}
  # Work buffers
  _u_h::S             # nnzh
  _u_a::S             # nnzj
end

# Forward .meta, .counters, .data to the inner QuadraticModel
function Base.getproperty(nlp::ParametricQuadraticModel, s::Symbol)
  if s in (:qp, :θ, :data_fn, :data_jac_fn,
           :dH, :dA, :dc, :dc0, :dlcon, :ducon, :dlvar, :duvar,
           :H_rows, :H_cols, :H_sym, :A_rows, :A_cols, :_u_h, :_u_a)
    return getfield(nlp, s)
  else
    return getproperty(getfield(nlp, :qp), s)
  end
end

function ParametricQuadraticModel(
  Hrows::AbstractVector{Int},
  Hcols::AbstractVector{Int},
  Arows::AbstractVector{Int},
  Acols::AbstractVector{Int},
  nvar::Int,
  ncon::Int,
  data_fn::DF,
  data_jac_fn::DJF,
  θ::S;
  name::String = "ParametricQP",
  dH_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, length(Hrows), length(θ)),
  dA_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, length(Arows), length(θ)),
  dc_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, nvar, length(θ)),
  dlcon_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, ncon, length(θ)),
  ducon_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, ncon, length(θ)),
  dlvar_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, nvar, length(θ)),
  duvar_structure::SparseMatrixCOO{T, Int} = _empty_coo(T, nvar, length(θ)),
) where {T, S <: AbstractVector{T}, DF, DJF}
  nnzh = length(Hrows)
  nnzj = length(Arows)
  nparam = length(θ)

  H_nzvals, A_nzvals, c, c0, lcon, ucon, lvar, uvar = data_fn(θ)

  has_param = nparam > 0
  has_con = ncon > 0

  # Compute nnz for bounds Jacobians
  nnzjplcon = length(dlcon_structure.vals)
  nnzjpucon = length(ducon_structure.vals)
  nnzjplvar = length(dlvar_structure.vals)
  nnzjpuvar = length(duvar_structure.vals)

  has_bounds_con = has_con && has_param
  has_bounds_var = nvar > 0 && has_param

  qp = QuadraticModel(
    S(c), Vector{Int}(Hrows), Vector{Int}(Hcols), S(H_nzvals);
    Arows = Vector{Int}(Arows), Acols = Vector{Int}(Acols), Avals = S(A_nzvals),
    lcon = S(lcon), ucon = S(ucon), lvar = S(lvar), uvar = S(uvar),
    c0 = T(c0), name = name,
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
    lcon_jac_available = has_bounds_con && nnzjplcon > 0,
    ucon_jac_available = has_bounds_con && nnzjpucon > 0,
    lvar_jac_available = has_bounds_var && nnzjplvar > 0,
    uvar_jac_available = has_bounds_var && nnzjpuvar > 0,
    lcon_jpprod_available = has_bounds_con && nnzjplcon > 0,
    lcon_jptprod_available = has_bounds_con && nnzjplcon > 0,
    ucon_jpprod_available = has_bounds_con && nnzjpucon > 0,
    ucon_jptprod_available = has_bounds_con && nnzjpucon > 0,
    lvar_jpprod_available = has_bounds_var && nnzjplvar > 0,
    lvar_jptprod_available = has_bounds_var && nnzjplvar > 0,
    uvar_jpprod_available = has_bounds_var && nnzjpuvar > 0,
    uvar_jptprod_available = has_bounds_var && nnzjpuvar > 0,
  )

  jac_result = data_jac_fn(θ)
  dH_nzvals_init, dA_nzvals_init, dc_nzvals_init, dc0_init,
    dlcon_nzvals_init, ducon_nzvals_init, dlvar_nzvals_init, duvar_nzvals_init = jac_result

  # Build COO Jacobians from structures + nzvals
  dH = SparseMatrixCOO(dH_structure.m, dH_structure.n,
    copy(dH_structure.rows), copy(dH_structure.cols), T.(dH_nzvals_init))
  dA = SparseMatrixCOO(dA_structure.m, dA_structure.n,
    copy(dA_structure.rows), copy(dA_structure.cols), T.(dA_nzvals_init))
  dc = SparseMatrixCOO(dc_structure.m, dc_structure.n,
    copy(dc_structure.rows), copy(dc_structure.cols), T.(dc_nzvals_init))
  dlcon = SparseMatrixCOO(dlcon_structure.m, dlcon_structure.n,
    copy(dlcon_structure.rows), copy(dlcon_structure.cols), T.(dlcon_nzvals_init))
  ducon = SparseMatrixCOO(ducon_structure.m, ducon_structure.n,
    copy(ducon_structure.rows), copy(ducon_structure.cols), T.(ducon_nzvals_init))
  dlvar = SparseMatrixCOO(dlvar_structure.m, dlvar_structure.n,
    copy(dlvar_structure.rows), copy(dlvar_structure.cols), T.(dlvar_nzvals_init))
  duvar = SparseMatrixCOO(duvar_structure.m, duvar_structure.n,
    copy(duvar_structure.rows), copy(duvar_structure.cols), T.(duvar_nzvals_init))

  H_sym = S(undef, nnzh)
  for k in 1:nnzh
    H_sym[k] = Hrows[k] == Hcols[k] ? one(T) : T(2)
  end

  M1 = typeof(qp.data.H)
  M2 = typeof(qp.data.A)

  return ParametricQuadraticModel{T, S, M1, M2, DF, DJF}(
    qp, copy(θ), data_fn, data_jac_fn,
    dH, dA, dc, S(dc0_init),
    dlcon, ducon, dlvar, duvar,
    Vector{Int}(Hrows), Vector{Int}(Hcols), H_sym,
    Vector{Int}(Arows), Vector{Int}(Acols),
    S(undef, nnzh), S(undef, nnzj),
  )
end

# Helper to create empty SparseMatrixCOO
function _empty_coo(::Type{T}, m::Int, n::Int) where {T}
  SparseMatrixCOO(m, n, Int[], Int[], T[])
end

# ── Forward QP-type-specific methods to inner QuadraticModel ──

NLPModels.hess_structure!(nlp::ParametricQuadraticModel, rows, cols) =
  hess_structure!(nlp.qp, rows, cols)

NLPModels.hess_coord!(
  nlp::ParametricQuadraticModel{T},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T} = hess_coord!(nlp.qp, x, vals; obj_weight = obj_weight)

NLPModels.hess_coord!(
  nlp::ParametricQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hess_coord!(nlp.qp, x, y, vals; obj_weight = obj_weight)

NLPModels.jac_lin_structure!(nlp::ParametricQuadraticModel, rows, cols) =
  jac_lin_structure!(nlp.qp, rows, cols)

NLPModels.jac_lin_coord!(nlp::ParametricQuadraticModel, x, vals) =
  jac_lin_coord!(nlp.qp, x, vals)

# ── Parametric API ──

NLPModels.get_param_values(nlp::ParametricQuadraticModel) = nlp.θ

function NLPModels.set_param_values!(nlp::ParametricQuadraticModel, θ::AbstractVector)
  copyto!(nlp.θ, θ)

  H_nzvals, A_nzvals, c, c0, lcon, ucon, lvar, uvar = nlp.data_fn(θ)
  nlp.data.H.vals .= H_nzvals
  nlp.data.A.vals .= A_nzvals
  copyto!(nlp.data.c, c)
  nlp.data.c0 = c0
  copyto!(nlp.meta.lvar, lvar)
  copyto!(nlp.meta.uvar, uvar)
  copyto!(nlp.meta.lcon, lcon)
  copyto!(nlp.meta.ucon, ucon)

  jac_result = nlp.data_jac_fn(θ)
  dH_nz, dA_nz, dc_nz, dc0_v, dlcon_nz, ducon_nz, dlvar_nz, duvar_nz = jac_result
  _update_coo_vals!(nlp.dH, dH_nz)
  _update_coo_vals!(nlp.dA, dA_nz)
  _update_coo_vals!(nlp.dc, dc_nz)
  copyto!(nlp.dc0, dc0_v)
  _update_coo_vals!(nlp.dlcon, dlcon_nz)
  _update_coo_vals!(nlp.ducon, ducon_nz)
  _update_coo_vals!(nlp.dlvar, dlvar_nz)
  _update_coo_vals!(nlp.duvar, duvar_nz)

  return nlp
end

# Update COO vals in-place (nzvals only)
@inline function _update_coo_vals!(coo::SparseMatrixCOO, nzvals)
  if length(coo.vals) > 0
    copyto!(coo.vals, nzvals)
  end
end

# ∇_θ f = dc0 + dc'x + ½ dH' * u_h
# where u_h[k] = x[rows[k]] * x[cols[k]] * sym[k]
function NLPModels.grad_param!(
  nlp::ParametricQuadraticModel,
  x::AbstractVector,
  g::AbstractVector,
)
  T = eltype(g)
  for k in eachindex(nlp._u_h)
    nlp._u_h[k] = x[nlp.H_rows[k]] * x[nlp.H_cols[k]] * nlp.H_sym[k]
  end
  mul!(g, transpose(nlp.dc), x)
  g .+= nlp.dc0
  if length(nlp._u_h) > 0
    mul!(g, transpose(nlp.dH), nlp._u_h, one(T) / 2, one(T))
  end
  return g
end

# jac_param structure: dense ncon × nparam
function NLPModels.jac_param_structure!(
  nlp::ParametricQuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nparam = length(nlp.θ)
  count = 1
  for j in 1:nparam
    for i in 1:nlp.meta.ncon
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

# jac_param coord: J[i,j] = Σ_k dA[k,j] * x[A_cols[k]]  for A_rows[k] == i
function NLPModels.jac_param_coord!(
  nlp::ParametricQuadraticModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  ncon = nlp.meta.ncon
  nparam = length(nlp.θ)
  fill!(vals, zero(eltype(vals)))
  dA = nlp.dA
  for nz in 1:length(dA.vals)
    k = dA.rows[nz]  # row in dA = index into A nonzeros
    j = dA.cols[nz]  # col in dA = param index
    offset = (j - 1) * ncon
    vals[offset + nlp.A_rows[k]] += dA.vals[nz] * x[nlp.A_cols[k]]
  end
  return vals
end

# jpprod: Jv  where J = ∂cons/∂θ
function NLPModels.jpprod!(
  nlp::ParametricQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  mul!(nlp._u_a, nlp.dA, v)
  fill!(Jv, zero(eltype(Jv)))
  for k in eachindex(nlp.A_rows)
    Jv[nlp.A_rows[k]] += x[nlp.A_cols[k]] * nlp._u_a[k]
  end
  return Jv
end

# jptprod: J'v = dA' * (x[cols] .* v[rows])
function NLPModels.jptprod!(
  nlp::ParametricQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  for k in eachindex(nlp._u_a)
    nlp._u_a[k] = x[nlp.A_cols[k]] * v[nlp.A_rows[k]]
  end
  mul!(Jtv, transpose(nlp.dA), nlp._u_a)
  return Jtv
end

# hess_param structure: dense nvar × nparam
function NLPModels.hess_param_structure!(
  nlp::ParametricQuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nparam = length(nlp.θ)
  count = 1
  for j in 1:nparam
    for i in 1:nlp.meta.nvar
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

# hess_param coord: ∇²_{x,θ} L
# = obj_weight * (dc + ∂H/∂θ · x) + (∂A/∂θ)' · y
function NLPModels.hess_param_coord!(
  nlp::ParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  nvar = nlp.meta.nvar
  nparam = length(nlp.θ)
  ow = T(obj_weight)
  fill!(vals, zero(T))

  # dc contribution (sparse)
  dc = nlp.dc
  for nz in 1:length(dc.vals)
    i, j = dc.rows[nz], dc.cols[nz]
    offset = (j - 1) * nvar
    vals[offset + i] += ow * dc.vals[nz]
  end

  # H contribution: ow * (∂H/∂θ_j) x  (symmetric matvec)
  dH = nlp.dH
  for nz in 1:length(dH.vals)
    k = dH.rows[nz]  # index into H nonzeros
    j = dH.cols[nz]  # param index
    offset = (j - 1) * nvar
    row, col = nlp.H_rows[k], nlp.H_cols[k]
    dh = dH.vals[nz]
    vals[offset + row] += ow * dh * x[col]
    if row != col
      vals[offset + col] += ow * dh * x[row]
    end
  end

  # A contribution: (∂A/∂θ_j)' y
  dA = nlp.dA
  for nz in 1:length(dA.vals)
    k = dA.rows[nz]  # index into A nonzeros
    j = dA.cols[nz]  # param index
    offset = (j - 1) * nvar
    vals[offset + nlp.A_cols[k]] += dA.vals[nz] * y[nlp.A_rows[k]]
  end
  return vals
end

# hpprod: (∇²_{x,θ} L) v
function NLPModels.hpprod!(
  nlp::ParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  ow = T(obj_weight)
  mul!(Hv, nlp.dc, v)
  Hv .*= ow

  if length(nlp._u_h) > 0
    mul!(nlp._u_h, nlp.dH, v)
    for k in eachindex(nlp.H_rows)
      row, col = nlp.H_rows[k], nlp.H_cols[k]
      Hv[row] += ow * nlp._u_h[k] * x[col]
      if row != col
        Hv[col] += ow * nlp._u_h[k] * x[row]
      end
    end
  end

  if length(nlp._u_a) > 0
    mul!(nlp._u_a, nlp.dA, v)
    for k in eachindex(nlp.A_rows)
      Hv[nlp.A_cols[k]] += nlp._u_a[k] * y[nlp.A_rows[k]]
    end
  end

  return Hv
end

# hptprod: (∇²_{x,θ} L)' v
function NLPModels.hptprod!(
  nlp::ParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Htv::AbstractVector;
  obj_weight = one(T),
) where {T}
  ow = T(obj_weight)
  mul!(Htv, transpose(nlp.dc), v)
  Htv .*= ow

  if length(nlp._u_h) > 0
    for k in eachindex(nlp.H_rows)
      row, col = nlp.H_rows[k], nlp.H_cols[k]
      nlp._u_h[k] = x[row] * v[col]
      if row != col
        nlp._u_h[k] += x[col] * v[row]
      end
    end
    mul!(Htv, transpose(nlp.dH), nlp._u_h, ow, one(T))
  end

  if length(nlp._u_a) > 0
    for k in eachindex(nlp.A_rows)
      nlp._u_a[k] = y[nlp.A_rows[k]] * v[nlp.A_cols[k]]
    end
    mul!(Htv, transpose(nlp.dA), nlp._u_a, one(T), one(T))
  end

  return Htv
end

# ── Bounds parametric API ──

# lcon
function NLPModels.lcon_jac_param_structure!(nlp::ParametricQuadraticModel, rows, cols)
  copyto!(rows, nlp.dlcon.rows)
  copyto!(cols, nlp.dlcon.cols)
  return rows, cols
end

function NLPModels.lcon_jac_param_coord!(nlp::ParametricQuadraticModel, vals)
  copyto!(vals, nlp.dlcon.vals)
  return vals
end

function NLPModels.lcon_jpprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, nlp.dlcon, v)
  return Jv
end

function NLPModels.lcon_jptprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jtv::AbstractVector)
  mul!(Jtv, transpose(nlp.dlcon), v)
  return Jtv
end

# ucon
function NLPModels.ucon_jac_param_structure!(nlp::ParametricQuadraticModel, rows, cols)
  copyto!(rows, nlp.ducon.rows)
  copyto!(cols, nlp.ducon.cols)
  return rows, cols
end

function NLPModels.ucon_jac_param_coord!(nlp::ParametricQuadraticModel, vals)
  copyto!(vals, nlp.ducon.vals)
  return vals
end

function NLPModels.ucon_jpprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, nlp.ducon, v)
  return Jv
end

function NLPModels.ucon_jptprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jtv::AbstractVector)
  mul!(Jtv, transpose(nlp.ducon), v)
  return Jtv
end

# lvar
function NLPModels.lvar_jac_param_structure!(nlp::ParametricQuadraticModel, rows, cols)
  copyto!(rows, nlp.dlvar.rows)
  copyto!(cols, nlp.dlvar.cols)
  return rows, cols
end

function NLPModels.lvar_jac_param_coord!(nlp::ParametricQuadraticModel, vals)
  copyto!(vals, nlp.dlvar.vals)
  return vals
end

function NLPModels.lvar_jpprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, nlp.dlvar, v)
  return Jv
end

function NLPModels.lvar_jptprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jtv::AbstractVector)
  mul!(Jtv, transpose(nlp.dlvar), v)
  return Jtv
end

# uvar
function NLPModels.uvar_jac_param_structure!(nlp::ParametricQuadraticModel, rows, cols)
  copyto!(rows, nlp.duvar.rows)
  copyto!(cols, nlp.duvar.cols)
  return rows, cols
end

function NLPModels.uvar_jac_param_coord!(nlp::ParametricQuadraticModel, vals)
  copyto!(vals, nlp.duvar.vals)
  return vals
end

function NLPModels.uvar_jpprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, nlp.duvar, v)
  return Jv
end

function NLPModels.uvar_jptprod!(nlp::ParametricQuadraticModel, v::AbstractVector, Jtv::AbstractVector)
  mul!(Jtv, transpose(nlp.duvar), v)
  return Jtv
end
