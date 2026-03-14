# Helpers for parametric matrix structure/values (work for COO, CSC, dense)
_param_nzvals(M::SparseMatrixCOO) = M.vals
_param_nzvals(M::SparseMatrixCSC) = nonzeros(M)
_param_nzvals(M::AbstractMatrix) = vec(M)

_param_fill_structure!(M::SparseMatrixCOO, rows, cols) = fill_structure!(M, rows, cols)
_param_fill_structure!(M::SparseMatrixCSC, rows, cols) = fill_structure!(M, rows, cols)
function _param_fill_structure!(M::AbstractMatrix, rows, cols)
  count = 1
  for j in 1:size(M, 2), i in 1:size(M, 1)
    rows[count] = i
    cols[count] = j
    count += 1
  end
end

"""
    LinearParametricQuadraticModel

A parametric quadratic program:

    min  ½xᵀHx + (c + Fθ)ᵀx + c0
    s.t. lcon ≤ Ax + Bθ ≤ ucon
         lvar ≤ x ≤ uvar

Wraps an inner `QuadraticModel` whose linear cost `c` is maintained as
`c_base + Fθ`.  Constraints evaluate to `Ax + Bθ` (Bθ cached in a buffer).
Implements the NLPModels parametric API.
"""
mutable struct LinearParametricQuadraticModel{T, S, M1, M2, MF, MB} <: AbstractQuadraticModel{T, S}
  qp::QuadraticModel{T, S, M1, M2}
  c_base::S       # nvar
  F::MF           # nvar × nparam
  B::MB           # ncon × nparam
  θ::S            # nparam
  Bθ::S           # ncon — cached B*θ
end

# Forward .meta, .counters, .data to the inner QuadraticModel
function Base.getproperty(nlp::LinearParametricQuadraticModel, s::Symbol)
  if s in (:qp, :c_base, :F, :B, :θ, :Bθ)
    return getfield(nlp, s)
  else
    return getproperty(getfield(nlp, :qp), s)
  end
end

function LinearParametricQuadraticModel(
  c::S, H, A, F::MF, B::MB, θ::S;
  c0 = zero(eltype(c)),
  lcon = fill!(S(undef, size(A, 1)), eltype(c)(-Inf)),
  ucon = fill!(S(undef, size(A, 1)), eltype(c)(Inf)),
  lvar = fill!(S(undef, length(c)), eltype(c)(-Inf)),
  uvar = fill!(S(undef, length(c)), eltype(c)(Inf)),
  name = "LinearParametricQP",
) where {S, MF, MB}
  T = eltype(c)
  nvar = length(c)
  ncon = size(A, 1)
  nparam = length(θ)

  @assert size(A, 2) == nvar
  @assert size(F) == (nvar, nparam)
  @assert size(B) == (ncon, nparam)

  c_eff = c + F * θ
  Bθ = B * θ

  has_param = nparam > 0
  has_con = ncon > 0

  qp = QuadraticModel(
    c_eff, H;
    A = A, lcon = lcon, ucon = ucon,
    lvar = lvar, uvar = uvar, c0 = c0, name = name,
    nparam = nparam,
    nnzgp = nparam,
    nnzjp = nnz(B),
    nnzhp = nnz(F),
    grad_param_available = has_param,
    jac_param_available = has_param && has_con,
    hess_param_available = has_param,
    jpprod_available = has_param && has_con,
    jptprod_available = has_param && has_con,
    hpprod_available = has_param,
    hptprod_available = has_param,
  )

  M1 = typeof(qp.data.H)
  M2t = typeof(qp.data.A)
  return LinearParametricQuadraticModel{T, S, M1, M2t, MF, MB}(
    qp, copy(c), F, B, copy(θ), Bθ,
  )
end

# ── Forward QP-type-specific methods to inner QuadraticModel ──

NLPModels.hess_structure!(nlp::LinearParametricQuadraticModel, rows, cols) =
  hess_structure!(nlp.qp, rows, cols)

NLPModels.hess_coord!(
  nlp::LinearParametricQuadraticModel{T},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T} = hess_coord!(nlp.qp, x, vals; obj_weight = obj_weight)

NLPModels.hess_coord!(
  nlp::LinearParametricQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hess_coord!(nlp.qp, x, y, vals; obj_weight = obj_weight)

NLPModels.jac_lin_structure!(nlp::LinearParametricQuadraticModel, rows, cols) =
  jac_lin_structure!(nlp.qp, rows, cols)

NLPModels.jac_lin_coord!(nlp::LinearParametricQuadraticModel, x, vals) =
  jac_lin_coord!(nlp.qp, x, vals)

# ── Override cons to add Bθ term ──

function NLPModels.cons_lin!(nlp::LinearParametricQuadraticModel, x::AbstractVector, c::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin c
  NLPModels.increment!(nlp, :neval_cons_lin)
  mul!(c, nlp.data.A, x)
  c .+= nlp.Bθ
  return c
end

# ── Parametric API ──

NLPModels.get_param_values(nlp::LinearParametricQuadraticModel) = nlp.θ

function NLPModels.set_param_values!(nlp::LinearParametricQuadraticModel, θ::AbstractVector)
  copyto!(nlp.θ, θ)
  # data.c = c_base + F*θ
  mul!(nlp.data.c, nlp.F, θ)
  nlp.data.c .+= nlp.c_base
  # Bθ cache
  mul!(nlp.Bθ, nlp.B, θ)
  return nlp
end

# ∇_θ f = F'x
function NLPModels.grad_param!(
  nlp::LinearParametricQuadraticModel,
  x::AbstractVector,
  g::AbstractVector,
)
  mul!(g, transpose(nlp.F), x)
  return g
end

# ∂c/∂θ = B  (constraint Jacobian wrt parameters)
function NLPModels.jac_param_structure!(
  nlp::LinearParametricQuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  _param_fill_structure!(nlp.B, rows, cols)
  return rows, cols
end

function NLPModels.jac_param_coord!(
  nlp::LinearParametricQuadraticModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  vals .= _param_nzvals(nlp.B)
  return vals
end

function NLPModels.jpprod!(
  nlp::LinearParametricQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  mul!(Jv, nlp.B, v)
  return Jv
end

function NLPModels.jptprod!(
  nlp::LinearParametricQuadraticModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  mul!(Jtv, transpose(nlp.B), v)
  return Jtv
end

# ∇²_{x,θ} L = obj_weight * F  (no x·θ cross terms in constraints)
function NLPModels.hess_param_structure!(
  nlp::LinearParametricQuadraticModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  _param_fill_structure!(nlp.F, rows, cols)
  return rows, cols
end

function NLPModels.hess_param_coord!(
  nlp::LinearParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  vals .= T(obj_weight) .* _param_nzvals(nlp.F)
  return vals
end

function NLPModels.hpprod!(
  nlp::LinearParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  mul!(Hv, nlp.F, v)
  if obj_weight != one(T)
    Hv .*= T(obj_weight)
  end
  return Hv
end

function NLPModels.hptprod!(
  nlp::LinearParametricQuadraticModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Htv::AbstractVector;
  obj_weight = one(T),
) where {T}
  mul!(Htv, transpose(nlp.F), v)
  if obj_weight != one(T)
    Htv .*= T(obj_weight)
  end
  return Htv
end
