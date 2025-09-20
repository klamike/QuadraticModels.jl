"""
    PQPData{T, S, M1, M2, M3, M4, M5}

Data structure for parametric quadratic programming problems.

The parametric quadratic program has the form:
    min  (1/2) x' H x + c' x + (F θ)' x + c₀
    s.t. lcon ≤ A x + B θ ≤ ucon
         lvar ≤ x ≤ uvar
         lparam ≤ P θ ≤ uparam

where θ is the parameter vector.

It follows that the dual problem has the form:
    max  -(1/2) w' H w + yₗ'lcon - yᵤ'ucon + zₗ'lvar - zᵤ'uvar + (Bθ)'(yₗ - yᵤ) + c₀
    s.t. A'(yₗ - yᵤ) + zₗ - zᵤ = Hw + c + Fθ
         yₗ, yᵤ, zₗ, zᵤ ≥ 0
         lparam ≤ P θ ≤ uparam

# Fields
- `c0::T`: constant term in objective
- `c::S`: linear term in objective (coefficient of x)
- `F::M1`: parameter coefficient matrix in objective (F θ)' x
- `H::M2`: Hessian matrix (quadratic term)
- `A::M3`: constraint matrix for x
- `B::M4`: constraint matrix for θ
- `P::M5`: parameter constraint matrix
- `lparam::S`: lower bounds for parameter constraints
- `uparam::S`: upper bounds for parameter constraints
"""
mutable struct PQPData{
  T,
  S,
  M1 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M2 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M3 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M4 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M5 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
}
  c0::T         # constant term in objective
  c::S          # linear term in objective
  F::M1         # parameter coefficient matrix in objective
  H::M2         # Hessian matrix
  A::M3         # constraint matrix for x
  B::M4         # constraint matrix for θ
  P::M5         # parameter constraint matrix  # FIXME: remove?
  lparam::S     # lower bounds for parameter constraints
  uparam::S     # upper bounds for parameter constraints
  θ::S          # current parameter value # FIXME: move to model level?
  v::S          # workspace vector 1 (nvar)
  vf::S         # workspace vector 2 (nvar)
end

@inline PQPData(c0, c, F, H, A, B, P, lparam, uparam, θ) =
  PQPData(c0, c, F, H, A, B, P, lparam, uparam, θ, similar(c), similar(c))
isdense(data::PQPData{T, S, M1, M2, M3, M4, M5}) where {T, S, M1, M2, M3, M4, M5} =
  M1 <: DenseMatrix ||
  M2 <: DenseMatrix ||
  M3 <: DenseMatrix ||
  M4 <: DenseMatrix ||
  M5 <: DenseMatrix

# TODO: convert helper
function Base.convert(
  ::Type{PQPData{T, S, MCOO, MCOO, MCOO, MCOO, MCOO}},
  data::PQPData{T, S, M1, M2, M3, M4, M5},
) where {
  T,
  S,
  M1 <: AbstractMatrix,
  M2 <: AbstractMatrix,
  M3 <: AbstractMatrix,
  M4 <: AbstractMatrix,
  M5 <: AbstractMatrix,
  MCOO <: SparseMatrixCOO{T},
}
  HCOO = (M1 <: SparseMatrixCOO) ? data.H : SparseMatrixCOO(data.H)
  ACOO = (M2 <: SparseMatrixCOO) ? data.A : SparseMatrixCOO(data.A)
  BCOO = (M3 <: SparseMatrixCOO) ? data.B : SparseMatrixCOO(data.B)
  PCOO = (M4 <: SparseMatrixCOO) ? data.P : SparseMatrixCOO(data.P)
  return PQPData(data.c0, data.c, data.F, HCOO, ACOO, BCOO, PCOO, data.lparam, data.uparam, data.θ)
end
Base.convert(
  ::Type{PQPData{T, S, MCOO, MCOO, MCOO, MCOO, MCOO}},
  data::PQPData{T, S, M1, M2, M3, M4, M5},
) where {
  T,
  S,
  M1 <: SparseMatrixCOO,
  M2 <: SparseMatrixCOO,
  M3 <: SparseMatrixCOO,
  M4 <: SparseMatrixCOO,
  M5 <: SparseMatrixCOO,
  MCOO <: SparseMatrixCOO{T},
} = data

abstract type AbstractParametricQuadraticModel{T, S} <: AbstractNLPModel{T, S} end

"""
    ParametricQuadraticModel{T, S, M1, M2, M3, M4, M5}

Parametric quadratic model implementing the NLPModels interface.

The parametric quadratic program has the form:
    min  (1/2) x' H x + c' x + (F θ)' x + c₀
    s.t. lcon ≤ A x + B θ ≤ ucon
         lvar ≤ x ≤ uvar
         lparam ≤ P θ ≤ uparam

where θ is the parameter vector.
"""
mutable struct ParametricQuadraticModel{T, S, M1, M2, M3, M4, M5} <:
               AbstractParametricQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::PQPData{T, S, M1, M2, M3, M4, M5}
end

function Base.convert(
  ::Type{ParametricQuadraticModel{T, S, Mconv, Mconv, Mconv, Mconv, Mconv}},
  qm::ParametricQuadraticModel{T, S, M1, M2, M3, M4, M5},
) where {
  T,
  S,
  M1 <: AbstractMatrix,
  M2 <: AbstractMatrix,
  M3 <: AbstractMatrix,
  M4 <: AbstractMatrix,
  M5 <: AbstractMatrix,
  Mconv,
}
  data_conv = convert(PQPData{T, S, Mconv, Mconv, Mconv, Mconv, Mconv}, qm.data)
  return ParametricQuadraticModel(qm.meta, qm.counters, data_conv)
end

# TODO: sparse constructor

# Constructor for ParametricQuadraticModel
function ParametricQuadraticModel(
  c::S,
  F::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  H::Union{AbstractMatrix{T}, AbstractLinearOperator{T}};
  θ::S = fill!(S(undef, size(F, 2)), T(0)),
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = similar_empty_matrix(H, length(c)),
  B::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = similar_empty_matrix(H, length(θ)),
  P::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = similar_empty_matrix(H, length(θ)),
  lcon::S = S(undef, 0),
  ucon::S = S(undef, 0),
  lvar::S = fill!(S(undef, length(c)), T(-Inf)),
  uvar::S = fill!(S(undef, length(c)), T(Inf)),
  lparam::S = fill!(S(undef, size(P, 1)), T(-Inf)),
  uparam::S = fill!(S(undef, size(P, 1)), T(Inf)),
  c0::T = zero(T),
  name::String = "ParametricQuadraticModel",
) where {T, S}
  @assert all(lvar .≤ uvar)
  @assert all(lcon .≤ ucon)
  @assert all(lparam .≤ uparam)

  ncon, nvar = size(A)

  if typeof(H) <: AbstractLinearOperator # convert A to a LinOp if A is a Matrix?
    nnzh = 0
    nnzj = 0
    data = PQPData(c0, c, F, H, A, B, P, lparam, uparam, θ)
  else
    nnzh = typeof(H) <: DenseMatrix ? nvar * (nvar + 1) / 2 : nnz(H)
    nnzj = nnz(A)
    data =
      typeof(H) <: Symmetric ? PQPData(c0, c, F, H.data, A, B, P, lparam, uparam, θ) :
      PQPData(c0, c, F, H, A, B, P, lparam, uparam, θ)
  end

  return ParametricQuadraticModel(
    NLPModelMeta{T, typeof(c)}(
      nvar,
      lvar = lvar,
      uvar = uvar,
      ncon = ncon,
      lcon = lcon,
      ucon = ucon,
      nnzj = nnzj,
      lin_nnzj = nnzj,
      nln_nnzj = 0,
      nnzh = nnzh,
      lin = collect(1:ncon),  # All constraints are linear
      islp = (nnzh == 0),
      name = name,
    ),
    Counters(),
    data,
  )
end

"""
    set_parameter!(pqp::AbstractParametricQuadraticModel, θ::AbstractVector; check_feasibility::Bool = true)

Sets the current parameter value θ in the parametric quadratic model.

# Arguments
- `pqp`: The parametric quadratic model
- `θ`: The parameter vector
- `check_feasibility`: If true (default), checks if θ satisfies the parameter constraints lparam ≤ Pθ ≤ uparam

# Throws
- `ArgumentError`: If `check_feasibility=true` and θ does not satisfy the parameter constraints
"""
function set_parameter!(
  pqp::AbstractParametricQuadraticModel{T, S},
  θ::AbstractVector;
  check_feasibility::Bool = true,
) where {T, S}
  # Check parameter feasibility if requested
  if check_feasibility && length(θ) > 0 && size(pqp.data.P, 1) > 0 && size(pqp.data.P, 2) > 0
    Pθ = similar(pqp.data.lparam)
    mul!(Pθ, pqp.data.P, θ)

    if !all(pqp.data.lparam .≤ Pθ .≤ pqp.data.uparam)
      violated_lower = findall(Pθ .< pqp.data.lparam)
      violated_upper = findall(Pθ .> pqp.data.uparam)

      error_msg = "Parameter vector θ does not satisfy parameter constraints lparam ≤ Pθ ≤ uparam.\n"
      if !isempty(violated_lower)
        error_msg *= "Violated lower bounds at constraints: $violated_lower\n"
        error_msg *= "Values: $(Pθ[violated_lower]), Lower bounds: $(pqp.data.lparam[violated_lower])\n"
      end
      if !isempty(violated_upper)
        error_msg *= "Violated upper bounds at constraints: $violated_upper\n"
        error_msg *= "Values: $(Pθ[violated_upper]), Upper bounds: $(pqp.data.uparam[violated_upper])\n"
      end
      throw(ArgumentError(error_msg))
    end
  end

  # Set the parameter value
  copy!(pqp.data.θ, θ)
  return pqp
end

"""
    evaluate_at_parameter(pqp::AbstractParametricQuadraticModel, θ::AbstractVector; check_feasibility::Bool = true)

Returns a QuadraticModel instance with the parameter θ fixed.

# Arguments
- `pqp`: The parametric quadratic model
- `θ`: The parameter vector
- `check_feasibility`: If true (default), checks if θ satisfies the parameter constraints lparam ≤ Pθ ≤ uparam

# Throws
- `ArgumentError`: If `check_feasibility=true` and θ does not satisfy the parameter constraints
"""
function evaluate_at_parameter(
  pqp::AbstractParametricQuadraticModel{T, S},
  θ::AbstractVector;
) where {T, S}
  # Compute the effective linear term: c + Fθ
  c_eff = copy(linobj(pqp, θ))

  # Compute the effective constraint bounds: lcon - B*θ, ucon - B*θ  # FIXME: make this two functions
  lcon_eff = similar(pqp.meta.lcon)
  ucon_eff = similar(pqp.meta.ucon)
  copy!(lcon_eff, pqp.meta.lcon)
  copy!(ucon_eff, pqp.meta.ucon)

  if length(θ) > 0 && size(pqp.data.B, 1) > 0 && size(pqp.data.B, 2) > 0
    Bθ = similar(lcon_eff)
    mul!(Bθ, pqp.data.B, θ)
    lcon_eff .-= Bθ
    ucon_eff .-= Bθ
  end

  # Create and return a QuadraticModel
  return QuadraticModel(
    c_eff,
    pqp.data.H,
    A = pqp.data.A,
    lcon = lcon_eff,
    ucon = ucon_eff,
    lvar = pqp.meta.lvar,
    uvar = pqp.meta.uvar,
    c0 = pqp.data.c0,
  )
end

@inline linobj(pqp::AbstractParametricQuadraticModel, θ) = begin
  copy!(pqp.data.vf, pqp.data.c)
  mul!(pqp.data.vf, pqp.data.F, θ, 1, 1)  # c_eff = c + Fθ
  return pqp.data.vf
end

function NLPModels.objgrad!(
  pqp::ParametricQuadraticModel{T, S},
  x::AbstractVector,
  g::AbstractVector,
) where {T, S}
  NLPModels.increment!(pqp, :neval_obj)
  NLPModels.increment!(pqp, :neval_grad)

  # v ← Hx
  mul!(pqp.data.v, Symmetric(pqp.data.H, :L), x)

  # g ← Hx + (c+Fθ)
  g = pqp.data.v .+ linobj(pqp, pqp.data.θ) # sets pqp.data.vf to c+Fθ

  # f ← c0 + (c+Fθ)'x + (Hx)'x / 2
  f = pqp.data.c0 + dot(pqp.data.vf, x) + dot(pqp.data.v, x) / 2

  return f, g
end

function NLPModels.obj(pqp::AbstractParametricQuadraticModel{T, S}, x::AbstractVector) where {T, S}
  NLPModels.increment!(pqp, :neval_obj)

  # v ← H*x
  mul!(pqp.data.v, Symmetric(pqp.data.H, :L), x)

  # vf ← (c+Fθ)
  linobj(pqp, pqp.data.θ)

  # c0 + (c+Fθ)'x + (Hx)'x / 2
  return pqp.data.c0 + dot(pqp.data.vf, x) + dot(pqp.data.v, x) / 2
end

function NLPModels.grad!(
  pqp::AbstractParametricQuadraticModel,
  x::AbstractVector,
  g::AbstractVector,
)
  NLPModels.increment!(pqp, :neval_grad)
  # g ← H*x
  mul!(g, Symmetric(pqp.data.H, :L), x)
  # vf ← (c+Fθ)
  copy!(pqp.data.vf, pqp.data.c)
  mul!(pqp.data.vf, pqp.data.F, pqp.data.θ, 1, 1)
  # g ← H*x + (c+Fθ)
  g .+= pqp.data.vf
  return g
end

function NLPModels.cons_lin!(
  pqp::ParametricQuadraticModel{T, S},
  x::AbstractVector,
  c::AbstractVector,
) where {T, S}
  @lencheck pqp.meta.nvar x
  @lencheck pqp.meta.nlin c
  NLPModels.increment!(pqp, :neval_cons_lin)

  # c ← Ax
  mul!(c, pqp.data.A, x)

  # c ← Bθ + Ax
  mul!(c, pqp.data.B, pqp.data.θ, 1, 1)
  return c
end

# begin same as QuadraticModel

function NLPModels.hess_structure!(
  pqp::ParametricQuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCOO}
  rows .= pqp.data.H.rows
  cols .= pqp.data.H.cols
  return rows, cols
end

function NLPModels.hess_structure!(
  pqp::ParametricQuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: SparseMatrixCSC}
  fill_structure!(pqp.data.H, rows, cols)
  return rows, cols
end

function NLPModels.hess_structure!(
  pqp::ParametricQuadraticModel{T, S, M1},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1 <: Matrix}
  count = 1
  for j = 1:(pqp.meta.nvar)
    for i = j:(pqp.meta.nvar)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  pqp::ParametricQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: SparseMatrixCOO}
  NLPModels.increment!(pqp, :neval_hess)
  vals .= obj_weight .* pqp.data.H.vals
  return vals
end

function NLPModels.hess_coord!(
  pqp::ParametricQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: SparseMatrixCSC}
  NLPModels.increment!(pqp, :neval_hess)
  fill_coord!(pqp.data.H, vals, obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  pqp::ParametricQuadraticModel{T, S, M1},
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: Matrix}
  NLPModels.increment!(pqp, :neval_hess)
  count = 1
  for j = 1:(pqp.meta.nvar)
    for i = j:(pqp.meta.nvar)
      vals[count] = obj_weight * pqp.data.H[i, j]
      count += 1
    end
  end
  return vals
end

NLPModels.hess_coord!(
  pqp::ParametricQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hess_coord!(pqp, x, vals, obj_weight = obj_weight)

function NLPModels.jac_lin_structure!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCOO}
  @lencheck pqp.meta.lin_nnzj rows cols
  rows .= pqp.data.A.rows
  cols .= pqp.data.A.cols
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: SparseMatrixCSC}
  @lencheck pqp.meta.lin_nnzj rows cols
  fill_structure!(pqp.data.A, rows, cols)
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: Matrix}
  @lencheck pqp.meta.lin_nnzj rows cols
  count = 1
  for j = 1:(pqp.meta.nvar)
    for i = 1:(pqp.meta.ncon)
      rows[count] = i
      cols[count] = j
      count += 1
    end
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: SparseMatrixCOO}
  @lencheck pqp.meta.nvar x
  @lencheck pqp.meta.lin_nnzj vals
  NLPModels.increment!(pqp, :neval_jac_lin)
  vals .= pqp.data.A.vals
  return vals
end

function NLPModels.jac_lin_coord!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: SparseMatrixCSC}
  @lencheck pqp.meta.nvar x
  @lencheck pqp.meta.lin_nnzj vals
  NLPModels.increment!(pqp, :neval_jac_lin)
  fill_coord!(pqp.data.A, vals, one(T))
  return vals
end

function NLPModels.jac_lin_coord!(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S, M1, M2 <: Matrix}
  @lencheck pqp.meta.nvar x
  @lencheck pqp.meta.lin_nnzj vals
  NLPModels.increment!(pqp, :neval_jac_lin)
  count = 1
  for j = 1:(pqp.meta.nvar)
    for i = 1:(pqp.meta.ncon)
      vals[count] = pqp.data.A[i, j]
      count += 1
    end
  end
  return vals
end

function NLPModels.jac_lin(
  pqp::ParametricQuadraticModel{T, S, M1, M2},
  x::AbstractVector,
) where {T, S, M1 <: AbstractLinearOperator, M2 <: AbstractLinearOperator}
  @lencheck pqp.meta.nvar x
  increment!(pqp, :neval_jac_lin)
  return pqp.data.A
end

## below are not needed since PQM <: AbstractQM
# function NLPModels.hprod!(
#   pqp::ParametricQuadraticModel{T, S},
#   x::AbstractVector,
#   v::AbstractVector,
#   Hv::AbstractVector;
#   obj_weight::Real = one(eltype(x)),
# )
#   NLPModels.increment!(pqp, :neval_hprod)
#   mul!(Hv, Symmetric(pqp.data.H, :L), v)
#   if obj_weight != 1
#     Hv .*= obj_weight
#   end
#   return Hv
# end

# NLPModels.hprod!(
#   pqp::ParametricQuadraticModel{T, S},
#   x::AbstractVector,
#   y::AbstractVector,
#   v::AbstractVector,
#   Hv::AbstractVector;
#   obj_weight::Real = one(eltype(x)),
# ) = hprod!(pqp, x, v, Hv, obj_weight = obj_weight)

# function NLPModels.jprod_lin!(
#   pqp::ParametricQuadraticModel{T, S},
#   x::AbstractVector,
#   v::AbstractVector,
#   Av::AbstractVector,
# )
#   @lencheck pqp.meta.nvar x v
#   @lencheck pqp.meta.nlin Av
#   NLPModels.increment!(pqp, :neval_jprod_lin)
#   mul!(Av, pqp.data.A, v)
#   return Av
# end

# function NLPModels.jtprod!(
#   pqp::ParametricQuadraticModel{T, S},
#   x::AbstractVector,
#   v::AbstractVector,
#   Atv::AbstractVector,
# )
#   @lencheck pqp.meta.nvar x Atv
#   @lencheck pqp.meta.ncon v
#   NLPModels.increment!(pqp, :neval_jtprod)
#   mul!(Atv, transpose(pqp.data.A), v)
#   return Atv
# end

# function NLPModels.jtprod_lin!(
#   pqp::ParametricQuadraticModel{T, S},
#   x::AbstractVector,
#   v::AbstractVector,
#   Atv::AbstractVector,
# )
#   @lencheck pqp.meta.nvar x Atv
#   @lencheck pqp.meta.nlin v
#   NLPModels.increment!(pqp, :neval_jtprod_lin)
#   mul!(Atv, transpose(pqp.data.A), v)
#   return Atv
# end

# end same as QuadraticModel