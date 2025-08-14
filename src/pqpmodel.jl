export ParametricQuadraticModel, ParametricQPData


# TODO: allow Dθ ≤ d?
"""
    ParametricQPData{T, S, M1, M2, M3, M4}

Data structure for parametric quadratic optimization problems.

The problem is:
    min_{x}  (1/2)x^T H x + (f + F θ)^T x
    s.t.     lcon ≤ A x + B θ ≤ ucon
             ltheta ≤ θ ≤ utheta

Fields:
- `c0::T`: constant term in objective
- `f::S`: linear term in objective (independent of θ)
- `F::M1`: linear term in objective (dependent on θ)
- `H::M2`: Hessian matrix (must be positive definite)
- `A::M3`: constraint matrix
- `B::M4`: constraint matrix for parameters
- `lcon::S`: lower bounds on constraints
- `ucon::S`: upper bounds on constraints
- `ltheta::S`: lower bounds on parameters θ
- `utheta::S`: upper bounds on parameters θ
- `v::S`: workspace vector 1 (stores Hx)
- `vf::S`: workspace vector 2 (stores f+Fθ)
"""
mutable struct ParametricQPData{
  T,
  S,
  M1 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M2 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M3 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  M4 <: Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
}
  c0::T         # constant term in objective
  f::S          # linear term in objective (independent of θ)
  F::M1         # linear term in objective (dependent on θ)
  H::M2         # Hessian matrix
  A::M3         # constraint matrix
  B::M4         # constraint matrix for parameters
  lcon::S       # lower bounds on constraints
  ucon::S       # upper bounds on constraints
  ltheta::S     # lower bounds on parameters θ
  utheta::S     # upper bounds on parameters θ
  v::S          # workspace vector 1 (nvar)
  vf::S         # workspace vector 2 (nvar)
end

@inline ParametricQPData(c0, f, F, H, A, B, lcon, ucon, ltheta, utheta) =
  ParametricQPData(c0, f, F, H, A, B, lcon, ucon, ltheta, utheta, similar(f), similar(f))

isdense(data::ParametricQPData{T, S, M1, M2, M3, M4}) where {T, S, M1, M2, M3, M4} =
  M1 <: DenseMatrix || M2 <: DenseMatrix || M3 <: DenseMatrix || M4 <: DenseMatrix

function Base.convert(
  ::Type{ParametricQPData{T, S, MCOO, MCOO, MCOO, MCOO}},
  data::ParametricQPData{T, S, M1, M2, M3, M4},
) where {T, S, M1 <: AbstractMatrix, M2 <: AbstractMatrix, M3 <: AbstractMatrix, M4 <: AbstractMatrix, MCOO <: SparseMatrixCOO{T}}
  HCOO = (M1 <: SparseMatrixCOO) ? data.H : SparseMatrixCOO(data.H)
  ACOO = (M2 <: SparseMatrixCOO) ? data.A : SparseMatrixCOO(data.A)
  BCOO = (M3 <: SparseMatrixCOO) ? data.B : SparseMatrixCOO(data.B)
  return ParametricQPData(data.c0, data.f, data.F, HCOO, ACOO, BCOO, data.lcon, data.ucon, data.ltheta, data.utheta, data.v, data.vf)
end
Base.convert(
  ::Type{ParametricQPData{T, S, MCOO, MCOO, MCOO, MCOO}},
  data::ParametricQPData{T, S, M1, M2, M3, M4},
) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO, M3 <: SparseMatrixCOO, M4 <: SparseMatrixCOO, MCOO <: SparseMatrixCOO{T}} = data
  

# FIXME: parametric abstract type?
"""
    ParametricQuadraticModel{T, S, M1, M2, M3, M4}

A parametric quadratic optimization model representing:

    min_{x}  (1/2)x^T H x + (f + F θ)^T x
    s.t.     lcon ≤ A x + B θ ≤ ucon
             ltheta ≤ θ ≤ utheta
"""
mutable struct ParametricQuadraticModel{T, S, M1, M2, M3, M4} <: AbstractQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::ParametricQPData{T, S, M1, M2, M3, M4}
  n_theta::Int  # number of parameters
end

"""
    ParametricQuadraticModel(f, F, H, A, B, lcon, ucon, ltheta, utheta; 
                            lvar = lvar, uvar = uvar, c0 = zero(T))

Create a ParametricQuadraticModel with the given data.

Arguments:
- `f`: linear term in objective (independent of θ)
- `F`: linear term in objective (dependent on θ)
- `H`: Hessian matrix (must be positive definite)
- `A`: constraint matrix
- `B`: constraint matrix for parameters
- `lcon`: lower bounds on constraints
- `ucon`: upper bounds on constraints
- `ltheta`: lower bounds on parameters θ
- `utheta`: upper bounds on parameters θ

Keyword arguments:
- `lvar`: lower bounds on decision variables x
- `uvar`: upper bounds on decision variables x
- `c0`: constant term in objective
"""
function ParametricQuadraticModel(
  f::S,
  F::M1,
  H::M2,
  A::M3,
  B::M4,
  lcon::S,
  ucon::S,
  ltheta::S,
  utheta::S;
  lvar::S = fill!(similar(f), eltype(f)(-Inf)),
  uvar::S = fill!(similar(f), eltype(f)(Inf)),
  c0::T = zero(eltype(f)),
  kwargs...,
) where {T, S, M1, M2, M3, M4}
  @assert all(lvar .≤ uvar)
  @assert all(ltheta .≤ utheta)
  @assert all(lcon .≤ ucon)
  @assert length(f) == length(lvar) == length(uvar)
  @assert size(H, 1) == size(H, 2) == length(f)
  @assert size(F, 1) == length(f)
  @assert size(A, 2) == length(f)
  @assert size(B, 2) == size(F, 2)
  @assert length(ltheta) == length(utheta) == size(F, 2)
  @assert size(A, 1) == size(B, 1) == length(lcon) == length(ucon)

  nvar = length(f)
  ncon = size(A, 1)
  n_theta = size(F, 2)

  # Determine sparsity patterns
  nnzh = typeof(H) <: DenseMatrix ? nvar * (nvar + 1) / 2 : nnz(H)
  nnzj = nnz(A) + nnz(B)

  data = ParametricQPData(c0, f, F, H, A, B, lcon, ucon, ltheta, utheta)

  return ParametricQuadraticModel(
    NLPModelMeta{T, S}(
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
      lin = 1:ncon,
      islp = false;
      kwargs...,
    ),
    Counters(),
    data,
    n_theta,
  )
end

function linobj(pqp::ParametricQuadraticModel, theta::AbstractVector, args...)
  copy!(pqp.data.vf, pqp.data.f)
  mul!(pqp.data.vf, pqp.data.F, theta, 1, 1)
  return pqp.data.vf
end

function objgrad!(
  pqp::ParametricQuadraticModel{T, S},
  x::AbstractVector,
  theta::AbstractVector,
  g::AbstractVector,
) where {T, S}
  NLPModels.increment!(pqp, :neval_obj)
  NLPModels.increment!(pqp, :neval_grad)
  
  # v ← H*x
  mul!(pqp.data.v, Symmetric(pqp.data.H, :L), x)
  
  # vf ← (f+F*θ)
  copy!(pqp.data.vf, pqp.data.f)
  mul!(pqp.data.vf, pqp.data.F, theta, 1, 1)

  # g ← H*x + (f+F*θ)
  g .= pqp.data.v .+ pqp.data.vf

  # f ← c0 + (f+F*θ)'x + (H*x)'x / 2
  f = pqp.data.c0 + dot(pqp.data.vf, x) + dot(pqp.data.v, x) / 2

  return f, g
end

function NLPModels.obj(
  pqp::ParametricQuadraticModel{T, S},
  x::AbstractVector,
  theta::AbstractVector,
) where {T, S}
  NLPModels.increment!(pqp, :neval_obj)
  
  # v ← H*x
  mul!(pqp.data.v, Symmetric(pqp.data.H, :L), x)
  
  # vf ← (f+F*θ)
  copy!(pqp.data.vf, pqp.data.f)
  mul!(pqp.data.vf, pqp.data.F, theta, 1, 1)
  
  # c0 + (f+F*θ)'x + (H*x)'x / 2
  return pqp.data.c0 + dot(pqp.data.vf, x) + dot(pqp.data.v, x) / 2
end

function NLPModels.grad!(
  pqp::ParametricQuadraticModel{T, S},
  x::AbstractVector,
  theta::AbstractVector,
  g::AbstractVector,
) where {T, S}
  NLPModels.increment!(pqp, :neval_grad)
  # g ← H*x
  mul!(g, Symmetric(pqp.data.H, :L), x)
  # vf ← (f+F*θ)
  copy!(pqp.data.vf, pqp.data.f)
  mul!(pqp.data.vf, pqp.data.F, theta, 1, 1)
  # g ← H*x + (f+F*θ)
  g .+= pqp.data.vf
  return g
end

function NLPModels.cons_lin!(
    pqp::ParametricQuadraticModel{T, S},
    x::AbstractVector,
    theta::AbstractVector,
    c::AbstractVector,
  ) where {T, S}
    @lencheck pqp.meta.nvar x
    @lencheck pqp.meta.nlin c
    NLPModels.increment!(pqp, :neval_cons_lin)
  
    # c ← A*x
    mul!(c, pqp.data.A, x)
  
    # c ← B*θ + (A*x)
    mul!(c, pqp.data.B, theta, 1, 1)
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

"""
    is_feasible_theta(pqp::ParametricQuadraticModel, theta)

Check if theta is in the feasible set.
"""
function is_feasible_theta(pqp::ParametricQuadraticModel{T, S}, theta::AbstractVector) where {T, S}
  # Check bounds
  if !all(pqp.data.ltheta .≤ theta .≤ pqp.data.utheta)
    return false
  end

  return true
end

"""
    get_theta_bounds(pqp::ParametricQuadraticModel)

Get the bounds on theta.
"""
function get_theta_bounds(pqp::ParametricQuadraticModel)
  return pqp.data.ltheta, pqp.data.utheta
end


# TODO: slack model
# TODO: convert to QM corresponding to given fixed theta