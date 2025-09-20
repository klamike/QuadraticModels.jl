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
- `v::S`: vector that stores products with the hessian v = H*u
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
  v::S          # vector that stores products with the hessian v = H*u
  H::M2         # Hessian matrix
  A::M3         # constraint matrix for x
  B::M4         # constraint matrix for θ
  P::M5         # parameter constraint matrix
  lparam::S     # lower bounds for parameter constraints
  uparam::S     # upper bounds for parameter constraints
end

@inline PQPData(c0, c, F, H, A, B, P, lparam, uparam) = PQPData(c0, c, F, similar(c), H, A, B, P, lparam, uparam)
isdense(data::PQPData{T, S, M1, M2, M3, M4, M5}) where {T, S, M1, M2, M3, M4, M5} = 
  M1 <: DenseMatrix || M2 <: DenseMatrix || M3 <: DenseMatrix || M4 <: DenseMatrix || M5 <: DenseMatrix

"""
    AbstractParametricQuadraticModel{T, S}

Abstract type for parametric quadratic models.
"""
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
mutable struct ParametricQuadraticModel{T, S, M1, M2, M3, M4, M5} <: AbstractParametricQuadraticModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  data::PQPData{T, S, M1, M2, M3, M4, M5}
end

# Constructor for ParametricQuadraticModel
function ParametricQuadraticModel(
  c::AbstractVector{T},
  F::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  H::Union{AbstractMatrix{T}, AbstractLinearOperator{T}};
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = spzeros(T, 0, length(c)),
  B::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = spzeros(T, 0, size(F, 2)),
  P::Union{AbstractMatrix{T}, AbstractLinearOperator{T}} = spzeros(T, 0, size(F, 2)),
  lcon::AbstractVector{T} = T[],
  ucon::AbstractVector{T} = T[],
  lvar::AbstractVector{T} = fill(-T(Inf), length(c)),
  uvar::AbstractVector{T} = fill(T(Inf), length(c)),
  lparam::AbstractVector{T} = T[],
  uparam::AbstractVector{T} = T[],
  c0::T = zero(T),
  name::String = "ParametricQuadraticModel",
) where {T}
  n = length(c)
  m = length(lcon)
  p = size(F, 2)
  pcon = length(lparam)
  
  # Validate dimensions
  @assert size(F, 1) == n "F must have $(n) rows, got $(size(F, 1))"
  @assert size(H, 1) == size(H, 2) == n "H must be $(n)×$(n), got $(size(H, 1))×$(size(H, 2))"
  @assert size(A, 1) == m "A must have $(m) rows, got $(size(A, 1))"
  @assert size(A, 2) == n "A must have $(n) columns, got $(size(A, 2))"
  @assert size(B, 1) == m "B must have $(m) rows, got $(size(B, 1))"
  @assert size(B, 2) == p "B must have $(p) columns, got $(size(B, 2))"
  @assert size(P, 1) == pcon "P must have $(pcon) rows, got $(size(P, 1))"
  @assert size(P, 2) == p "P must have $(p) columns, got $(size(P, 2))"
  @assert length(lcon) == length(ucon) "lcon and ucon must have the same length"
  @assert length(lvar) == length(uvar) == n "lvar and uvar must have length $(n)"
  @assert length(lparam) == length(uparam) == pcon "lparam and uparam must have length $(pcon)"
  
  # Create data structure
  data = PQPData(c0, c, F, H, A, B, P, lparam, uparam)
  
  # Create meta information
  meta = NLPModelMeta{T, typeof(c)}(
    n,
    lvar = lvar,
    uvar = uvar,
    ncon = m,
    lcon = lcon,
    ucon = ucon,
    nnzj = m * n,
    lin_nnzj = m * n,
    nln_nnzj = 0,
    nnzh = n * (n + 1) ÷ 2,
    lin = collect(1:m),  # All constraints are linear
    islp = (m == 0),
    name = name,
  )
  
  # Create counters
  counters = Counters()
  
  return ParametricQuadraticModel(meta, counters, data)
end

# Convert PQPData to QPData (for compatibility)
function Base.convert(::Type{QPData}, pqp_data::PQPData)
  return QPData(
    pqp_data.c0,
    pqp_data.c,
    pqp_data.v,
    pqp_data.H,
    pqp_data.A,
  )
end

# Convert ParametricQuadraticModel to QuadraticModel (for compatibility)
function Base.convert(::Type{QuadraticModel}, pqp::ParametricQuadraticModel)
  return QuadraticModel(
    convert(QPData, pqp.data),
    pqp.meta,
    pqp.counters,
  )
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
function evaluate_at_parameter(pqp::AbstractParametricQuadraticModel{T, S}, θ::AbstractVector; check_feasibility::Bool = true) where {T, S}
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
  
  # Compute the effective linear term: c + F*θ
  c_eff = similar(pqp.data.c)
  copy!(c_eff, pqp.data.c)
  if length(θ) > 0 && size(pqp.data.F, 2) > 0
    mul!(c_eff, pqp.data.F, θ, 1.0, 1.0)  # c_eff = c + F*θ
  end
  
  # Compute the effective constraint bounds: lcon - B*θ, ucon - B*θ
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

# NLPModels interface implementations

function obj(pqp::AbstractParametricQuadraticModel, x::AbstractVector)
  return pqp.data.c0 + dot(pqp.data.c, x) + 0.5 * dot(x, pqp.data.H * x)
end

function grad!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, g::AbstractVector)
  copy!(g, pqp.data.c)
  mul!(pqp.data.v, pqp.data.H, x)
  g .+= pqp.data.v
  return g
end

function objgrad!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, g::AbstractVector)
  grad!(pqp, x, g)
  return obj(pqp, x)
end

function hess_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, vals::AbstractVector; obj_weight::Real = 1.0)
  fill_coord!(pqp.data.H, vals)
  vals .*= obj_weight
  return vals
end

# NLPModels interface with y parameter
NLPModels.hess_coord!(
  pqp::AbstractParametricQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hess_coord!(pqp, x, vals, obj_weight = obj_weight)

function hess_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  fill_structure!(pqp.data.H, rows, cols)
  return rows, cols
end

function hess(pqp::AbstractParametricQuadraticModel, x::AbstractVector)
  return pqp.data.H
end

function hess_op!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
  mul!(Hv, pqp.data.H, v)
  return Hv
end

function hprod!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real = 1.0)
  mul!(Hv, pqp.data.H, v)
  Hv .*= obj_weight
  return Hv
end

# NLPModels interface with y parameter
NLPModels.hprod!(
  pqp::AbstractParametricQuadraticModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
) = hprod!(pqp, x, v, Hv, obj_weight = obj_weight)

function cons!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, c::AbstractVector)
  mul!(c, pqp.data.A, x)
  return c
end

function jac_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, vals::AbstractVector)
  fill_coord!(pqp.data.A, vals)
  return vals
end

function jac_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  fill_structure!(pqp.data.A, rows, cols)
  return rows, cols
end

function jac(pqp::AbstractParametricQuadraticModel, x::AbstractVector)
  return pqp.data.A
end

function jac_op!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, pqp.data.A, v)
  return Jv
end

function jprod!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  mul!(Jv, pqp.data.A, v)
  return Jv
end

function jtprod!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  mul!(Jtv, pqp.data.A', v)
  return Jtv
end

# Parameter sensitivity functions

"""
    jac_param(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector)

Returns the constraint Jacobian with respect to parameters: ∂(Ax + Bθ)/∂θ = B.
"""
jac_param(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector) = pqp.data.B

"""
    jac_param!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, B::AbstractMatrix)

Copies the constraint Jacobian with respect to parameters B into the provided matrix.
"""
function jac_param!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, B::AbstractMatrix)
  copy!(B, pqp.data.B)
  return B
end

"""
    hess_param(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector)

Returns the mixed Hessian of the Lagrangian with respect to decision variables and parameters: ∂²L/∂x∂θ = F.
"""
hess_param(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector) = pqp.data.F

"""
    hess_param!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector, F::AbstractMatrix)

Copies the mixed Hessian of the Lagrangian with respect to decision variables and parameters F into the provided matrix.
"""
function hess_param!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector, F::AbstractMatrix)
  copy!(F, pqp.data.F)
  return F
end

"""
    jac_param_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})

Fills the sparsity structure of the constraint Jacobian with respect to parameters.
"""
function jac_param_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  fill_structure!(pqp.data.B, rows, cols)
  return rows, cols
end

"""
    hess_param_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})

Fills the sparsity structure of the mixed Hessian with respect to parameters.
"""
function hess_param_structure!(pqp::AbstractParametricQuadraticModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  fill_structure!(pqp.data.F, rows, cols)
  return rows, cols
end

"""
    jac_param_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, vals::AbstractVector)

Fills the coordinate values of the constraint Jacobian with respect to parameters.
"""
function jac_param_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, vals::AbstractVector)
  fill_coord!(pqp.data.B, vals, one(eltype(vals)))
  return vals
end

"""
    hess_param_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector, vals::AbstractVector)

Fills the coordinate values of the mixed Hessian with respect to parameters.
"""
function hess_param_coord!(pqp::AbstractParametricQuadraticModel, x::AbstractVector, θ::AbstractVector, y::AbstractVector, vals::AbstractVector)
  fill_coord!(pqp.data.F, vals, one(eltype(vals)))
  return vals
end

"""
    dualize(pqp::AbstractParametricQuadraticModel)

Returns the dual of the parametric quadratic program.

The dual problem has the form:
    max  -(-(1/2) w' H w + yₗ'lcon - yᵤ'ucon + zₗ'lvar - zᵤ'uvar + (Bθ)'(yₗ - yᵤ) + c₀)
    s.t. A'(yₗ - yᵤ) + zₗ - zᵤ = Hw + c + Fθ
         yₗ, yᵤ, zₗ, zᵤ ≥ 0
         lparam ≤ P θ ≤ uparam

We arrange the variables like x ← [w, yₗ, yᵤ, zₗ, zᵤ]. Its problem data is then:

H ← [
    H 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
]
c ← [0, -lcon, ucon, -lvar, uvar]
F ← [0, -B', B', 0, 0]
c0 ← -c₀
A ← [-H A' -A' I -I]
lcon ← c
ucon ← c
B ← -F
P ← P
lparam ← lparam
uparam ← uparam
lvar ← 0
uvar ← Inf

where:
- w is the dual variable for the quadratic terms
- yₗ, yᵤ are dual variables for lower and upper constraint bounds
- zₗ, zᵤ are dual variables for lower and upper variable bounds
- θ is the parameter vector
"""
function dualize(pqp::AbstractParametricQuadraticModel{T, S}) where {T, S}
    # Get dimensions
    n = pqp.meta.nvar  # number of primal variables
    m = pqp.meta.ncon  # number of constraints
    p = size(pqp.data.F, 2)  # number of parameters
    
    n′ = n + 2m + 2n
    m′ = n

    zeros = issparse(pqp.data.H) ? spzeros : Base.zeros  #FIXME

    𝓌 = 1:n
    𝓎ₗ = n+1:n+m
    𝓎ᵤ = n+m+1:n+2m
    𝓏ₗ = n+2m+1:n+2m+n
    𝓏ᵤ = n+2m+n+1:n′

    c′ = [
        zeros(T, n);
        -pqp.meta.lcon;
        pqp.meta.ucon;
        -pqp.meta.lvar;
        pqp.meta.uvar;
    ]

    F′ = zeros(T, n′, p)
    F′[𝓎ₗ, :] = -pqp.data.B'
    F′[𝓎ᵤ, :] = pqp.data.B'

    H′ = zeros(T, n′, n′)
    H′[𝓌, 𝓌] = pqp.data.H

    A′ = zeros(T, m′, n′)
    A′[:, 𝓌] = -pqp.data.H
    A′[:, 𝓎ₗ] = pqp.data.A'
    A′[:, 𝓎ᵤ] = -pqp.data.A'
    A′[:, 𝓏ₗ] = I(n)
    A′[:, 𝓏ᵤ] = -I(n)

    # Create dual model
    dual_pqp = ParametricQuadraticModel(
        c′,
        F′,
        H′;
        A=A′,
        B = -pqp.data.F,
        P = pqp.data.P,
        lcon = pqp.data.c,
        ucon = pqp.data.c,
        lvar = zeros(n′),
        uvar = Inf * ones(n′),
        lparam = pqp.data.lparam,
        uparam = pqp.data.uparam,
        c0 = -pqp.data.c0,
        name = "Dual of $(pqp.meta.name)",
    )

    return dual_pqp
end
