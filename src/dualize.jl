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
  𝓎ₗ = (n + 1):(n + m)
  𝓎ᵤ = (n + m + 1):(n + 2m)
  𝓏ₗ = (n + 2m + 1):(n + 2m + n)
  𝓏ᵤ = (n + 2m + n + 1):n′

  c′ = [
    zeros(T, n)
    -pqp.meta.lcon
    pqp.meta.ucon
    -pqp.meta.lvar
    pqp.meta.uvar
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
    A = A′,
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
