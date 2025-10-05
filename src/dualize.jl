"""
    dualize(pqp::AbstractParametricQuadraticModel)

Returns the dual of the parametric quadratic program.

The dual problem has the form:
    max  -(1/2) w' H w + yₗ'lcon - yᵤ'ucon + zₗ'lvar - zᵤ'uvar + (Bθ)'(yₗ - yᵤ) + c₀
    s.t. A'(yₗ - yᵤ) + zₗ - zᵤ = Hw + c + Fθ
         yₗ, yᵤ, zₗ, zᵤ ≥ 0

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
lvar ← 0
uvar ← Inf

where:
- w is the dual variable for the quadratic terms
- yₗ, yᵤ are dual variables for lower and upper constraint bounds
- zₗ, zᵤ are dual variables for lower and upper variable bounds
- θ is the parameter vector
"""
function dualize(pqp::AbstractParametricQuadraticModel{T, S}; skip_adding_equality_constraints::Bool = false) where {T, S}
  # Get dimensions
  n = pqp.meta.nvar  # number of primal variables
  m = pqp.meta.ncon  # number of constraints
  p = size(pqp.data.F, 2)  # number of parameters

  n′ = n + 2m + 2n
  m′ = n

  𝓌 = 1:n
  𝓎ₗ = (n + 1):(n + m)
  𝓎ᵤ = (n + m + 1):(n + 2m)
  𝓏ₗ = (n + 2m + 1):(n + 2m + n)
  𝓏ᵤ = (n + 2m + n + 1):n′

  c′ = [
    fill!(typeof(pqp.data.c)(undef, n), zero(T))
    -pqp.meta.lcon
    pqp.meta.ucon
    -pqp.meta.lvar
    pqp.meta.uvar
  ]

  F′ = similar(pqp.data.F, n′, p)
  fill!(F′, zero(T))
  F′[𝓎ₗ, :] = -pqp.data.B'
  F′[𝓎ᵤ, :] = pqp.data.B'

  H′ = similar(pqp.data.H, n′, n′)
  fill!(H′, zero(T))
  H′[𝓌, 𝓌] = pqp.data.H

  A′, B′, lcon′, ucon′ = if !skip_adding_equality_constraints
    A′ = similar(pqp.data.A, m′, n′)
    A′[:, 𝓌] = -pqp.data.H
    A′[:, 𝓎ₗ] = pqp.data.A'
    A′[:, 𝓎ᵤ] = -pqp.data.A'
    A′[:, 𝓏ₗ] = I(n)
    A′[:, 𝓏ᵤ] = -I(n)

    B′ = -pqp.data.F
    lcon′ = pqp.data.c
    ucon′ = pqp.data.c

    (A′, B′, lcon′, ucon′)
  else
    A′ = similar(pqp.data.A, 0, n′)
    B′ = similar(pqp.data.B, 0, n′)
    lcon′ = similar(pqp.data.lcon, 0)
    ucon′ = similar(pqp.data.ucon, 0)

    (A′, B′, lcon′, ucon′)
  end

  return ParametricQuadraticModel(
    c′,
    F′,
    H′;
    A = A′,
    B = B′,
    lcon = lcon′,
    ucon = ucon′,
    lvar = fill!(typeof(pqp.data.lvar)(undef, n′), zero(T)),
    uvar = fill!(typeof(pqp.data.uvar)(undef, n′), Inf),
    c0 = -pqp.data.c0,
    name = "Dual of $(pqp.meta.name)",  # FIXME  no name?
  )
end
