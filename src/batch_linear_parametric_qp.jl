"""
    BatchLinearParametricQuadraticModel

Batch version of `LinearParametricQuadraticModel`:

    min  ½xᵀHx + (c + Fθᵢ)ᵀx + c0       for i = 1, ..., nbatch
    s.t. lconᵢ ≤ Ax + Bθᵢ ≤ uconᵢ
         lvarᵢ ≤ x ≤ uvarᵢ

Shared structure (H, A, F, B) across all instances.
Per-instance: θ (parameters), bounds, and effective cost/constraint RHS.
"""
struct BatchLinearParametricQuadraticModel{T, S, M1, M2, MF, MB, MT} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  data::QPData{T, S, M1, M2}
  F::MF            # nvar × nparam (shared)
  B::MB            # ncon × nparam (shared)
  θ::MT            # nparam × nbatch
  c_base::S        # nvar (shared base cost)
  c_batch::MT      # nvar × nbatch (c_base + F*θᵢ)
  _Bθ::MT          # ncon × nbatch (cached B*θ)
  _HX::MT          # nvar × nbatch (work buffer)
end

function BatchLinearParametricQuadraticModel(
  lpqp::LinearParametricQuadraticModel{T, S, M1, M2, MF, MB},
  nbatch::Int;
  MT = typeof(similar(lpqp.c_base, T, 0, 0)),
  θ = copyto!(MT(undef, length(lpqp.θ), nbatch), repeat(lpqp.θ, 1, nbatch)),
  lvar = fill!(MT(undef, lpqp.meta.nvar, nbatch), T(-Inf)),
  uvar = fill!(MT(undef, lpqp.meta.nvar, nbatch), T(Inf)),
  lcon = fill!(MT(undef, lpqp.meta.ncon, nbatch), T(-Inf)),
  ucon = fill!(MT(undef, lpqp.meta.ncon, nbatch), T(Inf)),
  name::String = "BatchLinearParametricQP",
) where {T, S, M1, M2, MF, MB}
  nvar = lpqp.meta.nvar
  ncon = lpqp.meta.ncon
  nparam = length(lpqp.θ)
  nnzj = lpqp.meta.nnzj
  nnzh = lpqp.meta.nnzh

  nnzjp = length(_param_nzvals(lpqp.B))
  nnzhp = length(_param_nzvals(lpqp.F))
  has_param = nparam > 0
  has_con = ncon > 0

  meta = NLPModels.BatchNLPModelMeta{T, MT}(
    nbatch, nvar;
    lvar = lvar, uvar = uvar,
    ncon = ncon, lcon = lcon, ucon = ucon,
    nnzj = nnzj, nnzh = nnzh,
    islp = (nnzh == 0),
    name = name,
    nparam = nparam,
    nnzgp = nparam,
    nnzjp = nnzjp,
    nnzhp = nnzhp,
    grad_param_available = has_param,
    jac_param_available = has_param && has_con,
    hess_param_available = has_param,
    jpprod_available = has_param && has_con,
    jptprod_available = has_param && has_con,
    hpprod_available = has_param,
    hptprod_available = has_param,
  )

  c_batch = MT(undef, nvar, nbatch)
  _update_c_batch!(c_batch, lpqp.c_base, lpqp.F, θ)
  _Bθ = MT(undef, ncon, nbatch)
  mul!(_Bθ, lpqp.B, θ)
  _HX = MT(undef, nvar, nbatch)

  return BatchLinearParametricQuadraticModel{T, S, M1, M2, MF, MB, MT}(
    meta, lpqp.qp.data, lpqp.F, lpqp.B, copy(θ), copy(lpqp.c_base),
    c_batch, _Bθ, _HX,
  )
end

function _update_c_batch!(c_batch, c_base, F, θ)
  mul!(c_batch, F, θ)
  c_batch .+= c_base
  return c_batch
end

# Constructor from vector of LinearParametricQuadraticModels
function BatchLinearParametricQuadraticModel(
  lpqps::Vector{<:LinearParametricQuadraticModel{T, S, M1, M2, MF, MB}};
  MT = typeof(similar(first(lpqps).c_base, T, 0, 0)),
  name::String = "BatchLinearParametricQP",
) where {T, S, M1, M2, MF, MB}
  nbatch = length(lpqps)
  lpqp1 = first(lpqps)
  θ    = reduce(hcat, [lpqp.θ    for lpqp in lpqps])
  lvar = reduce(hcat, [lpqp.meta.lvar for lpqp in lpqps])
  uvar = reduce(hcat, [lpqp.meta.uvar for lpqp in lpqps])
  lcon = reduce(hcat, [lpqp.meta.lcon for lpqp in lpqps])
  ucon = reduce(hcat, [lpqp.meta.ucon for lpqp in lpqps])
  return BatchLinearParametricQuadraticModel(
    lpqp1, nbatch;
    MT = MT, θ = MT(θ),
    lvar = MT(lvar), uvar = MT(uvar),
    lcon = MT(lcon), ucon = MT(ucon),
    name = name,
  )
end

# ── Standard NLP API ──

function NLPModels.obj!(bqp::BatchLinearParametricQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where T
  H = Symmetric(bqp.data.H, :L)
  mul!(bqp._HX, H, bx)
  bf .= bqp.data.c0 .+ vec(sum(bqp.c_batch .* bx, dims=1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims=1))
  return bf
end

function NLPModels.grad!(bqp::BatchLinearParametricQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where T
  mul!(bg, Symmetric(bqp.data.H, :L), bx)
  bg .+= bqp.c_batch
  return bg
end

function NLPModels.cons!(bqp::BatchLinearParametricQuadraticModel{T}, bx::AbstractMatrix, bc::AbstractMatrix) where T
  mul!(bc, bqp.data.A, bx)
  bc .+= bqp._Bθ
  return bc
end

# Forward structure/coord to ObjRHSBatchQuadraticModel-compatible methods
for (fname, field) in [(:jac_structure!, :A), (:jac_coord!, :A)]
  @eval function NLPModels.$fname(
    bqp::BatchLinearParametricQuadraticModel{T, S, M1, M2},
    args...
  ) where {T, S, M1, M2 <: SparseMatrixCOO}
    _objrhs = ObjRHSBatchQuadraticModel{T, S, M1, M2, typeof(bqp.c_batch)}(
      bqp.meta, bqp.data, bqp.c_batch, bqp._HX, similar(bqp._HX, T, bqp.meta.ncon, bqp.meta.nbatch))
    return NLPModels.$fname(_objrhs, args...)
  end
end

# Simpler forwarding for structure/coord methods
function NLPModels.jac_structure!(
  bqp::BatchLinearParametricQuadraticModel,
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
)
  _param_fill_structure!(bqp.data.A, jrows, jcols)
  return jrows, jcols
end

function NLPModels.jac_coord!(
  bqp::BatchLinearParametricQuadraticModel,
  bx::AbstractMatrix,
  bjvals::AbstractMatrix,
)
  bjvals .= _param_nzvals(bqp.data.A)
  return bjvals
end

function NLPModels.hess_structure!(
  bqp::BatchLinearParametricQuadraticModel,
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
)
  _param_fill_structure!(bqp.data.H, hrows, hcols)
  return hrows, hcols
end

function NLPModels.hess_coord!(
  bqp::BatchLinearParametricQuadraticModel{T},
  bx::AbstractMatrix,
  by::AbstractMatrix,
  bobj_weight::AbstractVector,
  bhvals::AbstractMatrix,
) where T
  nzv = _param_nzvals(bqp.data.H)
  length(nzv) == 0 && return bhvals
  mul!(bhvals, nzv, bobj_weight')
  return bhvals
end

# ── Batch Parametric API ──

function NLPModels.grad_param!(bqp::BatchLinearParametricQuadraticModel, bx::AbstractMatrix, bg::AbstractMatrix)
  mul!(bg, transpose(bqp.F), bx)
  return bg
end

function NLPModels.jpprod!(bqp::BatchLinearParametricQuadraticModel, bx::AbstractMatrix, bv::AbstractMatrix, bJv::AbstractMatrix)
  mul!(bJv, bqp.B, bv)
  return bJv
end

function NLPModels.jptprod!(bqp::BatchLinearParametricQuadraticModel, bx::AbstractMatrix, bv::AbstractMatrix, bJtv::AbstractMatrix)
  mul!(bJtv, transpose(bqp.B), bv)
  return bJtv
end

function NLPModels.hpprod!(
  bqp::BatchLinearParametricQuadraticModel{T},
  bx::AbstractMatrix, by::AbstractMatrix,
  bv::AbstractMatrix, bobj_weight::AbstractVector,
  bHv::AbstractMatrix,
) where T
  mul!(bHv, bqp.F, bv)
  bHv .*= bobj_weight'
  return bHv
end

function NLPModels.hptprod!(
  bqp::BatchLinearParametricQuadraticModel{T},
  bx::AbstractMatrix, by::AbstractMatrix,
  bv::AbstractMatrix, bobj_weight::AbstractVector,
  bHtv::AbstractMatrix,
) where T
  mul!(bHtv, transpose(bqp.F), bv)
  bHtv .*= bobj_weight'
  return bHtv
end
