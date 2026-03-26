module QuadraticModelsCUDAExt

using CUDA
using CUDA.CUSPARSE
using SparseArrays
using NLPModels
using KernelAbstractions
import QuadraticModels
import QuadraticModels: _gather_mul!, _batch_spmv_impl!, _to_gpu, BatchSparseOp,
    BatchQuadraticModel, ObjRHSBatchQuadraticModel, QPData

@kernel function _gather_mul_kernel!(
  out, @Const(A), @Const(a_map), @Const(B), @Const(b_map),
)
  j, k = @index(Global, NTuple)
  @inbounds out[k, j] = A[a_map[k], j] * B[b_map[k], j]
end

function _gather_mul!(
  out::CuMatrix, A::CuMatrix, a_map::CuVector{Int},
  B::CuMatrix, b_map::CuVector{Int},
)
  n = size(out, 1)
  bs = size(out, 2)
  if n > 0
    backend = CUDABackend()
    _gather_mul_kernel!(backend)(
      out, A, a_map, B, b_map;
      ndrange = (bs, n), workgroupsize = (32, 4),
    )
  end
  return out
end

const WARP_KERNEL_THRESHOLD = Int32(4)

function _batch_spmv_impl!(
    out::AbstractMatrix{T}, op::BatchSparseOp{<:CuVector}, B::AbstractMatrix{T},
    alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where T
    nout = Int32(length(op.rowptr) - 1)
    bs = Int32(size(out, 2))
    (nout == 0 || bs == 0) && return out
    if op.mean_row_nnz <= WARP_KERNEL_THRESHOLD
        _launch_scalar_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
    else
        _launch_warp_kernel!(out, op, B, alpha, beta, val_offset, nout, bs)
    end
    return out
end

# scalar kernel: 1 thread per (r, j)
_scalar_spmv_kernel!(
    out, A, B,
    flat_packed, rowptr,
    alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
    j = Int32((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)

    (r > nout || j > bs) && return nothing

    acc = zero(eltype(out))
    @inbounds begin
        for k in rowptr[r]:rowptr[r + Int32(1)] - Int32(1)
            packed = flat_packed[k]
            nz = Int32(packed >> 32)
            val = Int32(packed & 0xffffffff)
            acc += A[nz, j] * B[val + val_offset, j]
        end
        out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
    end
    return nothing
end

function _launch_scalar_kernel!(
    out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
    val_offset::Int32, nout::Int32, bs::Int32,
) where T
    tx, ty = Int32(32), Int32(8)
    threads = (tx, ty)
    blocks = (cld(Int(bs), Int(tx)), cld(Int(nout), Int(ty)))
    CUDA.@cuda always_inline=true threads=threads blocks=blocks _scalar_spmv_kernel!(
        out, op.nzVals, B, op.flat_packed, op.rowptr,
        alpha, beta, val_offset, nout, bs,
    )
end

# warp kernel: 1 warp / 32 threads per (r, j) output
_warp_spmv_kernel!(
    out, A, B,
    flat_packed, rowptr,
    alpha, beta, val_offset::Int32, nout::Int32, bs::Int32,
) = begin
    lane = Int32(threadIdx().x - Int32(1))  # 0:31
    r = Int32((blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y)
    j = Int32(blockIdx().x)

    (r > nout || j > bs) && return nothing

    acc = zero(eltype(out))
    @inbounds begin
        start = rowptr[r]
        stop = rowptr[r + Int32(1)] - Int32(1)
        k = start + lane
        while k <= stop
            packed = flat_packed[k]
            nz = Int32(packed >> 32)
            val = Int32(packed & 0xffffffff)
            acc += A[nz, j] * B[val + val_offset, j]
            k += Int32(32)
        end
    end

    offset = Int32(16)
    while offset > Int32(0)
        acc += CUDA.shfl_down_sync(0xffffffff, acc, offset)
        offset >>= Int32(1)
    end

    @inbounds if lane == Int32(0)
        out[r, j] = iszero(beta) ? alpha * acc : alpha * acc + beta * out[r, j]
    end
    return nothing
end

function _launch_warp_kernel!(
    out::AbstractMatrix{T}, op, B, alpha::T, beta::T,
    val_offset::Int32, nout::Int32, bs::Int32,
) where T
    rows_per_block = Int32(4)
    threads = (Int32(32), rows_per_block)
    blocks = (Int(bs), cld(Int(nout), Int(rows_per_block)))
    CUDA.@cuda always_inline=true threads=threads blocks=blocks _warp_spmv_kernel!(
        out, op.nzVals, B, op.flat_packed, op.rowptr,
        alpha, beta, val_offset, nout, bs,
    )
end

function _to_gpu(op::BatchSparseOp, nzVals_gpu::CuMatrix)
    BatchSparseOp(
        nzVals_gpu,
        CuVector{Int32}(op.rowptr),
        CuVector{Int32}(op.flat_nz),
        CuVector{Int32}(op.flat_val),
        CuVector{Int64}(op.flat_packed),
        op.max_row_nnz,
        op.mean_row_nnz,
    )
end

function Base.convert(::Type{BatchQuadraticModel{T, MT}}, bnlp::BatchQuadraticModel{T}) where {T, MT<:CuMatrix}
    nbatch = bnlp.meta.nbatch
    nvar = bnlp.meta.nvar
    ncon = bnlp.meta.ncon

    meta_gpu = NLPModels.BatchNLPModelMeta{T, MT}(
        nbatch, nvar;
        x0 = MT(bnlp.meta.x0),
        lvar = MT(bnlp.meta.lvar),
        uvar = MT(bnlp.meta.uvar),
        ncon = ncon,
        lcon = MT(bnlp.meta.lcon),
        ucon = MT(bnlp.meta.ucon),
        nnzj = bnlp.meta.nnzj,
        nnzh = bnlp.meta.nnzh,
        islp = bnlp.meta.islp,
    )

    VT = CuVector{T}
    VI = CuVector{Int}

    H_nzvals_gpu = MT(bnlp.H_nzvals)
    A_nzvals_gpu = MT(bnlp.A_nzvals)

    return BatchQuadraticModel{T, MT, VT, VI}(
        meta_gpu,
        MT(bnlp.c_batch), VT(bnlp.c0_batch),
        H_nzvals_gpu, A_nzvals_gpu,
        VI(bnlp.hess_rows), VI(bnlp.hess_cols),
        VI(bnlp.A_rows), VI(bnlp.A_cols),
        _to_gpu(bnlp.jac_op, A_nzvals_gpu),
        _to_gpu(bnlp.jact_op, A_nzvals_gpu),
        _to_gpu(bnlp.hess_op, H_nzvals_gpu),
        CUDA.zeros(T, nvar, nbatch),
    )
end

end # module
