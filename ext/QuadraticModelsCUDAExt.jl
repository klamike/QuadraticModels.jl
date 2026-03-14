module QuadraticModelsCUDAExt

using CUDA
using KernelAbstractions
import QuadraticModels: _gather_scatter!, _gather_scatter_add!, _gather_mul!

@kernel function _gather_scatter_kernel!(
  out, @Const(A), @Const(nz_map), @Const(B), @Const(val_map),
  @Const(rowptr), @Const(colidx),
)
  j, r = @index(Global, NTuple)
  val = zero(eltype(out))
  @inbounds for k in rowptr[r]:rowptr[r+1]-1
    i = colidx[k]
    val = muladd(A[nz_map[i], j], B[val_map[i], j], val)
  end
  @inbounds out[r, j] = val
end

function _gather_scatter!(
  out::CuMatrix, A::CuMatrix, nz_map::CuVector{Int},
  B::CuMatrix, val_map::CuVector{Int},
  rowptr::CuVector{Int}, colidx::CuVector{Int},
)
  nout = length(rowptr) - 1
  bs = size(out, 2)
  if nout > 0
    backend = CUDABackend()
    _gather_scatter_kernel!(backend)(
      out, A, nz_map, B, val_map, rowptr, colidx;
      ndrange = (bs, nout), workgroupsize = (32, 4),
    )
  end
  return out
end

@kernel function _gather_scatter_add_kernel!(
  out, @Const(A), @Const(nz_map), @Const(B), @Const(val_map),
  @Const(rowptr), @Const(colidx), @Const(α),
)
  j, r = @index(Global, NTuple)
  val = zero(eltype(out))
  @inbounds for k in rowptr[r]:rowptr[r+1]-1
    i = colidx[k]
    val = muladd(A[nz_map[i], j], B[val_map[i], j], val)
  end
  @inbounds out[r, j] += α * val
end

function _gather_scatter_add!(
  out::CuMatrix, A::CuMatrix, nz_map::CuVector{Int},
  B::CuMatrix, val_map::CuVector{Int},
  rowptr::CuVector{Int}, colidx::CuVector{Int},
  α = one(eltype(out)),
)
  nout = length(rowptr) - 1
  bs = size(out, 2)
  if nout > 0
    backend = CUDABackend()
    _gather_scatter_add_kernel!(backend)(
      out, A, nz_map, B, val_map, rowptr, colidx, α;
      ndrange = (bs, nout), workgroupsize = (32, 4),
    )
  end
  return out
end

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

end # module
