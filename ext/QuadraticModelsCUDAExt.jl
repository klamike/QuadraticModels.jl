module QuadraticModelsCUDAExt

using CUDA
using CUDA.CUSPARSE
using SparseArrays
using KernelAbstractions
import QuadraticModels: _gather_mul!, _coo_to_scatter

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

function _coo_to_scatter(coo_I::CuVector{Int}, nrows::Int, n_entries::Int)
  if n_entries == 0
    return CUSPARSE.CuSparseMatrixCSC(
      CuVector{Int32}(ones(Int32, 1)),
      CuVector{Int32}(Int32[]),
      CuVector{Float64}(Float64[]),
      (nrows, 0),
    )
  end
  # Build on CPU, transfer to GPU
  cpu_I = Vector{Int}(coo_I)
  cpu_scatter = SparseArrays.sparse(cpu_I, collect(1:n_entries), ones(Float64, n_entries), nrows, n_entries)
  return CUSPARSE.CuSparseMatrixCSC(cpu_scatter)
end

end # module
