"""
    _coo_to_csr(indices, n) → (rowptr, colidx)

Convert COO row/column indices to CSR format.
"""
function _coo_to_csr(indices::AbstractVector{Int}, n::Int)
  nnz = length(indices)
  rowptr = zeros(Int, n + 1)
  for i in 1:nnz
    rowptr[indices[i] + 1] += 1
  end
  rowptr[1] = 1
  for r in 1:n
    rowptr[r + 1] += rowptr[r]
  end
  colidx = Vector{Int}(undef, nnz)
  pos = copy(rowptr[1:n])
  for i in 1:nnz
    r = indices[i]
    colidx[pos[r]] = i
    pos[r] += 1
  end
  return rowptr, colidx
end

struct BatchSparseOp{VI, MT}
  nzVals::MT
  rowptr::VI
  flat_nz::VI
  flat_val::VI
  max_row_nnz::Int32
  mean_row_nnz::Float32
end

function _row_stats(rowptr::AbstractVector)
  nrows = length(rowptr) - 1
  nrows == 0 && return Int32(0), Float32(0)
  max_nnz = Int32(0)
  total = Int32(0)
  @inbounds for r in 1:nrows
    rl = Int32(rowptr[r+1] - rowptr[r])
    max_nnz = max(max_nnz, rl)
    total += rl
  end
  return max_nnz, Float32(total / nrows)
end

function _build_op(nzVals, rowptr, nz_map, val_map, colidx)
  max_nnz, mean_nnz = _row_stats(rowptr)
  flat_nz = similar(nz_map, length(colidx))
  flat_val = similar(val_map, length(colidx))
  if length(colidx) > 0
    flat_nz .= nz_map[colidx]
    flat_val .= val_map[colidx]
  end
  return BatchSparseOp(nzVals, rowptr, flat_nz, flat_val, max_nnz, mean_nnz)
end

function batch_spmv!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix,
  alpha::T = one(T), beta::T = zero(T); val_offset::Int = 0,
) where T
  _batch_spmv_impl!(out, op, B, alpha, beta, Int32(val_offset))
end

function _batch_spmv_impl!(
  out::AbstractMatrix{T}, op::BatchSparseOp, B::AbstractMatrix,
  alpha::T, beta::T, val_offset::Int32 = Int32(0),
) where T
  nout = length(op.rowptr) - 1
  bs = size(out, 2)
  @inbounds for r in 1:nout
    for j in 1:bs
      acc = zero(T)
      for k in op.rowptr[r]:op.rowptr[r+1]-1
        acc += op.nzVals[op.flat_nz[k], j] * B[op.flat_val[k] + val_offset, j]
      end
      out[r, j] = alpha * acc + beta * out[r, j]
    end
  end
  return out
end

"""
    _gather_mul!(out, A, a_map, B, b_map)

Element-wise batched gather-multiply: out[k, j] = A[a_map[k], j] * B[b_map[k], j]
"""
function _gather_mul!(
  out::AbstractMatrix, A::AbstractMatrix, a_map::AbstractVector,
  B::AbstractMatrix, b_map::AbstractVector,
)
  @inbounds for j in 1:size(out, 2), k in 1:size(out, 1)
    out[k, j] = A[a_map[k], j] * B[b_map[k], j]
  end
  return out
end
