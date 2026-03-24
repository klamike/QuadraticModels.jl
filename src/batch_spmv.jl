"""
    _coo_to_csr(indices, n) → (rowptr, colidx)

Convert COO row/column indices to CSR format.
Groups nonzero indices by their row (or column) value.

- `indices`: vector of row (or column) indices from COO format
- `n`: number of rows (or columns)
- Returns `rowptr` (length n+1) and `colidx` (length nnz)
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

"""
    BatchSparseOp{VI, SMT, MT}

Two-step sparse matrix-vector product operator for batched operations.

Step 1: `_gather_mul!` — element-wise gather-multiply into `buffer`
Step 2: `mul!(out, scatter, buffer)` — sparse scatter/reduction

Fields:
- `rowptr`, `flat_nz`, `flat_val`: CPU fused path (CSR)
- `scatter`: SparseMatrixCSC (CPU) or CuSparseMatrixCSC (GPU)
- `nz_map`, `val_map`: index maps for gather step
- `buffer`: workspace for gather-multiply output
"""
struct BatchSparseOp{VI, SMT, MT}
  # CPU fused path
  rowptr::VI
  flat_nz::VI
  flat_val::VI
  # Two-step path
  scatter::SMT
  nz_map::VI
  val_map::VI
  buffer::MT
end

"""
    _coo_to_scatter(coo_I, nrows, n_entries) → (scatter, buffer)

Build a SparseMatrixCSC scatter matrix and buffer from COO row indices.
The scatter matrix has dimensions `nrows × n_entries` with ones at positions
`(coo_I[k], k)` for each nonzero `k`.
"""
function _coo_to_scatter(coo_I::AbstractVector{Int}, nrows::Int, n_entries::Int)
  if n_entries == 0
    scatter = SparseArrays.sparse(Int[], Int[], Float64[], nrows, 0)
    return scatter
  end
  # Build COO triplets for scatter matrix
  coo_J = collect(1:n_entries)
  coo_V = ones(Float64, n_entries)
  scatter = SparseArrays.sparse(coo_I, coo_J, coo_V, nrows, n_entries)
  return scatter
end

"""
    _build_op(rowptr, nz_map, val_map, colidx, scatter, buffer) → BatchSparseOp

Build a `BatchSparseOp` from CSR data and scatter matrix + buffer.
"""
function _build_op(rowptr, nz_map, val_map, colidx, scatter, buffer)
  flat_nz = similar(nz_map, length(colidx))
  flat_val = similar(val_map, length(colidx))
  if length(colidx) > 0
    flat_nz .= nz_map[colidx]
    flat_val .= val_map[colidx]
  end
  return BatchSparseOp(rowptr, flat_nz, flat_val, scatter, nz_map, val_map, buffer)
end

"""
    batch_spmv!(out, A, B, op, alpha=1, beta=0)

Two-step batched sparse matrix-vector product:
  out = alpha * scatter * buffer + beta * out
where buffer = A[nz_map, :] .* B[val_map, :]

- `out`: nout × nbatch output matrix
- `A`: coefficient matrix (gather rows via nz_map)
- `B`: value matrix (gather rows via val_map)
- `op`: BatchSparseOp containing scatter, nz_map, val_map, buffer
"""
function batch_spmv!(
  out::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, op::BatchSparseOp,
  alpha::T = one(T), beta::T = zero(T),
) where T
  _gather_mul!(op.buffer, A, op.nz_map, B, op.val_map)
  mul!(out, op.scatter, op.buffer, alpha, beta)
  return out
end

"""
    batch_spmv_add!(out, A, B, op, alpha=1)

Additive variant: `out += alpha * scatter * buffer`
where buffer = A[nz_map, :] .* B[val_map, :]
"""
function batch_spmv_add!(
  out::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, op::BatchSparseOp,
  alpha::T = one(T),
) where T
  _gather_mul!(op.buffer, A, op.nz_map, B, op.val_map)
  mul!(out, op.scatter, op.buffer, alpha, one(T))
  return out
end

"""
    _gather_mul!(out, A, a_map, B, b_map)

Element-wise batched gather-multiply (no reduction):

    out[k, j] = A[a_map[k], j] * B[b_map[k], j]

CPU implementation. GPU backends (CUDA) override this via package extensions.
"""
function _gather_mul!(
  out::AbstractMatrix, A::AbstractMatrix, a_map::AbstractVector{Int},
  B::AbstractMatrix, b_map::AbstractVector{Int},
)
  n = size(out, 1)
  nbatch = size(out, 2)
  @inbounds for j in 1:nbatch
    for k in 1:n
      out[k, j] = A[a_map[k], j] * B[b_map[k], j]
    end
  end
  return out
end
