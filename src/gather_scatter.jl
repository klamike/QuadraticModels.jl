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
    _gather_scatter!(out, A, nz_map, B, val_map, rowptr, colidx)

Fused gather-scatter operation:

    out[r, j] = Σ_{k ∈ rowptr[r]:rowptr[r+1]-1} A[nz_map[colidx[k]], j] * B[val_map[colidx[k]], j]

- `out`: nout × nbatch output matrix
- `A`: coefficient matrix (gather rows via nz_map)
- `nz_map`: index map for gathering from A
- `B`: value matrix (gather rows via val_map)
- `val_map`: index map for gathering from B
- `rowptr`: CSR row pointers (length nout+1)
- `colidx`: CSR column indices (length nnz)

CPU implementation. GPU backends (CUDA) override this via package extensions.
"""
function _gather_scatter!(
  out::AbstractMatrix, A::AbstractMatrix, nz_map::AbstractVector{Int},
  B::AbstractMatrix, val_map::AbstractVector{Int},
  rowptr::AbstractVector{Int}, colidx::AbstractVector{Int},
)
  nout = length(rowptr) - 1
  nbatch = size(out, 2)
  @inbounds for j in 1:nbatch
    for r in 1:nout
      val = zero(eltype(out))
      for k in rowptr[r]:rowptr[r+1]-1
        i = colidx[k]
        val = muladd(A[nz_map[i], j], B[val_map[i], j], val)
      end
      out[r, j] = val
    end
  end
  return out
end

"""
    _gather_scatter_add!(out, A, nz_map, B, val_map, rowptr, colidx, α)

Additive variant: `out[r, j] += α * Σ ...`
"""
function _gather_scatter_add!(
  out::AbstractMatrix, A::AbstractMatrix, nz_map::AbstractVector{Int},
  B::AbstractMatrix, val_map::AbstractVector{Int},
  rowptr::AbstractVector{Int}, colidx::AbstractVector{Int},
  α = one(eltype(out)),
)
  nout = length(rowptr) - 1
  nbatch = size(out, 2)
  @inbounds for j in 1:nbatch
    for r in 1:nout
      val = zero(eltype(out))
      for k in rowptr[r]:rowptr[r+1]-1
        i = colidx[k]
        val = muladd(A[nz_map[i], j], B[val_map[i], j], val)
      end
      out[r, j] += α * val
    end
  end
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
