module QuadraticModels

# stdlib
using LinearAlgebra, SparseArrays

# our packages
using LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SparseMatricesCOO

import NLPModels:
  objgrad,
  objgrad!,
  obj,
  grad,
  grad!,
  hess_coord,
  hess,
  hess_op,
  hprod,
  cons,
  cons!,
  jac_coord,
  jac,
  jac_op,
  jprod,
  jtprod
import NLPModelsModifiers: SlackModel, slack_meta

import Base.convert

export AbstractQuadraticModel, QuadraticModel, presolve, postsolve, postsolve!, QMSolution
export LinearParametricQuadraticModel, BatchLinearParametricQuadraticModel
export ParametricQuadraticModel, BatchParametricQuadraticModel
export BatchSparseOp, batch_spmv!, _batch_spmv_impl!, _build_op, _row_stats, _coo_to_csr, _gather_mul!

include("linalg_utils.jl")
include("batch_spmv.jl")
include("qpmodel.jl")
include("presolve/presolve.jl")
include("objrhsbatchqp.jl")
include("batchqp.jl")
include("linear_parametric_qp.jl")
include("batch_linear_parametric_qp.jl")
include("parametric_qp.jl")
include("batch_parametric_qp.jl")

end # module
