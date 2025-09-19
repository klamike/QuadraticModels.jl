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
export AbstractParametricQuadraticModel, ParametricQuadraticModel, PQPData, evaluate_at_parameter
export jac_param, jac_param!, hess_param, hess_param!
export jac_param_structure!, hess_param_structure!, jac_param_coord!, hess_param_coord!

include("linalg_utils.jl")
include("qpmodel.jl")
include("pqpmodel.jl")
include("presolve/presolve.jl")

end # module
