#pragma once

// Core headers
#include "core/elsaDefines.h"
#include "core/DataContainer.h"
#include "core/DataDescriptor.h"
#include "core/LinearOperator.h"

// Functional headers
#include "functionals/Functional.h"
#include "functionals/Residual.h"
#include "functionals/LinearResidual.h"
#include "functionals/Huber.h"
#include "functionals/L1Norm.h"
#include "functionals/L2NormPow2.h"
#include "functionals/WeightedL2NormPow2.h"
#include "functionals/LInfNorm.h"
#include "functionals/PseudoHuber.h"
#include "functionals/Quadric.h"
#include "functionals/EmissionLogLikelihood.h"
#include "functionals/TransmissionLogLikelihood.h"

// Generators headers
#include "generators/PhantomGenerator.h"
#include "generators/CircleTrajectoryGenerator.h"

// IO headers
#include "io/EDFHandler.h"
#include "io/MHDHandler.h"

// Logging headers
#include "logging/Logger.h"
#include "logging/LogGuard.h"
#include "logging/Timer.h"

// Operator headers
#include "operators/Identity.h"
#include "operators/Scaling.h"
#include "operators/FiniteDifferences.h"

// Problem headers
#include "problems/Problem.h"
#include "problems/RegularizationTerm.h"
#include "problems/QuadricProblem.h"
#include "problems/WLSProblem.h"

// Projector headers
#include "projectors/Geometry.h"
#include "projectors/BinaryMethod.h"
#include "projectors/JosephsMethod.h"
#include "projectors/SiddonsMethod.h"

// CUDA projectors
#ifdef ELSA_CUDA_PROJECTORS
#include "projectors_cuda/SiddonsMethodCUDA.h"
#include "projectors_cuda/JosephsMethodCUDA.h"
#endif

// Solver headers
#include "solvers/Solver.h"
#include "solvers/GradientDescent.h"
#include "solvers/CG.h"

// Ml headers
#include "ml/Layer.h"
#include "ml/ActivationLayer.h"
#include "ml/ConvLayer.h"
#include "ml/PoolingLayer.h"
#include "ml/DenseLayer.h"
#include "ml/RandomInitializer.h"
#include "ml/SequentialNetwork.h"
