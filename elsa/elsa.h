#pragma once

// Core headers
#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "LinearOperator.h"
#include "IdenticalBlocksDescriptor.h"
#include "PartitionDescriptor.h"
#include "RandomBlocksDescriptor.h"

// Functional headers
#include "Functional.h"
#include "Residual.h"
#include "LinearResidual.h"
#include "Huber.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "LInfNorm.h"
#include "PseudoHuber.h"
#include "Quadric.h"
#include "EmissionLogLikelihood.h"
#include "TransmissionLogLikelihood.h"

// Generators headers
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"

// IO headers
#include "EDFHandler.h"
#include "MHDHandler.h"
#include "PGMHandler.h"

// Logging headers
#include "Logger.h"
#include "LogGuard.h"
#include "Timer.h"

// Operator headers
#include "Identity.h"
#include "Scaling.h"
#include "FiniteDifferences.h"
#include "BlockLinearOperator.h"

// Proximity Operator headers
#include "ProximityOperator.h"
#include "SoftThresholding.h"
#include "HardThresholding.h"

// Problem headers
#include "Problem.h"
#include "RegularizationTerm.h"
#include "QuadricProblem.h"
#include "WLSProblem.h"
#include "TikhonovProblem.h"
#include "LASSOProblem.h"

// Projector headers
#include "Geometry.h"
#include "BinaryMethod.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"

// CUDA projectors
#ifdef ELSA_CUDA_PROJECTORS
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"
#endif

// Solver headers
#include "Solver.h"
#include "GradientDescent.h"
#include "CG.h"
#include "ISTA.h"
#include "FISTA.h"
#include "FGM.h"
#include "OGM.h"
#include "ADMM.h"
#include "SIRT.h"
#include "Landweber.h"
