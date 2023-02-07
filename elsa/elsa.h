#pragma once

// Core headers
#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Error.h"
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
#include "SeparableSum.h"

// Generators headers
#include "Phantoms.h"
#include "CircleTrajectoryGenerator.h"
#include "LimitedAngleTrajectoryGenerator.h"
#include "SphereTrajectoryGenerator.h"
#include "NoiseGenerators.h"

// IO headers
#include "EDFHandler.h"
#include "MHDHandler.h"
#include "PGMHandler.h"
#include "IO.h"

// Logging headers
#include "Logger.h"
#include "LogGuard.h"
#include "Timer.h"

// Operator headers
#include "Identity.h"
#include "Scaling.h"
#include "FiniteDifferences.h"
#include "BlockLinearOperator.h"
#include "ShearletTransform.h"

// Proximal Operator headers
#include "ProximalOperator.h"
#include "ProximalL1.h"
#include "ProximalL0.h"
#include "CombinedProximal.h"

// Problem headers
#include "Problem.h"
#include "RegularizationTerm.h"
#include "QuadricProblem.h"
#include "WLSProblem.h"
#include "TikhonovProblem.h"
#include "LASSOProblem.h"
#include "SubsetProblem.h"
#include "WLSSubsetProblem.h"

// Projector headers
#include "Geometry.h"
#include "BinaryMethod.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"
#include "SiddonsMethodBranchless.h"
#include "SubsetSampler.h"
#include "VoxelProjector.h"

// CUDA projectors
#ifdef ELSA_CUDA_PROJECTORS
#include "SiddonsMethodCUDA.h"
#include "JosephsMethodCUDA.h"
#include "VoxelProjectorCUDA.h"
#endif

// Solver headers
#include "Solver.h"
#include "GradientDescent.h"
#include "PGD.h"
#include "APGD.h"
#include "CGLS.h"
#include "FGM.h"
#include "OGM.h"
#include "ADMM.h"
#include "SQS.h"
#include "SIRT.h"
#include "Landweber.h"
