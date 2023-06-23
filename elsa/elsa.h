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
#include "LinearResidual.h"
#include "Huber.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "L2Squared.h"
#include "L2Reg.h"
#include "WeightedL2Squared.h"
#include "LeastSquares.h"
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
#include "SymmetrizedDerivative.h"
#include "EmptyTransform.h"

// Proximal Operator headers
#include "ProximalOperator.h"
#include "ProximalIdentity.h"
#include "ProximalL1.h"
#include "ProximalL0.h"
#include "ProximalBoxConstraint.h"
#include "ProximalL2Squared.h"
#include "CombinedProximal.h"

// Projector headers
#include "Geometry.h"
#include "BinaryMethod.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"
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
#include "CGNE.h"
#include "CGLS.h"
#include "FGM.h"
#include "OGM.h"
#include "ADMML2.h"
#include "SQS.h"
#include "SIRT.h"
#include "Landweber.h"
#include "AB_GMRES.h"
#include "BA_GMRES.h"
#include "SplitBregman.h"
#include "LBK.h"
#include "LB.h"
#include "TGV_LADMM.h"
#include "LinearizedADMM.h"
