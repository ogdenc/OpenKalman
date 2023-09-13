/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to the interface to the Eigen3 library.
 *
 * \dir interfaces/eigen/details
 * \brief Support files for the Eigen Tensor module interface.
 *
 * \dir interfaces/eigen/tests
 * \brief Test files for Eigen Tensor module interface.
 *
 * \file
 * \brief The comprehensive header file for OpenKalman's interface to the Eigen library.
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_HPP
#define OPENKALMAN_EIGEN_TENSOR_HPP

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#include "interfaces/eigen/eigen.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "details/eigen-tensor-forward-declarations.hpp"
#include "traits/eigen-tensor-traits.hpp"

#include "traits/Tensor.hpp"
#include "traits/TensorFixedSize.hpp"
#include "traits/TensorMap.hpp"

//#include "traits/TensorArgMax.hpp"
//#include "traits/TensorAssign.hpp"
//#include "traits/TensorBlock.hpp"
//#include "traits/TensorBroadcasting.hpp"
//#include "traits/TensorChipping.hpp"
//#include "traits/TensorConcatenation.hpp"
#include "traits/TensorContractionOp.hpp"
//#include "traits/TensorContractionBlocking.hpp"
//#include "traits/TensorContractionCuda.hpp"
//#include "traits/TensorContractionGPU.hpp"
//#include "traits/TensorContractionMapper.hpp"
//#include "traits/TensorContractionSycl.hpp"
//#include "traits/TensorContractionThreadPool.hpp"
//#include "traits/TensorConversion.hpp"
//#include "traits/TensorConvolution.hpp"
//#include "traits/TensorConvolutionSycl.hpp"
//#include "traits/TensorCostModel.hpp"
//#include "traits/TensorCustomOp.hpp"
#include "traits/TensorCwiseBinaryOp.hpp"
#include "traits/TensorCwiseNullaryOp.hpp"
//#include "traits/TensorCwiseTernaryOp.hpp"
#include "traits/TensorCwiseUnaryOp.hpp"
//#include "traits/TensorDevice.hpp"
//#include "traits/TensorDeviceCuda.hpp"
//#include "traits/TensorDeviceDefault.hpp"
//#include "traits/TensorDeviceGpu.hpp"
//#include "traits/TensorDeviceSycl.hpp"
//#include "traits/TensorDeviceThreadPool.hpp"
//#include "traits/TensorDimensionList.hpp"
//#include "traits/TensorDimensions.hpp"
//#include "traits/TensorEvalTo.hpp"
//#include "traits/TensorEvaluator.hpp"
//#include "traits/TensorExecutor.hpp"
//#include "traits/TensorFFT.hpp"
//#include "traits/TensorForcedEval.hpp"
//#include "traits/TensorFunctors.hpp"
//#include "traits/TensorGenerator.hpp"
//#include "traits/TensorGlobalFunctions.hpp"
//#include "traits/TensorGpuHipCudaDefines.hpp"
//#include "traits/TensorGpuHipCudaUndefines.hpp"
//#include "traits/TensorImagePatch.hpp"
//#include "traits/TensorIndexList.hpp"
//#include "traits/TensorInflation.hpp"
//#include "traits/TensorInitializer.hpp"
//#include "traits/TensorIntDiv.hpp"
//#include "traits/TensorIO.hpp"
//#include "traits/TensorLayoutSwap.hpp"
//#include "traits/TensorMacros.hpp"
//#include "traits/TensorMeta.hpp"
//#include "traits/TensorMorphing.hpp"
//#include "traits/TensorPadding.hpp"
//#include "traits/TensorPatch.hpp"
//#include "traits/TensorRandom.hpp"
//#include "traits/TensorReductionOp.hpp"
//#include "traits/TensorReductionCuda.hpp"
//#include "traits/TensorReductionGpu.hpp"
//#include "traits/TensorReductionSycl.hpp"
//#include "traits/TensorRef.hpp"
//#include "traits/TensorReverse.hpp"
//#include "traits/TensorScan.hpp"
//#include "traits/TensorScanSycl.hpp"
//#include "traits/TensorSelectOp.hpp"
//#include "traits/TensorShuffling.hpp"
//#include "traits/TensorStorage.hpp"
//#include "traits/TensorStriding.hpp"
//#include "traits/TensorTrace.hpp"
//#include "traits/TensorTraits.hpp"
//#include "traits/TensorUInt128.hpp"
//#include "traits/TensorVolumePatch.hpp"


#include "details/EigenTensorAdapterBase.hpp"
#include "details/EigenTensorWrapper.hpp"


#endif //OPENKALMAN_EIGEN_TENSOR_HPP
