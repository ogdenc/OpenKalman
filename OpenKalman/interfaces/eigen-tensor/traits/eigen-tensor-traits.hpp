/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to the interface to the Eigen library.
 *
 * \file
 * \brief Header file for traits for Eigen tensor classes.
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_TRAITS_HPP
#define OPENKALMAN_EIGEN_TENSOR_TRAITS_HPP


#include "indexible_object_traits_tensor_base.hpp"
#include "eigen-tensor-library-interface.hpp"

#include "Tensor.hpp"
#include "TensorFixedSize.hpp"
#include "TensorMap.hpp"

//#include "TensorArgMax.hpp"
//#include "TensorAssign.hpp"
//#include "TensorBlock.hpp"
//#include "TensorBroadcasting.hpp"
//#include "TensorChipping.hpp"
//#include "TensorConcatenation.hpp"
#include "TensorContractionOp.hpp"
//#include "TensorContractionBlocking.hpp"
//#include "TensorContractionCuda.hpp"
//#include "TensorContractionGPU.hpp"
//#include "TensorContractionMapper.hpp"
//#include "TensorContractionSycl.hpp"
//#include "TensorContractionThreadPool.hpp"
//#include "TensorConversion.hpp"
//#include "TensorConvolution.hpp"
//#include "TensorConvolutionSycl.hpp"
//#include "TensorCostModel.hpp"
//#include "TensorCustomOp.hpp"
#include "TensorCwiseBinaryOp.hpp"
#include "TensorCwiseNullaryOp.hpp"
//#include "TensorCwiseTernaryOp.hpp"
#include "TensorCwiseUnaryOp.hpp"
//#include "TensorDevice.hpp"
//#include "TensorDeviceCuda.hpp"
//#include "TensorDeviceDefault.hpp"
//#include "TensorDeviceGpu.hpp"
//#include "TensorDeviceSycl.hpp"
//#include "TensorDeviceThreadPool.hpp"
//#include "TensorDimensionList.hpp"
//#include "TensorDimensions.hpp"
//#include "TensorEvalTo.hpp"
//#include "TensorEvaluator.hpp"
//#include "TensorExecutor.hpp"
//#include "TensorFFT.hpp"
//#include "TensorForcedEval.hpp"
//#include "TensorFunctors.hpp"
//#include "TensorGenerator.hpp"
//#include "TensorGlobalFunctions.hpp"
//#include "TensorGpuHipCudaDefines.hpp"
//#include "TensorGpuHipCudaUndefines.hpp"
//#include "TensorImagePatch.hpp"
//#include "TensorIndexList.hpp"
//#include "TensorInflation.hpp"
//#include "TensorInitializer.hpp"
//#include "TensorIntDiv.hpp"
//#include "TensorIO.hpp"
//#include "TensorLayoutSwap.hpp"
//#include "TensorMacros.hpp"
//#include "TensorMeta.hpp"
//#include "TensorMorphing.hpp"
//#include "TensorPadding.hpp"
//#include "TensorPatch.hpp"
//#include "TensorRandom.hpp"
//#include "TensorReductionOp.hpp"
//#include "TensorReductionCuda.hpp"
//#include "TensorReductionGpu.hpp"
//#include "TensorReductionSycl.hpp"
//#include "TensorRef.hpp"
//#include "TensorReverse.hpp"
//#include "TensorScan.hpp"
//#include "TensorScanSycl.hpp"
//#include "TensorSelectOp.hpp"
//#include "TensorShuffling.hpp"
//#include "TensorStorage.hpp"
//#include "TensorStriding.hpp"
//#include "TensorTrace.hpp"
//#include "TensorTraits.hpp"
//#include "TensorUInt128.hpp"
//#include "TensorVolumePatch.hpp"


#endif //OPENKALMAN_EIGEN_TENSOR_TRAITS_HPP
