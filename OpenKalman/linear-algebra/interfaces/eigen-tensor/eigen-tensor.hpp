/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "linear-algebra/interfaces/eigen/eigen.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "details/eigen-tensor-forward-declarations.hpp"

#include "functions/make_eigen_tensor_wrapper.hpp"

#include "traits/object_traits_tensor_base.hpp"
#include "eigen-tensor-library-interface.hpp"

#include "traits/eigen-tensor-traits.hpp"

#include "native-traits/eigen-native-tensor-traits.hpp"
#include "native-evaluators/LibraryWrapper.hpp"

#include "details/EigenTensorAdapterBase.hpp"


#endif
