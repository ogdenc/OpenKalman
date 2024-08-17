/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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
 * \brief Support files for the Eigen3 interface.
 *
 * \dir interfaces/eigen/tests
 * \brief Test files for Eigen3 interface.
 *
 * \file
 * \brief The comprehensive header file for OpenKalman's interface to the Eigen3 library.
 */

#ifndef OPENKALMAN_EIGEN_HPP
#define OPENKALMAN_EIGEN_HPP


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


// Note: c++20 mode requires at least Eigen version 3.3.9. See Eigen Commit 7a0a2a500, which fixes issue #2012.
#if __cplusplus >= 202002L and not EIGEN_VERSION_AT_LEAST(3,3,9)
static_assert(true, "Eigen 3.3.9 required for c++20 or higher standard.");
#endif

#if __cplusplus < 202002L and defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif

#include "../interfaces.hpp" // Includes basics.hpp

/**
 * \internal
 * \brief Namespace for all Eigen3-specific definitions, not intended for use outside of OpenKalman development.
 */
namespace OpenKalman::Eigen3 {}


#include "details/eigen-forward-declarations.hpp"

#include "functions/make_eigen_matrix.hpp"
#include "functions/eigen-wrapper.hpp"

#include "traits/eigen-traits.hpp"

#include "details/eigen-comma-initializers.hpp"

#include "native-traits/eigen-native-traits.hpp"
#include "native-traits/eigen-general-native-traits.hpp"

#include "native-evaluators/eigen-native-evaluators.hpp"

#include "details/EigenAdapterBase.hpp"

#if __cplusplus < 202002L and defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// Introduce key Eigen3 interface functions into OpenKalman namespace.
namespace OpenKalman
{
  using Eigen3::eigen_matrix_t;
}

#include "default-overloads.hpp"

#endif //OPENKALMAN_EIGEN_HPP
