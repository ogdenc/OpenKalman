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
 * \brief Files relating to the interface to the Eigen3 library.
 *
 * \dir interfaces/eigen3/details
 * \brief Support files for the Eigen3 interface.
 *
 * \dir interfaces/eigen3/tests
 * \brief Test files for Eigen3 interface.
 *
 * \file
 * \brief The comprehensive header file for OpenKalman's interface to the Eigen3 library.
 * \details This should be included ''before'' OpenKalman.hpp.
 */

#ifndef OPENKALMAN_EIGEN3_HPP
#define OPENKALMAN_EIGEN3_HPP


#include <Eigen/Dense>

// Note: c++20 mode requires at least Eigen version 3.3.9. See Eigen Commit 7a0a2a500, which fixes issue #2012.
#if not EIGEN_VERSION_AT_LEAST(3,3,9) and not defined(EIGEN_OPENKALMAN_CUSTOM_UPDATE_ADDING_COMMIT_7a0a2a500)
#define EIGEN_IS_NOT_CPLUSPLUS20_COMPATIBLE
#endif

#ifdef EIGEN_IS_NOT_CPLUSPLUS20_COMPATIBLE
#pragma push_macro("__cpp_concepts")
#undef __cpp_concepts
#endif


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif

#include "../interfaces.hpp" // Includes basics.hpp

#include "details/eigen3-forward-declarations.hpp"

#include "details/eigen3-matrix-traits.hpp"

#include "traits/eigen3-interface.hpp"
#include "traits/eigen3-traits.hpp"

#include "traits/functors/nullary.hpp"
#include "traits/functors/unary.hpp"
#include "traits/functors/binary.hpp"

#include "traits/CwiseBinaryOp.hpp"
#include "traits/CwiseNullaryOp.hpp"
#include "traits/CwiseUnaryOp.hpp"
#include "traits/CwiseUnaryView.hpp"
#include "traits/CwiseTernaryOp.hpp"

#include "traits/PartialReduxExpr.hpp"

#include "details/eigen3-functions.hpp"

#include "details/eigen3-cholesky-overloads.hpp"

#include "details/eigen3-comma-initializers.hpp"

#include "details/eigen3-native-traits.hpp"
#include "details/eigen3-native-evaluators.hpp"

#include "details/Eigen3AdapterBase.hpp"
#include "details/EigenWrapper.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// Introduce key Eigen3 interface functions into OpenKalman namespace.
namespace OpenKalman
{
  using Eigen3::eigen_matrix_t;
}

#ifdef EIGEN_IS_NOT_CPLUSPLUS20_COMPATIBLE
#pragma pop_macro("__cpp_concepts")
#endif

#include "default-overloads.hpp"

#endif //OPENKALMAN_EIGEN3_HPP
