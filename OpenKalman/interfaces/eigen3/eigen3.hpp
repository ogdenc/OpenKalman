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
 * \dir interfaces/eigen3/tests coefficient-types/tests
 * \brief Test files for coefficient types.
 *
 * \file
 * \brief The comprehensive header file for OpenKalman's interface to the Eigen3 library.
 * \details This should be included ''before'' OpenKalman.hpp.
 */

#ifndef OPENKALMAN_EIGEN3_HPP
#define OPENKALMAN_EIGEN3_HPP

#include "../interfaces.hpp"

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
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif


#include "details/eigen3-forward-declarations.hpp"
#include "details/eigen3-matrix-traits.hpp"
#include "details/eigen3-traits.hpp"

#include "details/eigen3-matrix-overloads.hpp"

#include "details/Eigen3Base.hpp"
#include "details/Eigen3MatrixBase.hpp"
#include "details/EigenDynamicBase.hpp"

#include "ConstantMatrix.hpp"
#include "ZeroMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "SelfAdjointMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"

#include "details/eigen3-cholesky-overloads.hpp"
#include "details/eigen3-special-matrix-overloads.hpp"
#include "details/eigen3-special-matrix-arithmetic.hpp"
#include "details/eigen3-euclidean-overloads.hpp"

#include "details/eigen3-native-traits.hpp"
#include "details/eigen3-native-evaluators.hpp"


#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// Introduce key Eigen3 interface functions into OpenKalman namespace.
namespace OpenKalman
{
  using Eigen3::Cholesky_square;
  using Eigen3::Cholesky_factor;

  using Eigen3::ConstantMatrix;
  using Eigen3::ZeroMatrix;
  using Eigen3::IdentityMatrix;
  using Eigen3::SelfAdjointMatrix;
  using Eigen3::TriangularMatrix;
  using Eigen3::DiagonalMatrix;
  using Eigen3::FromEuclideanExpr;
  using Eigen3::ToEuclideanExpr;

  using Eigen3::make_native_matrix;
  using Eigen3::make_ZeroMatrix;
  using Eigen3::make_EigenSelfAdjointMatrix;
  using Eigen3::make_EigenTriangularMatrix;

  using Eigen3::nested_matrix;
  using Eigen3::make_self_contained;
  using Eigen3::row_count;
  using Eigen3::column_count;
  using Eigen3::to_euclidean;
  using Eigen3::from_euclidean;
  using Eigen3::wrap_angles;
  using Eigen3::to_diagonal;
  using Eigen3::diagonal_of;
  using Eigen3::transpose;
  using Eigen3::adjoint;
  using Eigen3::determinant;
  using Eigen3::trace;
  using Eigen3::rank_update;
  using Eigen3::solve;
  using Eigen3::reduce_columns;
  using Eigen3::reduce_rows;
  using Eigen3::LQ_decomposition;
  using Eigen3::QR_decomposition;
  using Eigen3::concatenate_vertical;
  using Eigen3::concatenate_horizontal;
  using Eigen3::concatenate_diagonal;
  using Eigen3::split_vertical;
  using Eigen3::split_horizontal;
  using Eigen3::split_diagonal;
  using Eigen3::get_element;
  using Eigen3::set_element;
  using Eigen3::column;
  using Eigen3::row;
  using Eigen3::apply_columnwise;
  using Eigen3::apply_rowwise;
  using Eigen3::apply_coefficientwise;
  using Eigen3::randomize;

  using Eigen3::eigen_matrix_t;
}

#ifdef EIGEN_IS_NOT_CPLUSPLUS20_COMPATIBLE
#pragma pop_macro("__cpp_concepts")
#endif

#include "default-overloads.hpp"

#include "matrices/details/ElementAccessor.hpp"
#include "matrices/details/MatrixBase.hpp"


#endif //OPENKALMAN_EIGEN3_HPP
