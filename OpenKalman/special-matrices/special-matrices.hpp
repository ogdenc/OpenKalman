/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to matrices.
 *
 * \dir matrices/details
 * \brief Support files for matrices.
 *
 * \file
 * A meta-header file including all the headers relating to OpenKalman typed matrices.
 */

#ifndef OPENKALMAN_MATRICES_HPP
#define OPENKALMAN_MATRICES_HPP

#include "details/eigen3-special_matrix-traits.hpp"

#include "ConstantMatrix.hpp"
#include "ZeroMatrix.hpp"
#include "DiagonalMatrix.hpp"
#include "SelfAdjointMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"

#include "details/special_matrix-cholesky-overloads.hpp"

#include "details/eigen3-special-matrix-overloads.hpp"
#include "details/eigen3-special-matrix-arithmetic.hpp"
#include "details/eigen3-euclidean-overloads.hpp"


// Introduce key Eigen3 interface functions into OpenKalman namespace.
namespace OpenKalman
{
  using Eigen3::ConstantMatrix;
  using Eigen3::ZeroMatrix;
  using Eigen3::IdentityMatrix;
  using Eigen3::SelfAdjointMatrix;
  using Eigen3::TriangularMatrix;
  using Eigen3::DiagonalMatrix;
  using Eigen3::FromEuclideanExpr;
  using Eigen3::ToEuclideanExpr;

  using Eigen3::make_EigenSelfAdjointMatrix;
  using Eigen3::make_EigenTriangularMatrix;
}

#endif //OPENKALMAN_MATRICES_HPP
