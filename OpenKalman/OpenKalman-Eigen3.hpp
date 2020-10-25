/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * @file
 * This is the meta-header file for OpenKalman in combination with Eigen3.
 *
 * Include this file before using any OpenKalman classes or methods with Eigen3.
 */

#ifndef OPENKALMAN_OPENKALMAN_EIGEN3_HPP
#define OPENKALMAN_OPENKALMAN_EIGEN3_HPP

#include "interfaces/eigen3/eigen3.hpp"
#include "OpenKalman.hpp"

namespace OpenKalman
{
  using Eigen3::Cholesky_square;
  using Eigen3::Cholesky_factor;

  using Eigen3::EigenSelfAdjointMatrix;
  using Eigen3::EigenTriangularMatrix;
  using Eigen3::EigenDiagonal;
  using Eigen3::EigenZero;
  using Eigen3::FromEuclideanExpr;
  using Eigen3::ToEuclideanExpr;

  using Eigen3::make_native_matrix;
  using Eigen3::make_EigenSelfAdjointMatrix;
  using Eigen3::make_EigenTriangularMatrix;

  using Eigen3::base_matrix;
  using Eigen3::strict_matrix;
  using Eigen3::strict;
  using Eigen3::to_Euclidean;
  using Eigen3::from_Euclidean;
  using Eigen3::wrap_angles;
  using Eigen3::to_diagonal;
  using Eigen3::transpose;
  using Eigen3::adjoint;
  using Eigen3::determinant;
  using Eigen3::trace;
  using Eigen3::rank_update;
  using Eigen3::solve;
  using Eigen3::reduce_columns;
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
  using Eigen3::apply_columnwise;
  using Eigen3::apply_coefficientwise;
  using Eigen3::randomize;

}

#endif //OPENKALMAN_OPENKALMAN_EIGEN3_HPP
