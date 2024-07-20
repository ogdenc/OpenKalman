/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 tensor traits for Eigen3 general \ref DiagonalMatrix
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_DIAGONALMATRIX_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_tensor_general NestedMatrix>
  struct native_traits<OpenKalman::DiagonalMatrix<NestedMatrix>>
#else
  template<typename NestedMatrix>
  struct native_traits<OpenKalman::DiagonalMatrix<NestedMatrix>, std::enable_if_t<OpenKalman::Eigen3::eigen_tensor_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
    static const int NumDimensions = std::min(index_count_v<NestedMatrix>, 2);
  };


} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_DIAGONALMATRIX_HPP
