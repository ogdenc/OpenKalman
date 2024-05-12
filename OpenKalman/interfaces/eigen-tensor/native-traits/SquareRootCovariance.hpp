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
 * \brief Native Eigen3 tensor traits for Eigen3 general \ref SquqreRootCovariance
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SQUAREROOTCOVARIANCE_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SQUAREROOTCOVARIANCE_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<typename TypedIndex, OpenKalman::Eigen3::eigen_tensor_general NestedMatrix>
  struct native_traits<OpenKalman::SquareRootCovariance<TypedIndex, NestedMatrix>>
#else
  template<typename TypedIndex, typename NestedMatrix>
  struct native_traits<OpenKalman::SquareRootCovariance<TypedIndex, NestedMatrix>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_tensor_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<std::conditional_t<OpenKalman::triangular_matrix<NestedMatrix>, NestedMatrix, decltype(cholesky_factor(std::declval<NestedMatrix>()))>>>
  {
    using Base = Eigen::internal::traits<std::decay_t<std::conditional_t<
      OpenKalman::triangular_matrix<NestedMatrix>,
      NestedMatrix,
      decltype(cholesky_factor(std::declval<NestedMatrix>()))>>>;
    enum
    {
      Flags = Base::Flags & ~(OpenKalman::triangular_matrix<NestedMatrix> ? 0x0 : Eigen::LvalueBit),
    };
  };


} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SQUAREROOTCOVARIANCE_HPP
