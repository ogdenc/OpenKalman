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
 * \brief Native Eigen3 tensor traits for Eigen3 general \ref HermitianAdapter
 */

#ifndef OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SELFADJOINTMATRIX_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_tensor_general NestedMatrix, OpenKalman::HermitianAdapterType storage_triangle>
  struct native_traits<OpenKalman::HermitianAdapter<NestedMatrix, storage_triangle>>
#else
  template<typename NestedMatrix, OpenKalman::HermitianAdapterType storage_triangle>
  struct native_traits<OpenKalman::HermitianAdapter<NestedMatrix, storage_triangle>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_tensor_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
    using Base = Eigen::internal::traits<std::decay_t<NestedMatrix>>;
    enum
    {
      Flags = Base::Flags & ~(OpenKalman::value::complex<typename Base::Scalar> ? Eigen::LvalueBit : 0x0),
    };
  };


} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_TENSOR_NATIVE_TRAITS_SELFADJOINTMATRIX_HPP
