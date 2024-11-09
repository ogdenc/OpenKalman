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
 * \brief Native Eigen3 traits for Eigen3 general \ref DiagonalAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_DIAGONALMATRIX_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedMatrix>
  struct native_traits<OpenKalman::DiagonalAdapter<NestedMatrix>>
#else
  template<typename NestedMatrix>
  struct native_traits<OpenKalman::DiagonalAdapter<NestedMatrix>, std::enable_if_t<OpenKalman::Eigen3::eigen_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
    using Base = Eigen::internal::traits<std::decay_t<NestedMatrix>>;
    enum
    {
      Flags = Base::Flags &
        ~Eigen::DirectAccessBit &
        ~(OpenKalman::one_dimensional<NestedMatrix> ? 0x0 : Eigen::LinearAccessBit | Eigen::PacketAccessBit),
      ColsAtCompileTime = Base::RowsAtCompileTime,
      MaxColsAtCompileTime = Base::MaxRowsAtCompileTime,
    };
  };


} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_DIAGONALMATRIX_HPP
