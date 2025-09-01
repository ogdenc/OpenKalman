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
 * \brief Native Eigen3 traits for Eigen3 general \ref TriangularAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_TRIANGULARMATRIX_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_TRIANGULARMATRIX_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedMatrix, OpenKalman::triangle_type tri>
  struct native_traits<OpenKalman::TriangularAdapter<NestedMatrix, tri>>
#else
  template<typename NestedMatrix, OpenKalman::triangle_type tri>
  struct native_traits<OpenKalman::TriangularAdapter<NestedMatrix, tri>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
    static constexpr auto BaseFlags = Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags;
    enum
    {
      Flags = BaseFlags &
        ~Eigen::DirectAccessBit &
        ~(OpenKalman::one_dimensional<NestedMatrix> ? 0x0 : Eigen::LinearAccessBit | Eigen::PacketAccessBit),
    };
  };


}


#endif
