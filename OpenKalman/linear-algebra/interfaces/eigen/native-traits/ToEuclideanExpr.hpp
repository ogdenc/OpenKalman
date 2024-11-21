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
 * \brief Native Eigen3 traits for Eigen3 general \ref ToEuclideanExpr
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_TOEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_TOEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedObject>
  struct native_traits<OpenKalman::ToEuclideanExpr<NestedObject>>
#else
  template<typename NestedObject>
  struct native_traits<OpenKalman::ToEuclideanExpr<NestedObject>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedObject>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedObject>>
  {
    static constexpr auto BaseFlags = Eigen::internal::traits<std::decay_t<NestedObject>>::Flags;
	using V0 = vector_space_descriptor_of<NestedObject, 0>; 
    enum
    {
      Flags = OpenKalman::euclidean_vector_space_descriptor<V0> ? BaseFlags :
              BaseFlags & ~Eigen::DirectAccessBit & ~Eigen::PacketAccessBit & ~Eigen::LvalueBit &
              ~(OpenKalman::vector<NestedObject> ? 0 : Eigen::LinearAccessBit),
      RowsAtCompileTime = [] {
          if constexpr (OpenKalman::dynamic_vector_space_descriptor<V0>) return Eigen::Dynamic;
          else return static_cast<Eigen::Index>(OpenKalman::euclidean_dimension_size_of_v<V0>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_TOEUCLIDEANEXPR_HPP
