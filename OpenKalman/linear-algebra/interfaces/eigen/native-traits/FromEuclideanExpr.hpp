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
 * \brief Native Eigen3 traits for Eigen3 general \ref FromEuclideanExpr
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_FROMEUCLIDEANEXPR_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedObject, typename V0>
  struct native_traits<OpenKalman::FromEuclideanExpr<NestedObject, V0>>
#else
  template<typename NestedObject, typename V0>
  struct native_traits<OpenKalman::FromEuclideanExpr<NestedObject, V0>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedObject>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedObject>>
  {
    static constexpr auto BaseFlags = Eigen::internal::traits<std::decay_t<NestedObject>>::Flags;
    enum
    {
      Flags = OpenKalman::coordinates::euclidean_pattern<V0> ? BaseFlags :
              BaseFlags & ~Eigen::DirectAccessBit & ~Eigen::PacketAccessBit & ~Eigen::LvalueBit &
              ~(OpenKalman::vector<NestedObject> ? 0 : Eigen::LinearAccessBit),
      RowsAtCompileTime = [] {
          if constexpr (OpenKalman::dynamic_pattern<V0>) return Eigen::Dynamic;
          else return static_cast<Eigen::Index>(OpenKalman::coordinates::dimension_of_v<V0>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedObject, typename V0>
  struct native_traits<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<NestedObject>, V0>>
#else
    template<typename NestedObject, typename V0>
  struct native_traits<OpenKalman::FromEuclideanExpr<OpenKalman::ToEuclideanExpr<NestedObject>, V0>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedObject>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedObject>>
  {
    static constexpr auto BaseFlags = Eigen::internal::traits<std::decay_t<NestedObject>>::Flags;
    enum
    {
      Flags = OpenKalman::coordinates::euclidean_pattern<V0> ? BaseFlags :
              BaseFlags & ~Eigen::DirectAccessBit & ~Eigen::PacketAccessBit & ~Eigen::LvalueBit &
              ~(OpenKalman::vector<NestedObject> ? 0 : Eigen::LinearAccessBit),
      RowsAtCompileTime = [] {
          if constexpr (OpenKalman::dynamic_pattern<V0>) return Eigen::Dynamic;
          else return static_cast<Eigen::Index>(OpenKalman::coordinates::dimension_of_v<V0>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


}


#endif
