/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Map.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_MAP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_MAP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct IndexibleObjectTraits<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
  {
  private:

    using Xpr = Eigen::Map<PlainObjectType, MapOptions, StrideType>;

  public:

    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters =
      Xpr::RowsAtCompileTime == Eigen::Dynamic or Xpr::ColsAtCompileTime == Eigen::Dynamic or
      Xpr::OuterStrideAtCompileTime == Eigen::Dynamic or Xpr::InnerStrideAtCompileTime == Eigen::Dynamic;

    // Map is not self-contained in any circumstances.
    using type = std::tuple<decltype(*std::declval<typename Xpr::PointerType>())>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return *std::forward<Arg>(arg).data();
    }

    // get_constant() not defined

    // get_constant_diagonal() not defined
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_MAP_HPP
