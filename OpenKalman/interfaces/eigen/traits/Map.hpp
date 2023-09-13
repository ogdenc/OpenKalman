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

#ifndef OPENKALMAN_EIGEN_TRAITS_MAP_HPP
#define OPENKALMAN_EIGEN_TRAITS_MAP_HPP

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


    static constexpr Layout layout = std::is_same_v<StrideType, Eigen::Stride<0, 0>> ? layout_of_v<PlainObjectType> : Layout::stride;


    template<typename Arg>
    static constexpr auto
    strides(Arg&& arg)
    {
      constexpr auto outer = StrideType::OuterStrideAtCompileTime;
      constexpr auto inner = StrideType::InnerStrideAtCompileTime;
      if constexpr (outer != Eigen::Dynamic and inner != Eigen::Dynamic)
        return std::tuple {std::integral_constant<std::size_t, outer>{}, std::integral_constant<std::size_t, inner>{}};
      else if constexpr (outer != Eigen::Dynamic and inner == Eigen::Dynamic)
        return std::tuple {std::integral_constant<std::size_t, outer>{}, arg.innerStride()};
      else if constexpr (outer == Eigen::Dynamic and inner != Eigen::Dynamic)
        return std::tuple {arg.outerStride(), std::integral_constant<std::size_t, inner>{}};
      else if constexpr (outer == Eigen::Dynamic and inner == Eigen::Dynamic)
        return std::tuple {arg.outerStride(), arg.innerStride()};
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_MAP_HPP
