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
 * \brief Type traits as applied to Eigen::Ref.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_REF_HPP
#define OPENKALMAN_EIGEN_TRAITS_REF_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename PlainObjectType, int Options, typename StrideType>
  struct IndexibleObjectTraits<Eigen::Ref<PlainObjectType, Options, StrideType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Ref<PlainObjectType, Options, StrideType>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::Ref<PlainObjectType, Options, StrideType>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = false;

    // Ref is not self-contained in any circumstances.

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (constant_matrix<PlainObjectType, CompileTimeStatus::known, Likelihood::maybe>)
        return constant_coefficient<PlainObjectType> {};
      else
        return std::monostate {};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (constant_diagonal_matrix<PlainObjectType, CompileTimeStatus::known, Likelihood::maybe>)
        return constant_diagonal_coefficient<PlainObjectType> {};
      else
        return std::monostate {};
    }


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

#endif //OPENKALMAN_EIGEN_TRAITS_REF_HPP
