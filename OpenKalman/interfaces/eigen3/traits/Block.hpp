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
 * \brief Type traits as applied to Eigen::Block.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_BLOCK_HPP
#define OPENKALMAN_EIGEN3_TRAITS_BLOCK_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct IndexibleObjectTraits<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
  {
    static constexpr std::size_t max_indices = 2;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      using Xpr = Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>;
      constexpr Eigen::Index dim = N == 0 ? Xpr::RowsAtCompileTime : Xpr::ColsAtCompileTime;

      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    static constexpr bool has_runtime_parameters = true;
    using type = std::tuple<typename Eigen::internal::ref_selector<XprType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // convert_to_self_contained() not defined because Eigen::Block should always be converted to Matrix

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_BLOCK_HPP
