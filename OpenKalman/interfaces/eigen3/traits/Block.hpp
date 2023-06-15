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
#ifndef __cpp_concepts
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct IndexTraits<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : detail::IndexTraits_Eigen_default<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>> {};
#endif


  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct Dependencies<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type = std::tuple<typename Eigen::internal::ref_selector<XprType>::non_const_type>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    // Eigen::Block should always be converted to Matrix

  };


  // A block taken from a constant matrix is constant.
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct SingleConstant<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
  {
    const Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>& xpr;

    constexpr auto get_constant()
    {
      return constant_coefficient {xpr.nestedExpression()};
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_BLOCK_HPP
