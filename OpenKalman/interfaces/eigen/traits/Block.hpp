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

#ifndef OPENKALMAN_EIGEN_TRAITS_BLOCK_HPP
#define OPENKALMAN_EIGEN_TRAITS_BLOCK_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct indexible_object_traits<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : Eigen3::indexible_object_traits_base<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
  {
  private:

    using Xpr = Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;
    using XprTypeNested = typename Eigen::internal::ref_selector<XprType>::non_const_type;

  public:

    static constexpr bool has_runtime_parameters = true;
    using dependents = std::tuple<typename Eigen::internal::ref_selector<XprType>::non_const_type>;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_BLOCK_HPP
