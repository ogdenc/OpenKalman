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
 * \brief Type traits as applied to Eigen::VectorBlock.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP
#define OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename VectorType, int Size>
  struct indexible_object_traits<Eigen::VectorBlock<VectorType, Size>>
    : Eigen3::indexible_object_traits_base<Eigen::VectorBlock<VectorType, Size>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::VectorBlock<VectorType, Size>>;

  public:

    template<typename Arg>
    static constexpr auto
    count_indices(const Arg& arg)
    {
      constexpr bool is_row_major = (Eigen::internal::traits<std::decay_t<typename Arg::NestedExpression>>::Flags & Eigen::RowMajorBit) != 0x0;
      return std::integral_constant<std::size_t, is_row_major ? 1 : 0>{};
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    // Eigen::VectorBlock should always be converted to Matrix


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_VECTORBLOCK_HPP
