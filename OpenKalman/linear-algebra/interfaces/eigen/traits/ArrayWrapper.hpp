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
 * \brief Type traits as applied to Eigen::ArrayWrapper.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_ARRAYWRAPPER_HPP
#define OPENKALMAN_EIGEN_TRAITS_ARRAYWRAPPER_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename XprType>
  struct object_traits<Eigen::ArrayWrapper<XprType>>
    : Eigen3::object_traits_base<Eigen::ArrayWrapper<XprType>>
  {
  private:

    using Base = Eigen3::object_traits_base<Eigen::ArrayWrapper<XprType>>;
    using NestedXpr = typename Eigen::ArrayWrapper<XprType>::NestedExpressionType;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_pattern_collection(const Arg& arg, N n)
    {
      return OpenKalman::get_pattern_collection(arg.nestedExpression(), n);
    }

    template<typename Arg>
    static NestedXpr
    nested_object(Arg&& arg)
    {
      if constexpr (std::is_lvalue_reference_v<NestedXpr>)
        return const_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
      else
        return static_cast<NestedXpr>(std::forward<Arg>(arg).nestedExpression());
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_value{arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_value {arg.nestedExpression()};
    }


    template<applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<XprType, b>;


    template<applicability b>
    static constexpr bool is_square = square_shaped<XprType, b>;


    static constexpr triangle_type triangle_type_value = triangle_type_of<XprType>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<XprType, applicability::permitted>;


    static constexpr data_layout layout = layout_of_v<XprType>;

  };

}

#endif
