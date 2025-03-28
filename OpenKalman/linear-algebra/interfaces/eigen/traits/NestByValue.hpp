/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::NestByValue.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_NESTBYVALUE_HPP
#define OPENKALMAN_EIGEN_TRAITS_NESTBYVALUE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ExpressionType>
  struct indexible_object_traits<Eigen::NestByValue<ExpressionType>>
    : Eigen3::indexible_object_traits_base<Eigen::NestByValue<ExpressionType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::NestByValue<ExpressionType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }


    template<typename Arg>
    static const ExpressionType& nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {arg.nestedExpression()};
    }


    template<Applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<ExpressionType, b>;


    template<Applicability b>
    static constexpr bool is_square = square_shaped<ExpressionType, b>;


    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<ExpressionType, t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<ExpressionType, Applicability::permitted>;


    static constexpr Layout layout = layout_of_v<ExpressionType>;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_NESTBYVALUE_HPP
