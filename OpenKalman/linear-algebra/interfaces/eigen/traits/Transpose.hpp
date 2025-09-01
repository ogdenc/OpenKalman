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
 * \brief Type traits as applied to Eigen::Transpose.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TRANSPOSE_HPP
#define OPENKALMAN_EIGEN_TRAITS_TRANSPOSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType>
  struct indexible_object_traits<Eigen::Transpose<MatrixType>>
    : Eigen3::indexible_object_traits_base<Eigen::Transpose<MatrixType>>
  {
  private:

    using Xpr = Eigen::Transpose<MatrixType>;
    using Base = Eigen3::indexible_object_traits_base<Eigen::Transpose<MatrixType>>;

  public:

    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient{arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return constant_diagonal_coefficient {arg.nestedExpression()};
    }


    template<applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;


    template<applicability b>
    static constexpr bool is_square = square_shaped<MatrixType, b>;


    template<triangle_type t>
    static constexpr bool is_triangular = diagonal_matrix<MatrixType> or
      (t == triangle_type::lower and triangular_matrix<MatrixType, triangle_type::upper>) or
      (t == triangle_type::upper and triangular_matrix<MatrixType, triangle_type::lower>);


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, applicability::permitted>;


    static constexpr data_layout layout = layout_of_v<MatrixType>;

  };


}

#endif
