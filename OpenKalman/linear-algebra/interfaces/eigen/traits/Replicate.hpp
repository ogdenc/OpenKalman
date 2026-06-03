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
 * \brief Type traits as applied to Eigen::Replicate.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_REPLICATE_HPP
#define OPENKALMAN_EIGEN_TRAITS_REPLICATE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType, int RowFactor, int ColFactor>
  struct object_traits<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : Eigen3::object_traits_base<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
  {
  private:

    using Xpr = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;
    using Base = Eigen3::object_traits_base<Xpr>;

  public:

    using typename Base::scalar_type;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_value {arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (RowFactor == 1 and ColFactor == 1) return constant_diagonal_value {arg.nestedExpression()};
      else return std::monostate {};
    }


    template<applicability b>
    static constexpr bool one_dimensional =
      (b != applicability::guaranteed or (RowFactor == 1 and ColFactor == 1)) and
      (RowFactor == 1 or RowFactor == Eigen::Dynamic) and
      (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
        OpenKalman::one_dimensional<MatrixType, b>;


    template<std::size_t N, applicability b>
    static constexpr bool is_square = N == 2 and
      (b != applicability::guaranteed or not has_dynamic_dimensions<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>) and
      (RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic or
        ((RowFactor != ColFactor or square_shaped<MatrixType, b>) and
        (dynamic_dimension<MatrixType, 0> or RowFactor * index_dimension_of_v<MatrixType, 0> % ColFactor == 0) and
        (dynamic_dimension<MatrixType, 1> or ColFactor * index_dimension_of_v<MatrixType, 1> % RowFactor == 0))) and
      (has_dynamic_dimensions<MatrixType> or
        ((RowFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 0> * RowFactor % index_dimension_of_v<MatrixType, 1> == 0) and
        (ColFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 1> * ColFactor % index_dimension_of_v<MatrixType, 0> == 0)));


    static constexpr triangle_type
    triangle_type_value =
      triangular_matrix<MatrixType, triangle_type::diagonal> and RowFactor == 1 and ColFactor == 1 ? triangle_type::diagonal :
      triangular_matrix<MatrixType, triangle_type::upper> and RowFactor == 1 ? triangle_type::upper :
      triangular_matrix<MatrixType, triangle_type::lower> and ColFactor == 1 ? triangle_type::upper :
      triangle_type::none;


    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, applicability::permitted> and
      ((RowFactor == 1 and ColFactor == 1) or not values::complex<scalar_type> or
        values::not_complex<constant_value<MatrixType>> or values::not_complex<constant_diagonal_value<MatrixType>>);

  };

}

#endif
