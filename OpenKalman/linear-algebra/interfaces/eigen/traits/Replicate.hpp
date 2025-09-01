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
  struct indexible_object_traits<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : Eigen3::indexible_object_traits_base<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
  {
  private:

    using Xpr = Eigen::Replicate<MatrixType, RowFactor, ColFactor>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

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
      return constant_coefficient {arg.nestedExpression()};
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (RowFactor == 1 and ColFactor == 1) return constant_diagonal_coefficient {arg.nestedExpression()};
      else return std::monostate {};
    }


    template<applicability b>
    static constexpr bool one_dimensional =
      (b != applicability::guaranteed or (RowFactor == 1 and ColFactor == 1)) and
      (RowFactor == 1 or RowFactor == Eigen::Dynamic) and
      (ColFactor == 1 or ColFactor == Eigen::Dynamic) and
        OpenKalman::one_dimensional<MatrixType, b>;


    template<applicability b>
    static constexpr bool is_square =
      (b != applicability::guaranteed or not has_dynamic_dimensions<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>) and
      (RowFactor == Eigen::Dynamic or ColFactor == Eigen::Dynamic or
        ((RowFactor != ColFactor or square_shaped<MatrixType, b>) and
        (dynamic_dimension<MatrixType, 0> or RowFactor * index_dimension_of_v<MatrixType, 0> % ColFactor == 0) and
        (dynamic_dimension<MatrixType, 1> or ColFactor * index_dimension_of_v<MatrixType, 1> % RowFactor == 0))) and
      (has_dynamic_dimensions<MatrixType> or
        ((RowFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 0> * RowFactor % index_dimension_of_v<MatrixType, 1> == 0) and
        (ColFactor == Eigen::Dynamic or index_dimension_of_v<MatrixType, 1> * ColFactor % index_dimension_of_v<MatrixType, 0> == 0)));


    template<triangle_type t>
    static constexpr bool is_triangular = triangular_matrix<MatrixType, t> and
      ((RowFactor == 1 and ColFactor != 0 and (t == triangle_type::upper or t == triangle_type::any)) or
        (ColFactor == 1 and RowFactor != 0 and (t == triangle_type::lower or t == triangle_type::any)) or
        (RowFactor == 1 and ColFactor == 1 and t == triangle_type::diagonal));


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, applicability::permitted> and
      ((RowFactor == 1 and ColFactor == 1) or not values::complex<scalar_type> or
        values::not_complex<constant_coefficient<MatrixType>> or values::not_complex<constant_diagonal_coefficient<MatrixType>>);

  };

}

#endif
