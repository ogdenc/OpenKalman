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

    using Base = Eigen3::indexible_object_traits_base<Eigen::Transpose<MatrixType>>;

  public:

    using dependents = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using M = equivalent_self_contained_t<MatrixType>;
      using N = Eigen::Transpose<M>;
      if constexpr (not std::is_lvalue_reference_v<typename Eigen::internal::ref_selector<M>::non_const_type>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_object(std::forward<Arg>(arg));
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

    template<Likelihood b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_shaped<MatrixType, b>;

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = diagonal_matrix<MatrixType, b> or
      (t == TriangleType::lower and triangular_matrix<MatrixType, TriangleType::upper, b>) or
      (t == TriangleType::upper and triangular_matrix<MatrixType, TriangleType::lower, b>);

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, Likelihood::maybe>;

    static constexpr Layout layout = layout_of_v<MatrixType>;

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TRANSPOSE_HPP
