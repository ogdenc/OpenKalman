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
 * \brief Type traits as applied to Eigen::Reverse.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_REVERSE_HPP
#define OPENKALMAN_EIGEN_TRAITS_REVERSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType, int Direction>
  struct indexible_object_traits<Eigen::Reverse<MatrixType, Direction>>
    : Eigen3::indexible_object_traits_base<Eigen::Reverse<MatrixType, Direction>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::Reverse<MatrixType, Direction>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }

    using dependents = std::tuple<typename MatrixType::Nested>;

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
      if constexpr (not std::is_lvalue_reference_v<typename M::Nested>)
        return Eigen::Reverse<M, Direction> {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_object(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return constant_coefficient {arg.nestedExpression()};
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (Direction == Eigen::BothDirections) return constant_diagonal_coefficient {arg.nestedExpression()};
      else return std::monostate {};
    }

    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;

    template<Qualification b>
    static constexpr bool is_square = square_shaped<MatrixType, b>;

    template<TriangleType t>
    static constexpr bool is_triangular = triangular_matrix<MatrixType,
        t == TriangleType::upper ? TriangleType::lower :
        t == TriangleType::lower ? TriangleType::upper : t> and
      (Direction == Eigen::BothDirections or (Direction == Eigen::Horizontal and vector<MatrixType, 0>) or
        (Direction == Eigen::Vertical and vector<MatrixType, 1>));

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, Qualification::depends_on_dynamic_shape> and
        (Direction == Eigen::BothDirections or OpenKalman::one_dimensional<MatrixType, Qualification::depends_on_dynamic_shape>);
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_REVERSE_HPP
