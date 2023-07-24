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

#ifndef OPENKALMAN_EIGEN3_TRAITS_REVERSE_HPP
#define OPENKALMAN_EIGEN3_TRAITS_REVERSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType, int Direction>
  struct IndexibleObjectTraits<Eigen::Reverse<MatrixType, Direction>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Reverse<MatrixType, Direction>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<MatrixType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return OpenKalman::get_index_descriptor<N>(arg.nestedExpression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<MatrixType, b>;

    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename MatrixType::Nested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using M = equivalent_self_contained_t<MatrixType>;
      if constexpr (not std::is_lvalue_reference_v<typename M::Nested>)
        return Eigen::Reverse<M, Direction> {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
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

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<MatrixType,
        t == TriangleType::upper ? TriangleType::lower :
        t == TriangleType::lower ? TriangleType::upper : t, b> and
      (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType>);

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, Likelihood::maybe> and
        (Direction == Eigen::BothDirections or one_by_one_matrix<MatrixType, Likelihood::maybe>);
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_REVERSE_HPP
