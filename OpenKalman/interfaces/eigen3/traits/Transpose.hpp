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

#ifndef OPENKALMAN_EIGEN3_TRAITS_TRANSPOSE_HPP
#define OPENKALMAN_EIGEN3_TRAITS_TRANSPOSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename MatrixType>
  struct IndexibleObjectTraits<Eigen::Transpose<MatrixType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::Transpose<MatrixType>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<MatrixType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return OpenKalman::get_index_descriptor<N == 0 ? 1 : 0>(arg.nestedExpression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<MatrixType, b>;

    static constexpr bool has_runtime_parameters = false;

    using type = std::tuple<typename Eigen::internal::ref_selector<MatrixType>::non_const_type>;

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
      using N = Eigen::Transpose<M>;
      if constexpr (not std::is_lvalue_reference_v<typename Eigen::internal::ref_selector<M>::non_const_type>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
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

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = diagonal_matrix<MatrixType, b> or
      (t == TriangleType::lower and triangular_matrix<MatrixType, TriangleType::upper, b>) or
      (t == TriangleType::upper and triangular_matrix<MatrixType, TriangleType::lower, b>);

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = hermitian_matrix<MatrixType, Likelihood::maybe>;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_TRANSPOSE_HPP
