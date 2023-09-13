/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Traits for Eigen::CwiseUnaryView.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ViewOp, typename MatrixType>
  struct IndexibleObjectTraits<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
  {
    template<typename Arg, typename N>
    static constexpr auto get_index_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_index_descriptor(arg.nestedExpression(), n);
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<MatrixType, b>;

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename Eigen::CwiseUnaryView<ViewOp, MatrixType>::MatrixTypeNested, ViewOp>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      if constexpr (i == 0)
        return std::forward<Arg>(arg).nestedExpression();
      else
        return std::forward<Arg>(arg).functor();
      static_assert(i <= 1);
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseUnaryView<ViewOp, equivalent_self_contained_t<MatrixType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::MatrixTypeNested>)
        return N {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<ViewOp, MatrixType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<ViewOp, MatrixType>::template get_constant<true>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<ViewOp, MatrixType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<ViewOp, MatrixType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP
