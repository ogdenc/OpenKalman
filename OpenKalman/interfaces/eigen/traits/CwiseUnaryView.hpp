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
  struct indexible_object_traits<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::CwiseUnaryView<ViewOp, MatrixType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }

    using dependents = std::tuple<typename Eigen::CwiseUnaryView<ViewOp, MatrixType>::MatrixTypeNested, ViewOp>;

    static constexpr bool has_runtime_parameters = false;


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseUnaryView<ViewOp, equivalent_self_contained_t<MatrixType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::MatrixTypeNested>)
        return N {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_object(std::forward<Arg>(arg));
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

    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;

    template<Qualification b>
    static constexpr bool is_square = square_shaped<MatrixType, b>;

    template<TriangleType t, Qualification b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<ViewOp, MatrixType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<ViewOp, MatrixType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP
