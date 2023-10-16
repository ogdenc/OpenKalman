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
 * \brief Traits for Eigen::CwiseUnaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename UnaryOp, typename XprType>
  struct indexible_object_traits<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
  private:

    using Xpr = Eigen::CwiseUnaryOp<UnaryOp, XprType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

    template<typename T>
    struct is_bind_operator : std::false_type {};

    template<typename BinaryOp>
    struct is_bind_operator<Eigen::internal::bind1st_op<BinaryOp>> : std::true_type {};

    template<typename BinaryOp>
    struct is_bind_operator<Eigen::internal::bind2nd_op<BinaryOp>> : std::true_type {};

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }

    using type = std::tuple<typename Xpr::XprTypeNested>;

    static constexpr bool has_runtime_parameters = is_bind_operator<UnaryOp>::value;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 1);
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseUnaryOp<UnaryOp, equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
        return N {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<true>(arg);
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<XprType, b>;

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<UnaryOp, XprType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<UnaryOp, XprType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP
