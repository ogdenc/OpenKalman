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
 * \brief Type traits as applied to Eigen::TensorCwiseUnaryOp
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORCWISEUNARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORCWISEUNARYOP_HPP


namespace OpenKalman::interface
{
  template<typename NullaryOp, typename XprType>
  struct indexible_object_traits<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>
    : Eigen3::indexible_object_traits_base<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }


    using type = std::conditional_t<has_dynamic_dimensions<XprType>, std::tuple<typename XprType::Nested>, std::tuple<>>;


    static constexpr bool has_runtime_parameters = true;


    // get_nested_matrix not defined

#ifdef __cpp_concepts
    template<std::size_t i, typename Arg> requires has_dynamic_dimensions<XprType>
#else
    template<std::size_t i, typename X = XprType, typename Arg, std::enable_if_t<has_dynamic_dimensions<X>, int> = 0>
#endif
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }


    // convert_to_self_contained not defined


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::template get_constant<false>(arg);
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::template get_constant<true>(arg);
    }


    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;


    template<Likelihood b>
    static constexpr bool is_square = square_matrix<XprType, b>;


    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::template is_triangular<t, b>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::is_hermitian;


    static constexpr bool is_writable = false;


    // data() not defined


    // layout not defined

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCWISEUNARYOP_HPP
