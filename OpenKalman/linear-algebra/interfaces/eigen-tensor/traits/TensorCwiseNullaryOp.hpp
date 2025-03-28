/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::TensorCwiseNullaryOp
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORCWISENULLARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORCWISENULLARYOP_HPP


namespace OpenKalman::interface
{
  template<typename NullaryOp, typename XprType>
  struct indexible_object_traits<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>
    : Eigen3::indexible_object_traits_tensor_base<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_tensor_base<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }

#ifdef __cpp_concepts
    template<typename Arg> requires has_dynamic_dimensions<XprType>
#else
    template<typename X = XprType, typename Arg, std::enable_if_t<has_dynamic_dimensions<X>, int> = 0>
#endif
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }


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


    template<Applicability b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<XprType, b>;


    template<Applicability b>
    static constexpr bool is_square = square_shaped<XprType, b>;


    template<TriangleType t>
    static constexpr bool is_triangular = Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::template is_triangular<t>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = Eigen3::NullaryFunctorTraits<NullaryOp, XprType>::is_hermitian;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCWISENULLARYOP_HPP
