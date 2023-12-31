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
 * \brief Type traits as applied to Eigen::TensorCwiseBinaryOp
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP


namespace OpenKalman::interface
{
  template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
  struct indexible_object_traits<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
    : Eigen3::indexible_object_traits_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<LhsXprType, n>)
          return OpenKalman::get_vector_space_descriptor(arg.lhsExpression(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.rhsExpression(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.lhsExpression(), n);
    }


    using dependents = std::tuple<typename LhsXprType::Nested, typename RhsXprType::Nested>;


    static constexpr bool has_runtime_parameters = false;


    // nested_object() not defined


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::TensorCwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsXprType>, equivalent_self_contained_t<RhsXprType>>;
      // Do a partial evaluation as long as at least one argument is already self-contained.
      if constexpr ((self_contained<LhsXprType> or self_contained<RhsXprType>) and
        not std::is_lvalue_reference_v<typename LhsXprType::Nested> and
        not std::is_lvalue_reference_v<typename RhsXprType::Nested>)
      {
        return N {make_self_contained(arg.lhsExpression()), make_self_contained(arg.rhsExpression()), arg.functor()};
      }
      else
      {
        return make_dense_object(std::forward<Arg>(arg));
      }
    }


    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsXprType, RhsXprType>::template get_constant<false>(arg);
    }


    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsXprType, RhsXprType>::template get_constant<true>(arg);
    }


    template<Qualification b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsXprType, Qualification::depends_on_dynamic_shape> and one_dimensional<RhsXprType, Qualification::depends_on_dynamic_shape> and
      (b != Qualification::unqualified or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        (square_shaped<LhsXprType, b> and (dimension_size_of_index_is<RhsXprType, 0, 1> or dimension_size_of_index_is<RhsXprType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsXprType, 0, 1> or dimension_size_of_index_is<LhsXprType, 1, 1>) and square_shaped<RhsXprType, b>));


    template<Qualification b>
    static constexpr bool is_square =
      square_shaped<LhsXprType, Qualification::depends_on_dynamic_shape> and square_shaped<RhsXprType, Qualification::depends_on_dynamic_shape> and
      (b != Qualification::unqualified or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        square_shaped<LhsXprType, b> or square_shaped<RhsXprType, b>);


    template<TriangleType t, Qualification b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<BinaryOp, LhsXprType, RhsXprType>::template is_triangular<t, b>;


    static constexpr bool is_triangular_adapter = false;


    static constexpr bool is_hermitian = Eigen3::FunctorTraits<BinaryOp, LhsXprType, RhsXprType>::is_hermitian;


    static constexpr bool is_writable = false;


    // raw_data() not defined


    // layout not defined

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP
