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
 * \brief Traits for Eigen::CwiseBinaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP

#include <type_traits>

namespace OpenKalman::interface
{
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct indexible_object_traits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
  private:

    using Xpr = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;

  public:

    template<typename Arg, typename N>
    static constexpr auto \
    get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<LhsType, n>)
          return OpenKalman::get_vector_space_descriptor(arg.lhs(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.rhs(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.lhs(), n);
    }


    using dependents = std::tuple<typename Xpr::LhsNested, typename Xpr::RhsNested>;


    static constexpr bool has_runtime_parameters = false;


    // nested_object() not defined


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>>;
      // Do a partial evaluation as long as at least one argument is already self-contained.
      if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
        not std::is_lvalue_reference_v<typename N::LhsNested> and
        not std::is_lvalue_reference_v<typename N::RhsNested>)
      {
        return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs()), arg.functor()};
      }
      else
      {
        return make_dense_object(std::forward<Arg>(arg));
      }
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template get_constant<true>(arg);
    }

    template<Likelihood b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsType, Likelihood::maybe> and OpenKalman::one_dimensional<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        (square_shaped<LhsType, b> and (dimension_size_of_index_is<RhsType, 0, 1> or dimension_size_of_index_is<RhsType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsType, 0, 1> or dimension_size_of_index_is<LhsType, 1, 1>) and square_shaped<RhsType, b>));

    template<Likelihood b>
    static constexpr bool is_square =
      square_shaped<LhsType, Likelihood::maybe> and square_shaped<RhsType, Likelihood::maybe> and
      (b != Likelihood::definitely or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        square_shaped<LhsType, b> or square_shaped<RhsType, b>);

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<BinaryOp, LhsType, RhsType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP
