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
 * \brief Traits for Eigen::CwiseNullaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISENULLARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISENULLARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename NullaryOp, typename PlainObjectType>
  struct indexible_object_traits<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_base<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>;

    template<typename T>
    struct has_params : std::bool_constant<
      Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::RowsAtCompileTime == Eigen::Dynamic or
      Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::ColsAtCompileTime == Eigen::Dynamic> {};

    template<typename Scalar>
    struct has_params<Eigen::internal::scalar_constant_op<Scalar>> : std::true_type {};

    template<typename...Args>
    struct has_params<Eigen::internal::linspaced_op<Args...>> : std::true_type {};

  public:

    using dependents = std::tuple<>;

    static constexpr bool has_runtime_parameters = has_params<NullaryOp>::value;

    // get_nested_matrix not defined

    // convert_to_self_contained not defined

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::NullaryFunctorTraits<NullaryOp, PlainObjectType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::NullaryFunctorTraits<NullaryOp, PlainObjectType>::template get_constant<true>(arg);
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<PlainObjectType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<PlainObjectType, b>;

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::NullaryFunctorTraits<NullaryOp, PlainObjectType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::NullaryFunctorTraits<NullaryOp, PlainObjectType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISENULLARYOP_HPP
