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

#ifndef OPENKALMAN_EIGEN3_TRAITS_CWISENULLARYOP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_CWISENULLARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename NullaryOp, typename PlainObjectType>
  struct IndexibleObjectTraits<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
    : Eigen3::IndexibleObjectTraitsBase<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
  private:

    template<typename T>
    struct has_params : std::bool_constant<
      Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::RowsAtCompileTime == Eigen::Dynamic or
      Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::ColsAtCompileTime == Eigen::Dynamic> {};

    template<typename Scalar>
    struct has_params<Eigen::internal::scalar_constant_op<Scalar>> : std::true_type {};

    template<typename...Args>
    struct has_params<Eigen::internal::linspaced_op<Args...>> : std::true_type {};

  public:

    static constexpr std::size_t max_indices = max_indices_of_v<PlainObjectType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      constexpr Eigen::Index dim = N == 0 ? PlainObjectType::RowsAtCompileTime : PlainObjectType::ColsAtCompileTime;
      if constexpr (dim == Eigen::Dynamic)
      {
        if constexpr (N == 0) return static_cast<std::size_t>(arg.rows());
        else return static_cast<std::size_t>(arg.cols());
      }
      else return Dimensions<dim>{};
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<PlainObjectType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<PlainObjectType, b>;

    static constexpr bool has_runtime_parameters = has_params<NullaryOp>::value;

    using type = std::tuple<>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).functor();
    }

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::template get_constant<false>(arg);
    }

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      return Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::template get_constant<true>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;

    static constexpr bool is_hermitian = Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISENULLARYOP_HPP
