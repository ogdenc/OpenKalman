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
  namespace EGI = Eigen::internal;


  namespace detail
  {
    template<typename NullaryOp, typename PlainObjectType>
    struct CwiseNullaryOpDependenciesBase
    {
      static constexpr bool has_runtime_parameters =
        Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::RowsAtCompileTime == Eigen::Dynamic or
        Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>::ColsAtCompileTime == Eigen::Dynamic;

      using type = std::tuple<>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).functor();
      }
    };
  }

  template<typename NullaryOp, typename PlainObjectType>
  struct Dependencies<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
    : detail::CwiseNullaryOpDependenciesBase<NullaryOp, PlainObjectType> {};


  template<typename Scalar, typename PlainObjectType>
  struct Dependencies<Eigen::CwiseNullaryOp<EGI::scalar_constant_op<Scalar>, PlainObjectType>>
    : detail::CwiseNullaryOpDependenciesBase<EGI::scalar_constant_op<Scalar>, PlainObjectType>
  {
    static constexpr bool has_runtime_parameters = true;
  };


  template<typename PlainObjectType, typename...Args>
  struct Dependencies<Eigen::CwiseNullaryOp<EGI::linspaced_op<Args...>, PlainObjectType>>
    : detail::CwiseNullaryOpDependenciesBase<EGI::linspaced_op<Args...>, PlainObjectType>
  {
    static constexpr bool has_runtime_parameters = true;
  };


  template<typename NullaryOp, typename PlainObjectType>
  struct SingleConstant<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
    const Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::template get_constant<constant_coefficient>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::template get_constant<constant_diagonal_coefficient>(xpr);
    }
  };


  template<typename NullaryOp, typename PlainObjectType>
  struct DiagonalTraits<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
    static constexpr bool is_diagonal = Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::is_diagonal;
  };


  template<typename NullaryOp, typename PlainObjectType>
  struct TriangularTraits<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
    static constexpr TriangleType triangle_type = Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::triangle_type;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename NullaryOp, typename PlainObjectType>
  struct HermitianTraits<Eigen::CwiseNullaryOp<NullaryOp, PlainObjectType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<NullaryOp, PlainObjectType>::is_hermitian;

    static constexpr TriangleType adapter_type = TriangleType::none;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISENULLARYOP_HPP
