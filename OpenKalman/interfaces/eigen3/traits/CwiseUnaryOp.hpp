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

#ifndef OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYOP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYOP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  namespace EGI = Eigen::internal;


  template<typename UnaryOp, typename XprType>
  struct IndexTraits<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
    static constexpr std::size_t max_indices = max_indices_of_v<XprType>;

    template<std::size_t N, typename Arg>
    static constexpr auto get_index_descriptor(const Arg& arg)
    {
      return OpenKalman::get_index_descriptor<N>(arg.nestedExpression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<XprType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<XprType, b>;
  };


  template<typename UnaryOp, typename XprType>
  struct Dependencies<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
  private:

    using T = Eigen::CwiseUnaryOp<UnaryOp, XprType>;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::XprTypeNested>;

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
      using N = Eigen::CwiseUnaryOp<UnaryOp, equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
        return N {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  };


  template<typename BinaryOp, typename XprType>
  struct Dependencies<Eigen::CwiseUnaryOp<EGI::bind1st_op<BinaryOp>, XprType>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type =
      std::tuple<typename Eigen::CwiseUnaryOp<EGI::bind1st_op<BinaryOp>, XprType>::XprTypeNested>;
  };


  template<typename BinaryOp, typename XprType>
  struct Dependencies<Eigen::CwiseUnaryOp<EGI::bind2nd_op<BinaryOp>, XprType>>
  {
    static constexpr bool has_runtime_parameters = true;
    using type =
      std::tuple<typename Eigen::CwiseUnaryOp<EGI::bind2nd_op<BinaryOp>, XprType>::XprTypeNested>;
  };


  template<typename UnaryOp, typename XprType>
  struct SingleConstant<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
    const Eigen::CwiseUnaryOp<UnaryOp, XprType>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<false>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<true>(xpr);
    }
  };


  template<typename UnaryOp, typename XprType>
  struct TriangularTraits<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<UnaryOp, XprType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename UnaryOp, typename XprType>
  struct HermitianTraits<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<UnaryOp, XprType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYOP_HPP
