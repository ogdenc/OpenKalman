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

#ifndef OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYVIEW_HPP
#define OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYVIEW_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  namespace EGI = Eigen::internal;


  template<typename ViewOp, typename MatrixType>
  struct Dependencies<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
  {
  private:

    using T = Eigen::CwiseUnaryView<ViewOp, MatrixType>;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::MatrixTypeNested, ViewOp>;

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
      using N = Eigen::CwiseUnaryView<ViewOp, equivalent_self_contained_t<MatrixType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::MatrixTypeNested>)
        return N {make_self_contained(arg.nestedExpression()), arg.functor()};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  };


  template<typename UnaryOp, typename XprType>
  struct SingleConstant<Eigen::CwiseUnaryView<UnaryOp, XprType>>
  {
    const Eigen::CwiseUnaryView<UnaryOp, XprType>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<constant_coefficient>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<UnaryOp, XprType>::template get_constant<constant_diagonal_coefficient>(xpr);
    }
  };


  template<typename UnaryOp, typename XprType>
  struct DiagonalTraits<Eigen::CwiseUnaryView<UnaryOp, XprType>>
  {
    static constexpr bool is_diagonal = Eigen3::FunctorTraits<UnaryOp, XprType>::is_diagonal;
  };


  template<typename UnaryOp, typename XprType>
  struct TriangularTraits<Eigen::CwiseUnaryView<UnaryOp, XprType>>
  {
    static constexpr TriangleType triangle_type = Eigen3::FunctorTraits<UnaryOp, XprType>::triangle_type;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename UnaryOp, typename XprType>
  struct HermitianTraits<Eigen::CwiseUnaryView<UnaryOp, XprType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<UnaryOp, XprType>::is_hermitian;

    static constexpr TriangleType adapter_type = TriangleType::none;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYVIEW_HPP
