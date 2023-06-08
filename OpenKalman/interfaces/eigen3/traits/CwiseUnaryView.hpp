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
  struct IndexTraits<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
  {
    template<std::size_t N>
    static constexpr std::size_t dimension = index_dimension_of_v<MatrixType, N>;

    template<std::size_t N, typename Arg>
    static constexpr std::size_t dimension_at_runtime(const Arg& arg)
    {
      return get_index_dimension_of<N>(arg.nestedExpression());
    }

    template<Likelihood b>
    static constexpr bool is_one_by_one = one_by_one_matrix<MatrixType, b>;

    template<Likelihood b>
    static constexpr bool is_square = square_matrix<MatrixType, b>;
  };


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


  template<typename UnaryOp, typename MatrixType>
  struct SingleConstant<Eigen::CwiseUnaryView<UnaryOp, MatrixType>>
  {
    const Eigen::CwiseUnaryView<UnaryOp, MatrixType>& xpr;

    constexpr auto get_constant()
    {
      return Eigen3::FunctorTraits<UnaryOp, MatrixType>::template get_constant<false>(xpr);
    }

    constexpr auto get_constant_diagonal()
    {
      return Eigen3::FunctorTraits<UnaryOp, MatrixType>::template get_constant<true>(xpr);
    }
  };


  template<typename UnaryOp, typename MatrixType>
  struct TriangularTraits<Eigen::CwiseUnaryView<UnaryOp, MatrixType>>
  {
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = Eigen3::FunctorTraits<UnaryOp, MatrixType>::template is_triangular<t, b>;

    static constexpr bool is_triangular_adapter = false;
  };


  template<typename UnaryOp, typename MatrixType>
  struct HermitianTraits<Eigen::CwiseUnaryView<UnaryOp, MatrixType>>
  {
    static constexpr bool is_hermitian = Eigen3::FunctorTraits<UnaryOp, MatrixType>::is_hermitian;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_CWISEUNARYVIEW_HPP
