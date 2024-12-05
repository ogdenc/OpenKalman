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

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP

#include <type_traits>


namespace OpenKalman::interface
{
  template<typename ViewOp, typename MatrixType>
  struct indexible_object_traits<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseUnaryView<ViewOp, MatrixType>>
  {
  private:

    using Xpr = Eigen::CwiseUnaryView<ViewOp, MatrixType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;
    using Traits = Eigen3::UnaryFunctorTraits<std::decay_t<ViewOp>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }


    template<typename Arg>
    static decltype(auto) nested_object(Arg&& arg)
    {
      return std::forward<Arg>(arg).nestedExpression();
    }

  private:

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct custom_get_constant_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_defined<T, std::void_t<decltype(T::template get_constant(std::declval<const Xpr&>()))>>
      : std::true_type {};
#endif

  public:

    static constexpr auto
    get_constant(const Xpr& arg)
    {
#ifdef __cpp_concepts
      if constexpr (requires { Traits::get_constant(arg); })
#else
      if constexpr (custom_get_constant_defined<Traits>::value)
#endif
        return Traits::template get_constant(arg);
#ifdef __cpp_concepts
      else if constexpr (Eigen3::constexpr_unary_operation_defined<ViewOp>)
        return value::operation {Traits::constexpr_operation(), constant_coefficient {arg.nestedExpression()}};
#else
      else if constexpr (Eigen3::constexpr_unary_operation_defined<ViewOp> and value::fixed<constant_coefficient<MatrixType>>)
        return value::operation {Traits::constexpr_operation(), constant_coefficient<MatrixType>{}};
#endif
      else
        return value::operation {arg.functor(), constant_coefficient {arg.nestedExpression()}};
    }

  private:

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct custom_get_constant_diagonal_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_diagonal_defined<T, std::void_t<decltype(T::template get_constant_diagonal<MatrixType>(std::declval<const Xpr&>()))>>
      : std::true_type {};
#endif

  public:

    static constexpr auto
    get_constant_diagonal(const Xpr& arg)
    {
#ifdef __cpp_concepts
      if constexpr (requires { Traits::get_constant_diagonal(arg); })
#else
      if constexpr (custom_get_constant_diagonal_defined<Traits>::value)
#endif
        return Traits::template get_constant_diagonal<MatrixType>(arg);
      else if constexpr (not Traits::preserves_triangle)
        return std::monostate{};
      else if constexpr (Eigen3::constexpr_unary_operation_defined<ViewOp>)
        return value::operation {Traits::constexpr_operation(), constant_diagonal_coefficient{arg.nestedExpression()}};
      else
        return value::operation {arg.functor(), constant_diagonal_coefficient{arg.nestedExpression()}};
    }


    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<MatrixType, b>;


    template<Qualification b>
    static constexpr bool is_square = square_shaped<MatrixType, b>;


    static constexpr bool is_triangular_adapter = false;


    template<TriangleType t>
    static constexpr bool is_triangular = Traits::preserves_triangle and triangular_matrix<MatrixType, t>;


    static constexpr bool is_hermitian = Traits ::preserves_hermitian and hermitian_matrix<MatrixType, Qualification::depends_on_dynamic_shape>;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEUNARYVIEW_HPP
