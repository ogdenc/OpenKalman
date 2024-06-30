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
 * \brief Traits for Eigen::CwiseUnaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP


namespace OpenKalman::interface
{
  template<typename UnaryOp, typename XprType>
  struct indexible_object_traits<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
  {
  private:

    using Xpr = Eigen::CwiseUnaryOp<UnaryOp, XprType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;
    using Traits = Eigen3::UnaryFunctorTraits<std::decay_t<UnaryOp>>;

    template<typename T>
    struct is_bind_operator : std::false_type {};

    template<typename BinaryOp>
    struct is_bind_operator<Eigen::internal::bind1st_op<BinaryOp>> : std::true_type {};

    template<typename BinaryOp>
    struct is_bind_operator<Eigen::internal::bind2nd_op<BinaryOp>> : std::true_type {};

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_vector_space_descriptor(const Arg& arg, N n)
    {
      return OpenKalman::get_vector_space_descriptor(arg.nestedExpression(), n);
    }


    using dependents = std::tuple<typename Xpr::XprTypeNested>;


    static constexpr bool has_runtime_parameters = is_bind_operator<UnaryOp>::value;


    template<typename Arg>
    static decltype(auto)
    nested_object(Arg&& arg)
    {
      return std::as_const(arg).nestedExpression(); // There seems to be a bug in CwiseUnaryOp when the argument is non-const.
    }


    template<typename Arg>
    static auto
    convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseUnaryOp<UnaryOp, equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
        return make_self_contained(nested_object(std::forward<Arg>(arg))).unaryExpr(arg.functor());
      else
        return make_dense_object(std::forward<Arg>(arg));
    }

  private:

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct custom_get_constant_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_defined<T, std::void_t<decltype(T::get_constant(std::declval<const Xpr&>()))>>
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
        return Traits::get_constant(arg);
      else if constexpr (Eigen3::constexpr_unary_operation_defined<UnaryOp>)
        return values::scalar_constant_operation {Traits::constexpr_operation(), constant_coefficient {arg.nestedExpression()}};
      else
        return values::scalar_constant_operation {arg.functor(), constant_coefficient {arg.nestedExpression()}};
    }

  private:

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct custom_get_constant_diagonal_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_diagonal_defined<T, std::void_t<decltype(T::get_constant_diagonal(std::declval<const Xpr&>()))>>
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
        return Traits::get_constant_diagonal(arg);
      else if constexpr (not Traits::preserves_triangle)
        return std::monostate{};
      else if constexpr (Eigen3::constexpr_unary_operation_defined<UnaryOp>)
        return values::scalar_constant_operation {Traits::constexpr_operation(), constant_diagonal_coefficient{arg.nestedExpression()}};
      else
        return values::scalar_constant_operation {arg.functor(), constant_diagonal_coefficient{arg.nestedExpression()}};
    }


    template<Qualification b>
    static constexpr bool one_dimensional = OpenKalman::one_dimensional<XprType, b>;


    template<Qualification b>
    static constexpr bool is_square = square_shaped<XprType, b>;


    static constexpr bool is_triangular_adapter = false;


    template<TriangleType t>
    static constexpr bool is_triangular = Traits::preserves_triangle and triangular_matrix<XprType, t>;


    static constexpr bool is_hermitian = Traits::preserves_hermitian and hermitian_matrix<XprType, Qualification::depends_on_dynamic_shape>;

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEUNARYOP_HPP
