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
 * \brief Traits for Eigen::TensorCwiseBinaryOp
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP


namespace OpenKalman::interface
{
  template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
  struct object_traits<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
    : Eigen3::object_traits_tensor_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
  {
  private:

    using Xpr = Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>;
    using Base = Eigen3::object_traits_tensor_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>;
    using Traits = Eigen3::BinaryFunctorTraits<std::decay_t<BinaryOp>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_pattern_collection(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
      {
        if constexpr (not dynamic_dimension<LhsXprType, n>)
          return OpenKalman::get_pattern_collection(arg.lhsExpression(), n);
        else
          return OpenKalman::get_pattern_collection(arg.rhsExpression(), n);
      }
      else return OpenKalman::get_pattern_collection(arg.lhsExpression(), n);
    }


    // nested_object() not defined

  private:

#ifndef __cpp_concepts
    template<typename T = Traits, typename = void>
    struct custom_get_constant_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_defined<T, std::void_t<decltype(T::template get_constant<LhsXprType, RhsXprType>(std::declval<const Xpr&>()))>>
      : std::true_type {};

    template<typename T = Traits, typename = void>
    struct constexpr_operation_defined : std::false_type {};

    template<typename T>
    struct constexpr_operation_defined<T, std::void_t<decltype(T::constexpr_operation())>>
      : std::true_type {};
#endif

  public:

    template<typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
#ifdef __cpp_concepts
      if constexpr (requires { Traits::template get_constant<LhsXprType, RhsXprType>(arg); })
#else
        if constexpr (custom_get_constant_defined<>::value)
#endif
        return Traits::template get_constant<LhsXprType, RhsXprType>(arg);
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<LhsXprType>)
        return constant_value {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsXprType>)
        return constant_value {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsXprType>)
        return constant_value {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsXprType>)
        return constant_value {arg.rhsExpression()};
#ifdef __cpp_concepts
      else if constexpr (requires { Traits::constexpr_operation(); })
#else
        else if constexpr (constexpr_operation_defined<>::value)
#endif
        return values::operation(Traits::constexpr_operation(),
          constant_value {arg.lhsExpression()}, constant_value {arg.rhsExpression()});
      else
        return values::operation(arg.functor(),
          constant_value {arg.lhsExpression()}, constant_value {arg.rhsExpression()});
    }

  private:

#ifndef __cpp_concepts
    template<typename T = Traits, typename = void>
    struct custom_get_constant_diagonal_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_diagonal_defined<T, std::void_t<decltype(T::template get_constant_diagonal<LhsXprType, RhsXprType>(std::declval<const Xpr&>()))>>
      : std::true_type {};
#endif

  public:

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
#ifdef __cpp_concepts
      if constexpr (requires { Traits::template get_constant_diagonal<LhsXprType, RhsXprType>(arg); })
#else
      if constexpr (custom_get_constant_diagonal_defined<>::value)
#endif
        return Traits::template get_constant_diagonal<LhsXprType, RhsXprType>(arg);
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<LhsXprType>)
        return constant_diagonal_value {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsXprType>)
        return constant_diagonal_value {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsXprType>)
        return constant_value {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsXprType>)
        return constant_value {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product)
      {
        auto c_left = [](const Arg& arg){
            if constexpr (constant_matrix<LhsXprType>) return constant_value {arg.lhsExpression()};
            else return constant_diagonal_value {arg.lhsExpression()};
        }(arg);
        auto c_right = [](const Arg& arg){
            if constexpr (constant_matrix<RhsXprType>) return constant_value {arg.rhsExpression()};
            else return constant_diagonal_value {arg.rhsExpression()};
        }(arg);
#ifdef __cpp_concepts
        if constexpr (requires { Traits::constexpr_operation(); })
#else
        if constexpr (constexpr_operation_defined<>::value)
#endif
          return values::operation(Traits::constexpr_operation(), c_left, c_right);
        else
          return values::operation(arg.functor(), c_left, c_right);
      }
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum or Traits::preserves_constant_diagonal)
      {
#ifdef __cpp_concepts
        if constexpr (requires { Traits::constexpr_operation(); })
#else
        if constexpr (constexpr_operation_defined<>::value)
#endif
          return values::operation(Traits::constexpr_operation(),
            constant_diagonal_value {arg.lhsExpression()}, constant_diagonal_value {arg.rhsExpression()});
        else
          return values::operation(arg.functor(),
            constant_diagonal_value {arg.lhsExpression()}, constant_diagonal_value {arg.rhsExpression()});
      }
      else
      {
        return std::monostate{};
      }
    }


    template<applicability b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsXprType, values::unbounded_size, applicability::permitted> and
      OpenKalman::one_dimensional<RhsXprType, values::unbounded_size, applicability::permitted> and
      (b != applicability::guaranteed or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        (square_shaped<LhsXprType, 2, b> and (dimension_size_of_index_is<RhsXprType, 0, 1> or dimension_size_of_index_is<RhsXprType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsXprType, 0, 1> or dimension_size_of_index_is<LhsXprType, 1, 1>) and square_shaped<RhsXprType, 2, b>));


    template<applicability b>
    static constexpr bool is_square =
      square_shaped<LhsXprType, 2, applicability::permitted> and square_shaped<RhsXprType, 2, applicability::permitted> and
      (b != applicability::guaranteed or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        square_shaped<LhsXprType, 2, b> or square_shaped<RhsXprType, 2, b>);


    static constexpr bool is_triangular_adapter = false;


    template<triangle_type t>
    static constexpr bool triangle_type_value =
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum ?
      triangular_matrix<LhsXprType, t> and triangular_matrix<RhsXprType, t> and
      (t != triangle_type::any or triangle_type_of_v<LhsXprType, RhsXprType> != triangle_type::any) :
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and
      (triangular_matrix<LhsXprType, t> or triangular_matrix<RhsXprType, t> or
       (triangular_matrix<LhsXprType, triangle_type::lower> and triangular_matrix<RhsXprType, triangle_type::upper>) or
       (triangular_matrix<LhsXprType, triangle_type::upper> and triangular_matrix<RhsXprType, triangle_type::lower>));


    static constexpr bool is_hermitian = Traits::preserves_hermitian and
      hermitian_matrix<LhsXprType, applicability::permitted> and hermitian_matrix<RhsXprType, applicability::permitted>;;

  };

}

#endif
