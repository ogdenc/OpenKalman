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
 * \brief Traits for Eigen::CwiseBinaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP


namespace OpenKalman::interface
{
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct object_traits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : Eigen3::object_traits_base<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
  private:

    using Xpr = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;
    using Base = Eigen3::object_traits_base<Xpr>;
    using Traits = Eigen3::BinaryFunctorTraits<std::decay_t<BinaryOp>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_pattern_collection(const Arg& arg, N n)
    {
      if constexpr (square_shaped<LhsType> or square_shaped<RhsType>)
        return internal::most_fixed_pattern(
          OpenKalman::get_pattern_collection<0>(arg.lhs()),
          OpenKalman::get_pattern_collection<0>(arg.rhs()),
          OpenKalman::get_pattern_collection<1>(arg.lhs()),
          OpenKalman::get_pattern_collection<1>(arg.rhs()));
      else
        return internal::most_fixed_pattern(
          OpenKalman::get_pattern_collection(arg.lhs(), n),
          OpenKalman::get_pattern_collection(arg.rhs(), n));
    }


    // nested_object() not defined

  private:

#ifndef __cpp_concepts
    template<typename T = Traits, typename = void>
    struct custom_get_constant_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_defined<T, std::void_t<decltype(T::template get_constant<LhsType, RhsType>(std::declval<const Xpr&>()))>>
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
      if constexpr (requires { Traits::template get_constant<LhsType, RhsType>(arg); })
#else
      if constexpr (custom_get_constant_defined<>::value)
#endif
        return Traits::template get_constant<LhsType, RhsType>(arg);
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<LhsType>)
        return constant_value {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsType>)
        return constant_value {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsType>)
        return constant_value {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsType>)
        return constant_value {arg.rhs()};
#ifdef __cpp_concepts
      else if constexpr (requires { Traits::constexpr_operation(); })
#else
      else if constexpr (constexpr_operation_defined<>::value)
#endif
        return values::operation(Traits::constexpr_operation(),
          constant_value {arg.lhs()}, constant_value {arg.rhs()});
      else
        return values::operation(arg.functor(),
          constant_value {arg.lhs()}, constant_value {arg.rhs()});
    }

  private:

#ifndef __cpp_concepts
    template<typename T = Traits, typename = void>
    struct custom_get_constant_diagonal_defined : std::false_type {};

    template<typename T>
    struct custom_get_constant_diagonal_defined<T, std::void_t<decltype(T::template get_constant_diagonal<LhsType, RhsType>(std::declval<const Xpr&>()))>>
      : std::true_type {};
#endif

  public:

    template<typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
#ifdef __cpp_concepts
      if constexpr (requires { Traits::template get_constant_diagonal<LhsType, RhsType>(arg); })
#else
      if constexpr (custom_get_constant_diagonal_defined<>::value)
#endif
        return Traits::template get_constant_diagonal<LhsType, RhsType>(arg);
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<LhsType>)
        return constant_diagonal_value {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsType>)
        return constant_diagonal_value {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsType>)
        return constant_value {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsType>)
        return constant_value {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product)
      {
        auto c_left = [](const Arg& arg){
          if constexpr (constant_matrix<LhsType>) return constant_value {arg.lhs()};
          else return constant_diagonal_value {arg.lhs()};
        }(arg);
        auto c_right = [](const Arg& arg){
          if constexpr (constant_matrix<RhsType>) return constant_value {arg.rhs()};
          else return constant_diagonal_value {arg.rhs()};
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
            constant_diagonal_value {arg.lhs()}, constant_diagonal_value {arg.rhs()});
        else
          return values::operation(arg.functor(),
            constant_diagonal_value {arg.lhs()}, constant_diagonal_value {arg.rhs()});
      }
      else
      {
        return std::monostate{};
      }
    }


    template<applicability b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsType, applicability::permitted> and
      OpenKalman::one_dimensional<RhsType, applicability::permitted> and
      (b != applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        OpenKalman::one_dimensional<LhsType, b> or
        OpenKalman::one_dimensional<RhsType, b>);


    template<applicability b>
    static constexpr bool is_square =
      square_shaped<LhsType, applicability::permitted> and
      square_shaped<RhsType, applicability::permitted> and
      (b != applicability::guaranteed or
        not has_dynamic_dimensions<Xpr> or
        square_shaped<LhsType, b> or
        square_shaped<RhsType, b>);


    static constexpr bool is_triangular_adapter = false;


    template<triangle_type t>
    static constexpr bool triangle_type_value =
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum ?
        triangular_matrix<LhsType, t> and triangular_matrix<RhsType, t> and
          (t != triangle_type::any or triangle_type_of_v<LhsType, RhsType> != triangle_type::any) :
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and
        (triangular_matrix<LhsType, t> or triangular_matrix<RhsType, t> or
          (triangular_matrix<LhsType, triangle_type::lower> and triangular_matrix<RhsType, triangle_type::upper>) or
          (triangular_matrix<LhsType, triangle_type::upper> and triangular_matrix<RhsType, triangle_type::lower>));


    static constexpr bool is_hermitian = Traits::preserves_hermitian and
      hermitian_matrix<LhsType, applicability::permitted> and hermitian_matrix<RhsType, applicability::permitted>;;

  };


}

#endif
