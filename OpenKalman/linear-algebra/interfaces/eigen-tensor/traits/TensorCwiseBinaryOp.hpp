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
  struct indexible_object_traits<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
    : Eigen3::indexible_object_traits_tensor_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
  {
  private:

    using Xpr = Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>;
    using Base = Eigen3::indexible_object_traits_tensor_base<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>;
    using Traits = Eigen3::BinaryFunctorTraits<std::decay_t<BinaryOp>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto
    get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
      {
        if constexpr (not dynamic_dimension<LhsXprType, n>)
          return OpenKalman::get_vector_space_descriptor(arg.lhsExpression(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.rhsExpression(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.lhsExpression(), n);
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
        return constant_coefficient {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsXprType>)
        return constant_coefficient {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsXprType>)
        return constant_coefficient {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsXprType>)
        return constant_coefficient {arg.rhsExpression()};
#ifdef __cpp_concepts
      else if constexpr (requires { Traits::constexpr_operation(); })
#else
        else if constexpr (constexpr_operation_defined<>::value)
#endif
        return values::operation(Traits::constexpr_operation(),
          constant_coefficient {arg.lhsExpression()}, constant_coefficient {arg.rhsExpression()});
      else
        return values::operation(arg.functor(),
          constant_coefficient {arg.lhsExpression()}, constant_coefficient {arg.rhsExpression()});
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
        return constant_diagonal_coefficient {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsXprType>)
        return constant_diagonal_coefficient {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsXprType>)
        return constant_coefficient {arg.lhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsXprType>)
        return constant_coefficient {arg.rhsExpression()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product)
      {
        auto c_left = [](const Arg& arg){
            if constexpr (constant_matrix<LhsXprType>) return constant_coefficient {arg.lhsExpression()};
            else return constant_diagonal_coefficient {arg.lhsExpression()};
        }(arg);
        auto c_right = [](const Arg& arg){
            if constexpr (constant_matrix<RhsXprType>) return constant_coefficient {arg.rhsExpression()};
            else return constant_diagonal_coefficient {arg.rhsExpression()};
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
            constant_diagonal_coefficient {arg.lhsExpression()}, constant_diagonal_coefficient {arg.rhsExpression()});
        else
          return values::operation(arg.functor(),
            constant_diagonal_coefficient {arg.lhsExpression()}, constant_diagonal_coefficient {arg.rhsExpression()});
      }
      else
      {
        return std::monostate{};
      }
    }


    template<Applicability b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsXprType, Applicability::permitted> and OpenKalman::one_dimensional<RhsXprType, Applicability::permitted> and
      (b != Applicability::guaranteed or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        (square_shaped<LhsXprType, b> and (dimension_size_of_index_is<RhsXprType, 0, 1> or dimension_size_of_index_is<RhsXprType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsXprType, 0, 1> or dimension_size_of_index_is<LhsXprType, 1, 1>) and square_shaped<RhsXprType, b>));


    template<Applicability b>
    static constexpr bool is_square =
      square_shaped<LhsXprType, Applicability::permitted> and square_shaped<RhsXprType, Applicability::permitted> and
      (b != Applicability::guaranteed or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>> or
        square_shaped<LhsXprType, b> or square_shaped<RhsXprType, b>);


    static constexpr bool is_triangular_adapter = false;


    template<TriangleType t>
    static constexpr bool is_triangular =
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum ?
      triangular_matrix<LhsXprType, t> and triangular_matrix<RhsXprType, t> and
      (t != TriangleType::any or triangle_type_of_v<LhsXprType, RhsXprType> != TriangleType::any) :
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and
      (triangular_matrix<LhsXprType, t> or triangular_matrix<RhsXprType, t> or
       (triangular_matrix<LhsXprType, TriangleType::lower> and triangular_matrix<RhsXprType, TriangleType::upper>) or
       (triangular_matrix<LhsXprType, TriangleType::upper> and triangular_matrix<RhsXprType, TriangleType::lower>));


    static constexpr bool is_hermitian = Traits::preserves_hermitian and
      hermitian_matrix<LhsXprType, Applicability::permitted> and hermitian_matrix<RhsXprType, Applicability::permitted>;;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORCWISEBINARYOP_HPP
