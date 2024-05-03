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
 * \brief Traits for Eigen::CwiseBinaryOp.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP
#define OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP

#include <type_traits>

namespace OpenKalman::interface
{
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct indexible_object_traits<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : Eigen3::indexible_object_traits_base<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
  {
  private:

    using Xpr = Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>;
    using Base = Eigen3::indexible_object_traits_base<Xpr>;
    using Traits = Eigen3::BinaryFunctorTraits<std::decay_t<BinaryOp>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto \
    get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (static_index_value<N>)
      {
        if constexpr (not dynamic_dimension<LhsType, n>)
          return OpenKalman::get_vector_space_descriptor(arg.lhs(), n);
        else
          return OpenKalman::get_vector_space_descriptor(arg.rhs(), n);
      }
      else return OpenKalman::get_vector_space_descriptor(arg.lhs(), n);
    }


    using dependents = std::tuple<typename Xpr::LhsNested, typename Xpr::RhsNested>;


    static constexpr bool has_runtime_parameters = false;


    // nested_object() not defined


    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::CwiseBinaryOp<BinaryOp, equivalent_self_contained_t<LhsType>, equivalent_self_contained_t<RhsType>>;
      // Do a partial evaluation as long as at least one argument is already self-contained.
      if constexpr ((self_contained<LhsType> or self_contained<RhsType>) and
        not std::is_lvalue_reference_v<typename N::LhsNested> and
        not std::is_lvalue_reference_v<typename N::RhsNested>)
      {
        return N {make_self_contained(arg.lhs()), make_self_contained(arg.rhs()), arg.functor()};
      }
      else
      {
        return make_dense_object(std::forward<Arg>(arg));
      }
    }

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
        return constant_coefficient {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsType>)
        return constant_coefficient {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsType>)
        return constant_coefficient {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsType>)
        return constant_coefficient {arg.rhs()};
#ifdef __cpp_concepts
      else if constexpr (requires { Traits::constexpr_operation(); })
#else
      else if constexpr (constexpr_operation_defined<>::value)
#endif
        return internal::scalar_constant_operation {Traits::constexpr_operation(),
          constant_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
      else
        return internal::scalar_constant_operation {arg.functor(),
          constant_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
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
        return constant_diagonal_coefficient {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum and zero<RhsType>)
        return constant_diagonal_coefficient {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<LhsType>)
        return constant_coefficient {arg.lhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and zero<RhsType>)
        return constant_coefficient {arg.rhs()};
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::product)
      {
        auto c_left = [](const Arg& arg){
          if constexpr (constant_matrix<LhsType>) return constant_coefficient {arg.lhs()};
          else return constant_diagonal_coefficient {arg.lhs()};
        }(arg);
        auto c_right = [](const Arg& arg){
          if constexpr (constant_matrix<RhsType>) return constant_coefficient {arg.rhs()};
          else return constant_diagonal_coefficient {arg.rhs()};
        }(arg);
#ifdef __cpp_concepts
        if constexpr (requires { Traits::constexpr_operation(); })
#else
        if constexpr (constexpr_operation_defined<>::value)
#endif
          return internal::scalar_constant_operation {Traits::constexpr_operation(), c_left, c_right};
        else
          return internal::scalar_constant_operation {arg.functor(), c_left, c_right};
      }
      else if constexpr (Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum or Traits::preserves_constant_diagonal)
      {
#ifdef __cpp_concepts
        if constexpr (requires { Traits::constexpr_operation(); })
#else
        if constexpr (constexpr_operation_defined<>::value)
#endif
          return internal::scalar_constant_operation {Traits::constexpr_operation(),
            constant_diagonal_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
        else
          return internal::scalar_constant_operation {arg.functor(),
            constant_diagonal_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
      }
      else
      {
        return std::monostate{};
      }
    }


    template<Qualification b>
    static constexpr bool one_dimensional =
      OpenKalman::one_dimensional<LhsType, Qualification::depends_on_dynamic_shape> and OpenKalman::one_dimensional<RhsType, Qualification::depends_on_dynamic_shape> and
      (b != Qualification::unqualified or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        (square_shaped<LhsType, b> and (dimension_size_of_index_is<RhsType, 0, 1> or dimension_size_of_index_is<RhsType, 1, 1>)) or
        ((dimension_size_of_index_is<LhsType, 0, 1> or dimension_size_of_index_is<LhsType, 1, 1>) and square_shaped<RhsType, b>));


    template<Qualification b>
    static constexpr bool is_square =
      square_shaped<LhsType, Qualification::depends_on_dynamic_shape> and square_shaped<RhsType, Qualification::depends_on_dynamic_shape> and
      (b != Qualification::unqualified or not has_dynamic_dimensions<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>> or
        square_shaped<LhsType, b> or square_shaped<RhsType, b>);


    static constexpr bool is_triangular_adapter = false;


    template<TriangleType t>
    static constexpr bool is_triangular =
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::sum ?
        triangular_matrix<LhsType, t> and triangular_matrix<RhsType, t> and
          (t != TriangleType::any or triangle_type_of_v<LhsType, RhsType> != TriangleType::any) :
      Traits::binary_functor_type == Eigen3::BinaryFunctorType::product and
        (triangular_matrix<LhsType, t> or triangular_matrix<RhsType, t> or
          (triangular_matrix<LhsType, TriangleType::lower> and triangular_matrix<RhsType, TriangleType::upper>) or
          (triangular_matrix<LhsType, TriangleType::upper> and triangular_matrix<RhsType, TriangleType::lower>));


    static constexpr bool is_hermitian = Traits::preserves_hermitian and
      hermitian_matrix<LhsType, Qualification::depends_on_dynamic_shape> and hermitian_matrix<RhsType, Qualification::depends_on_dynamic_shape>;;
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_CWISEBINARYOP_HPP
