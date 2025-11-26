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
 * \brief Trait details for Eigen binary functors.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_BINARY_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_BINARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // Default binary functor traits
  template<typename Operation>
  struct BinaryFunctorTraits
  {
    /// Construct Operation or (preferably) an equivalent constexpr operation equivalent to Operation.
    template<typename...Args>
    static constexpr auto constexpr_operation() = delete;

    /// Whether binary functor type.
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;

    /// Whether the operation applied to a hermitian matrix always yields a hermitian matrix.
    static constexpr bool preserves_hermitian = false;
  };


  // --------------- //
  //  stl operators  //
  // --------------- //

  template<typename Scalar>
  struct BinaryFunctorTraits<std::plus<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::plus<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::minus<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::minus<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::multiplies<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::multiplies<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::product;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::divides<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::divides<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::logical_and<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::logical_and<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::product;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::logical_or<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::logical_or<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (values::fixed<constant_value<RhsType>>)
      {
        if constexpr (constant_value_v<RhsType> == true) return constant_value {arg.rhs()};
        else return constant_value {arg.lhs()};
      }
      else if constexpr (values::fixed<constant_value<LhsType>>)
      {
        if constexpr (constant_value_v<LhsType> == true) return constant_value {arg.lhs()};
        else return constant_value {arg.rhs()};
      }
      else
      {
        return std::monostate{};
      }
    }

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (values::fixed<constant_diagonal_value<RhsType>>)
      {
        if constexpr (constant_diagonal_value_v<RhsType> == true and diagonal_matrix<LhsType>)
          return constant_diagonal_value {arg.rhs()};
        else if constexpr (constant_diagonal_value_v<RhsType> == false)
          return constant_diagonal_value {arg.lhs()};
        else
          return std::monostate{};
      }
      else if constexpr (values::fixed<constant_diagonal_value<LhsType>>)
      {
        if constexpr (constant_diagonal_value_v<LhsType> == true and diagonal_matrix<RhsType>)
          return constant_diagonal_value {arg.lhs()};
        else if constexpr (constant_diagonal_value_v<LhsType> == false)
          return constant_diagonal_value {arg.rhs()};
        else
          return std::monostate{};
      }
      else
      {
        return std::monostate{};
      }
    }
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::equal_to<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::equal_to<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::not_equal_to<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::not_equal_to<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::greater<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::greater<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::less<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::less<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::greater_equal<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::greater_equal<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct BinaryFunctorTraits<std::less_equal<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::less_equal<Scalar>{}; };
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  // ----------------- //
  //  Eigen operators  //
  // ----------------- //

  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>>
    : BinaryFunctorTraits<std::plus<std::common_type_t<Scalar1, Scalar2>>> {};


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_product_op<Scalar1, Scalar2>>
    : BinaryFunctorTraits<std::multiplies<std::common_type_t<Scalar1, Scalar2>>> {};


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_conj_product_op<Scalar1, Scalar2>>
  {
    struct Op { constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const { return values::conj(arg1) * arg2; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::product;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_min_op<Scalar1, Scalar2>>
  {
    struct Op { constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const { return std::min(arg1, arg2); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (values::fixed<constant_value<LhsType>> and values::fixed<constant_diagonal_value<RhsType>>)
      {
        if constexpr (constant_value_v<LhsType> < 0 and constant_value_v<LhsType> < constant_diagonal_value_v<RhsType>)
          return constant_value {arg.lhs()};
        else return std::monostate{};
      }
      else if constexpr (values::fixed<constant_diagonal_value<LhsType>> and values::fixed<constant_value<RhsType>>)
      {
        if constexpr (constant_value_v<RhsType> < 0 and constant_value_v<RhsType> < constant_diagonal_value_v<LhsType>)
          return constant_value {arg.rhs()};
        else return std::monostate{};
      }
      else
      {
        return values::operation(constexpr_operation(), constant_value {arg.lhs()}, constant_value {arg.rhs()});
      }
    }

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (values::fixed<constant_value<LhsType>> and values::fixed<constant_diagonal_value<RhsType>>)
      {
        if constexpr (constant_value_v<LhsType> > 0 and constant_value_v<LhsType> > constant_diagonal_value_v<RhsType>)
          return constant_diagonal_value {arg.rhs()};
        else return std::monostate{};
      }
      else if constexpr (values::fixed<constant_diagonal_value<LhsType>> and values::fixed<constant_value<RhsType>>)
      {
        if constexpr (constant_value_v<RhsType> > 0 and constant_value_v<RhsType> > constant_diagonal_value_v<LhsType>)
          return constant_diagonal_value {arg.lhs()};
        else return std::monostate{};
      }
      else
      {
        return values::operation(constexpr_operation(),
          constant_diagonal_value {arg.lhs()}, constant_diagonal_value {arg.rhs()});
      }
    }
  };


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_max_op<Scalar1, Scalar2>>
  {
    struct Op { constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const { return std::max(arg1, arg2); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (values::fixed<constant_value<LhsType>> and values::fixed<constant_diagonal_value<RhsType>>)
      {
        if constexpr (constant_value_v<LhsType> > 0 and constant_value_v<LhsType> > constant_diagonal_value_v<RhsType>)
          return constant_value {arg.lhs()};
        else return std::monostate{};
      }
      else if constexpr (values::fixed<constant_diagonal_value<LhsType>> and values::fixed<constant_value<RhsType>>)
      {
        if constexpr (constant_value_v<RhsType> > 0 and constant_value_v<RhsType> > constant_diagonal_value_v<LhsType>)
          return constant_value {arg.rhs()};
        else return std::monostate{};
      }
      else
      {
        return values::operation(constexpr_operation(),
          constant_value {arg.lhs()}, constant_value {arg.rhs()});
      }
    }

    template<typename LhsType, typename RhsType, typename Arg>
    static constexpr auto get_constant_diagonal(const Arg& arg)
    {
      if constexpr (values::fixed<constant_value<LhsType>> and values::fixed<constant_diagonal_value<RhsType>>)
      {
        if constexpr (constant_value_v<LhsType> < 0 and constant_value_v<LhsType> < constant_diagonal_value_v<RhsType>)
          return constant_diagonal_value {arg.rhs()};
        else return std::monostate{};
      }
      else if constexpr (values::fixed<constant_diagonal_value<LhsType>> and values::fixed<constant_value<RhsType>>)
      {
        if constexpr (constant_value_v<RhsType> < 0 and constant_value_v<RhsType> < constant_diagonal_value_v<LhsType>)
          return constant_diagonal_value {arg.lhs()};
        else return std::monostate{};
      }
      else
      {
        return values::operation(constexpr_operation(),
          constant_diagonal_value {arg.lhs()}, constant_diagonal_value {arg.rhs()});
      }
    }
  };


  template<typename LhsScalar, typename RhsScalar, Eigen::internal::ComparisonName cmp>
  struct BinaryFunctorTraits<Eigen::internal::scalar_cmp_op<LhsScalar, RhsScalar, cmp>>
  {
    struct Op {
      constexpr auto operator()(LhsScalar a, RhsScalar b) const {
        if constexpr (cmp == Eigen::internal::ComparisonName::cmp_EQ) return a == b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_LT) return a < b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_LE) return a <= b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_GT) return a > b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_GE) return a >= b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_NEQ) return a != b;
        else if constexpr (cmp == Eigen::internal::ComparisonName::cmp_UNORD) return not (a<=b or b<=a);
        else return Eigen::internal::scalar_cmp_op<LhsScalar, RhsScalar, cmp> {}(a, b); // Failsafe, but not a constexpr function.
      }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_hypot_op<Scalar1, Scalar2>>
  {
    struct Op { constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const { return values::sqrt(arg1 * arg1 + arg2 * arg2); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar, typename Exponent>
  struct BinaryFunctorTraits<Eigen::internal::scalar_pow_op<Scalar, Exponent>>
  {
    struct Op { constexpr auto operator()(Scalar arg1, Exponent arg2) const { return values::pow(arg1, arg2); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = false;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::normal;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>>
    : BinaryFunctorTraits<std::minus<std::common_type_t<Scalar1, Scalar2>>> {};


  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>>
    : BinaryFunctorTraits<std::divides<std::common_type_t<Scalar1, Scalar2>>> {};


  template<>
  struct BinaryFunctorTraits<Eigen::internal::scalar_boolean_and_op> : BinaryFunctorTraits<std::logical_and<bool>> {};


  template<>
  struct BinaryFunctorTraits<Eigen::internal::scalar_boolean_or_op> : BinaryFunctorTraits<std::logical_or<bool>> {};


  template<>
  struct BinaryFunctorTraits<Eigen::internal::scalar_boolean_xor_op> : BinaryFunctorTraits<std::not_equal_to<bool>> {};


  template<typename Scalar>
  struct BinaryFunctorTraits<Eigen::numext::not_equal_to<Scalar>> : BinaryFunctorTraits<std::not_equal_to<bool>> {};


  template<typename Scalar>
  struct BinaryFunctorTraits<Eigen::numext::equal_to<Scalar>> : BinaryFunctorTraits<std::equal_to<bool>> {};


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar1, typename Scalar2>
  struct BinaryFunctorTraits<Eigen::internal::scalar_absolute_difference_op<Scalar1, Scalar2>>
  {
    struct Op { constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const { return arg2 > arg1 ? arg2 - arg1 : arg1 - arg2; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_constant_diagonal = true;
    static constexpr BinaryFunctorType binary_functor_type = BinaryFunctorType::sum;
    static constexpr bool preserves_hermitian = true;
  };
#endif

}

#endif
