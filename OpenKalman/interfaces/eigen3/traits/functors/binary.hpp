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
 * \brief Trait details for Eigen binary functors.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_BINARY_HPP
#define OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_BINARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  namespace EGI = Eigen::internal;

  namespace detail
  {
    template<typename LhsType, typename RhsType, template<typename...> typename T, CompileTimeStatus>
    struct has_constant_args;

    template<typename LhsType, typename RhsType, CompileTimeStatus c>
    struct has_constant_args<LhsType, RhsType, constant_coefficient, c> : std::bool_constant<
      constant_matrix<LhsType, Likelihood::maybe, c> and
      constant_matrix<RhsType, Likelihood::maybe, c>> {};

    template<typename LhsType, typename RhsType, CompileTimeStatus c>
    struct has_constant_args<LhsType, RhsType, constant_diagonal_coefficient, c> : std::bool_constant<
      constant_diagonal_matrix<LhsType, Likelihood::maybe, c> and
      constant_diagonal_matrix<RhsType, Likelihood::maybe, c>> {};


    template<typename LhsType, typename RhsType, template<typename...> typename T, typename Op, typename Arg>
    static constexpr auto default_get_constant(const Op& op, const Arg& arg)
    {
      if constexpr (has_constant_args<LhsType, RhsType, T, CompileTimeStatus::known>::value)
        return scalar_constant_operation {op, T {arg.lhs()}, T {arg.rhs()}};
      else if constexpr (has_constant_args<LhsType, RhsType, T, CompileTimeStatus::any>::value)
        return arg.functor()(T {arg.lhs()}(), T {arg.rhs()}());
      else
        return std::monostate {};
    }


    template<typename LhsType, typename RhsType, typename Op, template<typename...> typename T, typename Arg>
    static constexpr auto get_constant_sum_impl(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>)
        return T {arg.rhs()};
      else if constexpr (zero_matrix<RhsType>)
        return T {arg.lhs()};
      else
        return default_get_constant<LhsType, RhsType, T>(Op{}, arg);
    }


    template<typename LhsType, typename RhsType, typename Op, template<typename...> typename T, typename Arg>
    static constexpr auto get_constant_product_impl(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>)
      {
        return T {arg.lhs()};
      }
      else if constexpr (zero_matrix<RhsType>)
      {
        return T {arg.rhs()};
      }
      else if constexpr (std::is_same_v<T<LhsType>, constant_diagonal_coefficient<LhsType>> and
        constant_diagonal_matrix<LhsType, Likelihood::definitely, CompileTimeStatus::any> and
        constant_matrix<RhsType, Likelihood::maybe, CompileTimeStatus::any>)
      {
        return scalar_constant_operation {
          Op{}, constant_diagonal_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
      }
      else if constexpr (std::is_same_v<T<RhsType>, constant_diagonal_coefficient<RhsType>> and
        constant_matrix<LhsType, Likelihood::maybe, CompileTimeStatus::any> and
        constant_diagonal_matrix<RhsType, Likelihood::definitely, CompileTimeStatus::any>)
      {
        return scalar_constant_operation {
          Op{}, constant_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
      }
      else
      {
        return default_get_constant<LhsType, RhsType, T>(Op{}, arg);
      }
    }


    template<typename Arg1, typename Arg2>
    static constexpr auto is_diagonal_sum = diagonal_matrix<Arg1> and diagonal_matrix<Arg2>;


    template<typename Arg1, typename Arg2>
    static constexpr auto is_diagonal_product =
      diagonal_matrix<Arg1> or diagonal_matrix<Arg2> or
      (lower_triangular_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
      (upper_triangular_matrix<Arg1> and lower_triangular_matrix<Arg2>);


    template<typename Arg1, typename Arg2>
    static constexpr TriangleType triangle_type_product =
      (diagonal_matrix<Arg1> or diagonal_matrix<Arg2> or
        (lower_triangular_matrix<Arg1> and upper_triangular_matrix<Arg2>) or
        (upper_triangular_matrix<Arg1> and lower_triangular_matrix<Arg2>)) ? TriangleType::diagonal :
      ((lower_triangular_matrix<Arg1> or lower_triangular_matrix<Arg2>) ? TriangleType::lower :
      ((upper_triangular_matrix<Arg1> or upper_triangular_matrix<Arg2>) ? TriangleType::upper :
      TriangleType::none));

  } // namespace detail


  // Default binary traits, if BinaryOp is not specifically matched.
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct FunctorTraits<BinaryOp, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg> 
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (std::is_same_v<T<LhsType>, constant_coefficient<LhsType>>)
        return detail::default_get_constant<LhsType, RhsType, T>(arg.functor(), arg);
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_sum_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<LhsType, RhsType, std::plus<>, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_product_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<LhsType, RhsType, std::multiplies<>, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_product<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = detail::triangle_type_product<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_conj_product_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return constexpr_conj(arg1) * arg2; }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<LhsType, RhsType, Op, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_product<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = detail::triangle_type_product<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_min_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return std::min(arg1, arg2); }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<LhsType, RhsType, T>(Op{}, arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_max_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return std::max(arg1, arg2); }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<LhsType, RhsType, T>(Op{}, arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename LhsScalar, typename RhsScalar, Eigen::internal::ComparisonName cmp, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_cmp_op<LhsScalar, RhsScalar, cmp>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(LhsScalar a, RhsScalar b) const noexcept
      {
        if constexpr (cmp == EGI::ComparisonName::cmp_EQ) return a == b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_LT) return a < b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_LE) return a <= b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_GT) return a > b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_GE) return a >= b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_NEQ) return a != b;
        else if constexpr (cmp == EGI::ComparisonName::cmp_UNORD) return not (a<=b or b<=a);
        else return EGI::scalar_cmp_op<LhsScalar, RhsScalar, cmp> {}(a, b); // Failsafe, but not a constexpr function.
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<LhsType>, constant_coefficient<LhsType>>) return std::monostate {};
      else return detail::default_get_constant<LhsType, RhsType, constant_coefficient>(Op{}, arg);
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_hypot_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept
      {
        return OpenKalman::internal::constexpr_sqrt(arg1 * arg1 + arg2 * arg2);
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<LhsType, RhsType, Op, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar, typename Exponent, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_pow_op<Scalar, Exponent>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg1, Exponent arg2) const noexcept { return internal::constexpr_pow(arg1, arg2); }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<LhsType>, constant_coefficient<LhsType>>) return std::monostate {};
      else if constexpr (zero_matrix<RhsType>) return std::integral_constant<int, 1>{};
      else return detail::default_get_constant<LhsType, RhsType, constant_coefficient>(Op{}, arg);
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_difference_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>) return scalar_constant_operation {std::negate<>{}, T {arg.rhs()}};
      else if constexpr (zero_matrix<RhsType>) return T {arg.lhs()};
      else return detail::default_get_constant<LhsType, RhsType, T>(std::minus<>{}, arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_quotient_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<LhsType>, constant_coefficient<LhsType>> or zero_matrix<RhsType>) return std::monostate {};
      else return detail::default_get_constant<LhsType, RhsType, constant_coefficient>(std::divides<>{}, arg);
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_and_op, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<LhsType, RhsType, std::logical_and<>, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_product<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = detail::triangle_type_product<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_or_op, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (std::is_same_v<T<LhsType>, constant_coefficient<LhsType>> and
        constant_diagonal_matrix<LhsType, Likelihood::definitely, CompileTimeStatus::any> and
        constant_matrix<RhsType, Likelihood::maybe, CompileTimeStatus::known>)
      {
        if constexpr (constant_coefficient_v<RhsType> == true) return std::integral_constant<bool, true>{};
        else return constant_diagonal_coefficient {arg.lhs()};
      }
      else if constexpr (std::is_same_v<T<RhsType>, constant_coefficient<RhsType>> and
        constant_matrix<LhsType, Likelihood::maybe, CompileTimeStatus::known> and
        constant_diagonal_matrix<RhsType, Likelihood::definitely, CompileTimeStatus::any>)
      {
        if constexpr (constant_coefficient_v<LhsType> == true) return std::integral_constant<bool, true>{};
        else return constant_diagonal_coefficient {arg.rhs()};
      }
      else
      {
        return detail::get_constant_sum_impl<LhsType, RhsType, std::logical_or<>, T>(arg);
      }
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_xor_op, LhsType, RhsType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<LhsType, RhsType, T>(std::not_equal_to<>{}, arg);
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_absolute_difference_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return arg2 > arg1 ? arg2 - arg1 : arg1 - arg2; }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<LhsType, RhsType, Op, T>(arg);
    }

    static constexpr bool is_diagonal = detail::is_diagonal_sum<LhsType, RhsType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<LhsType, RhsType>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType> and hermitian_matrix<RhsType>;
  };
#endif

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_BINARY_HPP
