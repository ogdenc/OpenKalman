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
 * \brief Trait details for Eigen unary functors.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_UNARY_HPP
#define OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_UNARY_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  namespace EGI = Eigen::internal;


  namespace detail
  {
    template<typename XprType, template<typename...> typename T, CompileTimeStatus>
    struct has_constant_arg;

    template<typename XprType, CompileTimeStatus c>
    struct has_constant_arg<XprType, constant_coefficient, c>
      : std::bool_constant<constant_matrix<XprType, Likelihood::maybe, c>> {};

    template<typename XprType, CompileTimeStatus c>
    struct has_constant_arg<XprType, constant_diagonal_coefficient, c>
      : std::bool_constant<constant_diagonal_matrix<XprType, Likelihood::maybe, c>> {};


    template<typename XprType, template<typename...> typename T, typename Op, typename Arg>
    static constexpr auto default_get_constant(const Op& op, const Arg& arg)
    {
      if constexpr (has_constant_arg<XprType, T, CompileTimeStatus::known>::value)
        return scalar_constant_operation {op, T {arg.nestedExpression()}};
      else if constexpr (has_constant_arg<XprType, T, CompileTimeStatus::any>::value)
        return arg.functor()(T {arg.nestedExpression()}());
      else
        return std::monostate {};
    }

  } // namespace detail


  // Default unary traits, if UnaryOp is not specifically matched.
  template<typename BinaryOp, typename XprType>
  struct FunctorTraits<BinaryOp, XprType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (std::is_same_v<T<XprType>, constant_coefficient<XprType>>)
        return default_get_constant<XprType, T>(arg.functor(), arg);
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_opposite_op<Scalar>, XprType>
  {
    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {std::negate<>{}, T<XprType> {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_abs_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = real_part(arg);
          auto i = imaginary_part(arg);
          return constexpr_sqrt(r * r + i * i);
        }
        else return arg >= Scalar{0} ? arg : -arg;
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T<XprType> {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_score_coeff_op<Scalar>, XprType>
    : FunctorTraits<EGI::scalar_abs_op<Scalar>, XprType> {};


  // abs_knowing_score not implemented because it is not a true Eigen functor.


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_abs2_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = real_part(arg);
          auto i = imaginary_part(arg);
          return r * r + i * i;
        }
        else return arg * arg;
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_conjugate_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return conjugate(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_arg_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept {
      return internal::constexpr_atan2(imaginary_part(arg), real_part(arg)); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, constant_coefficient {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename NewType, typename XprType>
  struct FunctorTraits<EGI::scalar_cast_op<Scalar, NewType>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        if constexpr (not Eigen::NumTraits<Scalar>::IsComplex and Eigen::NumTraits<NewType>::IsComplex)
          return static_cast<NewType>(static_cast<typename Eigen::NumTraits<NewType>::Real>(arg));
        else
          return static_cast<NewType>(arg);
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, int N, typename XprType>
  struct FunctorTraits<EGI::scalar_shift_right_op<Scalar, N>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg >> N; } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, int N, typename XprType>
  struct FunctorTraits<EGI::scalar_shift_left_op<Scalar, N>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg << N; } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_real_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return real_part(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_imag_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return imaginary_part(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = not complex_number<Scalar> and hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_real_ref_op<Scalar>, XprType>
    : FunctorTraits<EGI::scalar_real_op<Scalar>, XprType> {};


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_imag_ref_op<Scalar>, XprType>
    : FunctorTraits<EGI::scalar_imag_op<Scalar>, XprType> {};


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_exp_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_exp(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_expm1_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_expm1(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_log_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_log1p_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log(arg + Scalar{1}); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_log10_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        using S = std::decay_t<decltype(real_part(arg))>;
        return internal::constexpr_log(arg) / numbers::ln10_v<S>;
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_log2_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        using S = std::decay_t<decltype(real_part(arg))>;
        return internal::constexpr_log(arg) / numbers::ln2_v<S>;
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_sqrt_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_sqrt(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_rsqrt_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_pow(arg, -0.5); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_cos_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_cos(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_sin_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_sin(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_tan_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_tan(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_acos_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_acos(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_asin_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_asin(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_atan_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_atan(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_tanh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        return internal::constexpr_tanh(arg);
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_atanh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log((1 + arg)/(1 - arg)) / 2; }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_sinh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        return internal::constexpr_sinh(arg);
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_asinh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log(arg + internal::constexpr_sqrt(arg * arg + 1)); }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_cosh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        return internal::constexpr_cosh(arg);
      }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_acosh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log(arg + internal::constexpr_sqrt(arg * arg - 1)); }
    };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
# endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_inverse_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return static_cast<Scalar>(1) / arg; } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero_matrix<XprType, Likelihood::maybe> or not std::is_same_v<T<XprType>, constant_coefficient<XprType>>)
        return std::monostate {};
      else
        return scalar_constant_operation {Op{}, constant_coefficient {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_square_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg * arg; } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_cube_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg * arg * arg; } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = diagonal_matrix<XprType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<XprType>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  // EGI::scalar_round_op not implemented
  // EGI::scalar_floor_op not implemented
  // EGI::scalar_rint_op not implemented (Eigen 3.4+)
  // EGI::scalar_ceil_op not implemented
  // EGI::scalar_isnan_op not implemented
  // EGI::scalar_isinf_op not implemented
  // EGI::scalar_isfinite_op not implemented


  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_boolean_not_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return not static_cast<bool>(arg); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not std::is_same_v<T<XprType>, constant_coefficient<XprType>>) return std::monostate {};
      else return scalar_constant_operation {Op{}, constant_coefficient {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };


  // EGI::scalar_sign_op not implemented


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<EGI::scalar_logistic_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return 1 / (1 + (internal::constexpr_exp(-arg))); } };

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return scalar_constant_operation {Op{}, T {arg.nestedExpression()}};
    }

    static constexpr bool is_diagonal = false;

    static constexpr TriangleType triangle_type = TriangleType::none;

    static constexpr bool is_hermitian = hermitian_matrix<XprType>;
  };
#endif


  template<typename BinaryOp, typename XprType>
  struct FunctorTraits<EGI::bind1st_op<BinaryOp>, XprType>
  {
    using ConstantType = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<scalar_type_of_t<XprType>>, XprType>;
    using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, ConstantType, XprType>;

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (detail::has_constant_arg<BinaryOpType, T, CompileTimeStatus::known>::value) //< e.g., sometimes if XprType is zero
        return T<BinaryOpType>{};
      else if constexpr (detail::has_constant_arg<XprType, T, CompileTimeStatus::any>::value)
        return arg.functor()(T {arg.nestedExpression()}());
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = diagonal_matrix<BinaryOpType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<BinaryOpType>;

    static constexpr bool is_hermitian = hermitian_matrix<BinaryOpType>;
  };


  template<typename BinaryOp, typename XprType>
  struct FunctorTraits<EGI::bind2nd_op<BinaryOp>, XprType>
  {
    using ConstantType = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<scalar_type_of_t<XprType>>, XprType>;
    using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, XprType, ConstantType>;

    template<template<typename...> typename T, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (detail::has_constant_arg<BinaryOpType, T, CompileTimeStatus::known>::value) //< e.g., sometimes if XprType is zero
        return T<BinaryOpType>{};
      else if constexpr (detail::has_constant_arg<XprType, T, CompileTimeStatus::any>::value)
        return arg.functor()(T {arg.nestedExpression()}());
      else
        return std::monostate {};
    }

    static constexpr bool is_diagonal = diagonal_matrix<BinaryOpType>;

    static constexpr TriangleType triangle_type = triangle_type_of_v<BinaryOpType>;

    static constexpr bool is_hermitian = hermitian_matrix<BinaryOpType>;
  };


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_UNARY_HPP
