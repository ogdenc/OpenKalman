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

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_UNARY_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_UNARY_HPP

#include <type_traits>
#include <complex>

namespace OpenKalman::Eigen3
{

  // Default unary functor traits
  template<typename Operation>
  struct UnaryFunctorTraits
  {
    /// Construct Operation or (preferably) an equivalent constexpr operation equivalent to Operation.
    static constexpr auto constexpr_operation() = delete;

    /// Whether the operation applied to a triangular matrix always yields a triangular matrix.
    static constexpr bool preserves_triangle = false;

    /// Whether the operation applied to a hermitian matrix always yields a hermitian matrix.
    static constexpr bool preserves_hermitian = false;
  };


#ifndef __cpp_concepts
  template<typename UnaryOp, typename = void>
  struct constexpr_unary_operation_defined_impl : std::false_type {};

  template<typename UnaryOp>
  struct constexpr_unary_operation_defined_impl<UnaryOp, std::void_t<decltype(Eigen3::UnaryFunctorTraits<UnaryOp>::constexpr_operation())>>
    : std::true_type {};
#endif


  /// Whether there is a constexpr version of functor UnaryOp
  template<typename UnaryOp>
#ifdef __cpp_concepts
  concept constexpr_unary_operation_defined = requires { Eigen3::UnaryFunctorTraits<std::decay_t<UnaryOp>>::constexpr_operation(); };
#else
  constexpr bool constexpr_unary_operation_defined = constexpr_unary_operation_defined_impl<std::decay_t<UnaryOp>>::value;
#endif


  // --------------- //
  //  stl operators  //
  // --------------- //

  template<typename Scalar>
  struct UnaryFunctorTraits<std::negate<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::negate<Scalar>{}; };
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<std::logical_not<Scalar>>
  {
    static constexpr auto constexpr_operation() { return std::logical_not<Scalar>{}; };
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  // ----------------- //
  //  Eigen operators  //
  // ----------------- //

  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_opposite_op<Scalar>> : UnaryFunctorTraits<std::negate<Scalar>> {};


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_abs_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_abs(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_score_coeff_op<Scalar>>
    : UnaryFunctorTraits<Eigen::internal::scalar_abs_op<Scalar>> {};


  // abs_knowing_score not implemented because it is not a true Eigen functor.


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_abs2_op<Scalar>>
  {
    struct Op1
    {
      constexpr auto operator()(Scalar arg) const
      {
        auto r = internal::constexpr_real(arg);
        auto i = internal::constexpr_imag(arg);
        return r * r + i * i;
      }
    };
    struct Op2 { constexpr auto operator()(Scalar arg) const { return arg * arg; } };
    static constexpr auto constexpr_operation()
    {
      if constexpr (value::complex_number<Scalar>) return Op1{};
      else return Op2{};
    }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_conjugate_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_conj(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_arg_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const {
        return internal::constexpr_atan2(internal::constexpr_imag(arg), internal::constexpr_real(arg));
      }
    };
    static constexpr auto constexpr_operation()
    {
      return Op{};
    }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar, typename NewType>
  struct UnaryFunctorTraits<Eigen::internal::scalar_cast_op<Scalar, NewType>>
  {
    struct Op1
    {
      constexpr auto operator()(Scalar arg) const
      {
        return static_cast<NewType>(static_cast<typename Eigen::NumTraits<NewType>::Real>(arg));
      }
    };
    struct Op2 { constexpr auto operator()(Scalar arg) const { return static_cast<NewType>(arg); } };
    static constexpr auto constexpr_operation()
    {
      if constexpr (not Eigen::NumTraits<Scalar>::IsComplex and Eigen::NumTraits<NewType>::IsComplex)
        return Op1{};
      else
        return Op2{};
    }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, int N>
  struct UnaryFunctorTraits<Eigen::internal::scalar_shift_right_op<Scalar, N>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return arg >> N; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar, int N>
  struct UnaryFunctorTraits<Eigen::internal::scalar_shift_left_op<Scalar, N>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return arg << N; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_real_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_real(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_imag_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_imag(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = not value::complex_number<Scalar>;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_real_ref_op<Scalar>>
    : UnaryFunctorTraits<Eigen::internal::scalar_real_op<Scalar>> {};


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_imag_ref_op<Scalar>>
    : UnaryFunctorTraits<Eigen::internal::scalar_imag_op<Scalar>> {};


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_exp_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_exp(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_expm1_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_expm1(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_log_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_log(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_log1p_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_log1p(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_log10_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const
      {
        using S = std::decay_t<decltype(internal::constexpr_real(arg))>;
        return internal::constexpr_log(arg) / numbers::ln10_v<S>;
      }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_log2_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const
      {
        using S = std::decay_t<decltype(internal::constexpr_real(arg))>;
        return internal::constexpr_log(arg) / numbers::ln2_v<S>;
      }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_sqrt_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_sqrt(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_rsqrt_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const
      {
        if (arg == Scalar{0}) return internal::constexpr_NaN<Scalar>();
        else return Scalar{1} / internal::constexpr_sqrt(arg);
      }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_cos_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_cos(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_sin_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_sin(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_tan_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_tan(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_acos_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_acos(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_asin_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_asin(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_atan_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_atan(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_tanh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_tanh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_atanh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_atanh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_sinh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_sinh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_asinh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_asinh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_cosh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_cosh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_acosh_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return internal::constexpr_acosh(arg); } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };
# endif


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_inverse_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const
      {
        if (arg == Scalar{0}) return internal::constexpr_NaN<Scalar>();
        else return static_cast<Scalar>(1) / arg;
      }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_square_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return arg * arg; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_cube_op<Scalar>>
  {
    struct Op { constexpr auto operator()(Scalar arg) const { return arg * arg * arg; } };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = true;
    static constexpr bool preserves_hermitian = true;
  };


  // Eigen::internal::scalar_round_op not implemented
  // Eigen::internal::scalar_floor_op not implemented
  // Eigen::internal::scalar_rint_op not implemented (Eigen 3.4+)
  // Eigen::internal::scalar_ceil_op not implemented
  // Eigen::internal::scalar_isnan_op not implemented
  // Eigen::internal::scalar_isinf_op not implemented
  // Eigen::internal::scalar_isfinite_op not implemented


  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_boolean_not_op<Scalar>>
    : UnaryFunctorTraits<std::logical_not<Scalar>> {};


  // Eigen::internal::scalar_sign_op not implemented


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar>
  struct UnaryFunctorTraits<Eigen::internal::scalar_logistic_op<Scalar>>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const { return Scalar{1} / (Scalar{1} + (internal::constexpr_exp(-arg))); }
    };
    static constexpr auto constexpr_operation() { return Op{}; }
    static constexpr bool preserves_triangle = false;
    static constexpr bool preserves_hermitian = true;
  };
#endif


  template<typename BinaryOp>
  struct UnaryFunctorTraits<Eigen::internal::bind1st_op<BinaryOp>>
  {
    using BinaryTraits = BinaryFunctorTraits<std::decay_t<BinaryOp>>;
    static constexpr bool preserves_triangle = BinaryTraits::binary_functor_type == BinaryFunctorType::product;
    static constexpr bool preserves_hermitian = BinaryTraits::preserves_hermitian;

    template<typename UnaryOp, typename XprType>
    static constexpr auto get_constant(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& arg)
    {
      using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
      using ConstType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
      using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, ConstType, XprType>;
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {ConstType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, x, arg.functor()};
      return constant_coefficient {bin};
    }

    template<typename UnaryOp, typename XprType>
    static constexpr auto get_constant_diagonal(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& arg)
    {
      using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
      using ConstType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
      using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, ConstType, XprType>;
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {ConstType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, x, arg.functor()};
      return constant_diagonal_coefficient {bin};
    }
  };


  template<typename BinaryOp>
  struct UnaryFunctorTraits<Eigen::internal::bind2nd_op<BinaryOp>>
  {
    using BinaryTraits = BinaryFunctorTraits<std::decay_t<BinaryOp>>;
    static constexpr bool preserves_triangle = BinaryTraits::binary_functor_type == BinaryFunctorType::product;
    static constexpr bool preserves_hermitian = BinaryTraits::preserves_hermitian;

    template<typename UnaryOp, typename XprType>
    static constexpr auto get_constant(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& arg)
    {
      using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
      using ConstType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
      using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, XprType, ConstType>;
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {x, ConstType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, arg.functor()};
      return constant_coefficient {bin};
    }

    template<typename UnaryOp, typename XprType>
    static constexpr auto get_constant_diagonal(const Eigen::CwiseUnaryOp<UnaryOp, XprType>& arg)
    {
      using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
      using ConstType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
      using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, XprType, ConstType>;
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {x, ConstType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, arg.functor()};
      return constant_diagonal_coefficient {bin};
    }
  };


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_UNARY_HPP
