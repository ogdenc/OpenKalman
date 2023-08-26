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
  namespace detail
  {
    template<typename Op, typename XprType, bool is_diag, typename Arg>
    static constexpr auto default_get_constant(const Arg& arg)
    {
      if constexpr (is_diag)
      {
        if constexpr (std::is_default_constructible_v<Op> and constant_diagonal_matrix<XprType, CompileTimeStatus::known, Likelihood::maybe>)
          return internal::scalar_constant_operation {Op{}, constant_diagonal_coefficient {arg.nestedExpression()}};
        else if constexpr (constant_diagonal_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe>)
          return arg.functor()(constant_diagonal_coefficient {arg.nestedExpression()}());
        else
          return std::monostate {};
      }
      else
      {
        if constexpr (std::is_default_constructible_v<Op> and constant_matrix<XprType, CompileTimeStatus::known, Likelihood::maybe>)
          return internal::scalar_constant_operation {Op{}, constant_coefficient {arg.nestedExpression()}};
        else if constexpr (constant_matrix<XprType, CompileTimeStatus::any, Likelihood::maybe>)
          return arg.functor()(constant_coefficient {arg.nestedExpression()}());
        else
          return std::monostate {};
      }
    }
  } // namespace detail


  // Default unary traits, if UnaryOp is not specifically matched.
  template<typename UnaryOp, typename XprType>
  struct FunctorTraits<UnaryOp, XprType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<UnaryOp, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_opposite_op<Scalar>, XprType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<std::negate<>, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_abs_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_abs(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_score_coeff_op<Scalar>, XprType>
    : FunctorTraits<Eigen::internal::scalar_abs_op<Scalar>, XprType> {};


  // abs_knowing_score not implemented because it is not a true Eigen functor.


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_abs2_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = internal::constexpr_real(arg);
          auto i = internal::constexpr_imag(arg);
          return r * r + i * i;
        }
        else return arg * arg;
      }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_conjugate_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_conj(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_arg_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept {
      return internal::constexpr_atan2(internal::constexpr_imag(arg), internal::constexpr_real(arg)); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename NewType, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_cast_op<Scalar, NewType>, XprType>
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

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, int N, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_shift_right_op<Scalar, N>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg >> N; } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, int N, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_shift_left_op<Scalar, N>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg << N; } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_real_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_real(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_imag_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_imag(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = not complex_number<Scalar> and hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_real_ref_op<Scalar>, XprType>
    : FunctorTraits<Eigen::internal::scalar_real_op<Scalar>, XprType> {};


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_imag_ref_op<Scalar>, XprType>
    : FunctorTraits<Eigen::internal::scalar_imag_op<Scalar>, XprType> {};


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_exp_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_exp(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_expm1_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_expm1(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_log_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_log1p_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_log1p(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_log10_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        using S = std::decay_t<decltype(internal::constexpr_real(arg))>;
        return internal::constexpr_log(arg) / numbers::ln10_v<S>;
      }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_log2_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        using S = std::decay_t<decltype(internal::constexpr_real(arg))>;
        return internal::constexpr_log(arg) / numbers::ln2_v<S>;
      }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_sqrt_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_sqrt(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_rsqrt_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const { return Scalar{1} / internal::constexpr_sqrt(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero_matrix<XprType> or is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_cos_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_cos(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_sin_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_sin(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_tan_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_tan(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_acos_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_acos(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_asin_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_asin(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_atan_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_atan(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_tanh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_tanh(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_atanh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_atanh(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_sinh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_sinh(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_asinh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_asinh(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_cosh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept
      {
        return internal::constexpr_cosh(arg);
      }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_acosh_op<Scalar>, XprType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg) const noexcept { return internal::constexpr_acosh(arg); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
# endif


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_inverse_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return static_cast<Scalar>(1) / arg; } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero_matrix<XprType> or is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_square_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg * arg; } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_cube_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return arg * arg * arg; } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  // Eigen::internal::scalar_round_op not implemented
  // Eigen::internal::scalar_floor_op not implemented
  // Eigen::internal::scalar_rint_op not implemented (Eigen 3.4+)
  // Eigen::internal::scalar_ceil_op not implemented
  // Eigen::internal::scalar_isnan_op not implemented
  // Eigen::internal::scalar_isinf_op not implemented
  // Eigen::internal::scalar_isfinite_op not implemented


  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_boolean_not_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept { return not static_cast<bool>(arg); } };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };


  // Eigen::internal::scalar_sign_op not implemented


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar, typename XprType>
  struct FunctorTraits<Eigen::internal::scalar_logistic_op<Scalar>, XprType>
  {
    struct Op { constexpr auto operator()(Scalar arg) const noexcept
    {
      return Scalar{1} / (Scalar{1} + (internal::constexpr_exp(-arg))); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, XprType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<XprType, Likelihood::maybe>;
  };
#endif


  template<typename BinaryOp, typename XprType>
  struct FunctorTraits<Eigen::internal::bind1st_op<BinaryOp>, XprType>
  {
    using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
    using ConstantType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
    using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, ConstantType, XprType>;

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {ConstantType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, x, arg.functor()};
      return FunctorTraits<BinaryOp, ConstantType, XprType>::template get_constant<is_diag>(bin);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<BinaryOpType, Likelihood::maybe>;
  };


  template<typename BinaryOp, typename XprType>
  struct FunctorTraits<Eigen::internal::bind2nd_op<BinaryOp>, XprType>
  {
    using CFunctor = Eigen::internal::scalar_constant_op<scalar_type_of_t<XprType>>;
    using ConstantType = Eigen::CwiseNullaryOp<CFunctor, XprType>;
    using BinaryOpType = Eigen::CwiseBinaryOp<BinaryOp, XprType, ConstantType>;

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      const auto& x = arg.nestedExpression();
      BinaryOpType bin {x, ConstantType{x.rows(), x.cols(), CFunctor{arg.functor().m_value}}, arg.functor()};
      return FunctorTraits<BinaryOp, XprType, ConstantType>::template get_constant<is_diag>(bin);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = triangular_matrix<XprType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<BinaryOpType>;
  };


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_UNARY_HPP
