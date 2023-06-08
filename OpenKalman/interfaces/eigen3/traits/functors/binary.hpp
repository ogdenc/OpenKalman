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
    template<typename Op, typename LhsType, typename RhsType, bool is_diag, typename Arg>
    static constexpr auto default_get_constant(const Arg& arg)
    {
      if constexpr (is_diag)
      {
        if constexpr (std::is_default_constructible_v<Op> and
            constant_diagonal_matrix<LhsType, CompileTimeStatus::known, Likelihood::maybe> and
            constant_diagonal_matrix<RhsType, CompileTimeStatus::known, Likelihood::maybe>)
          return scalar_constant_operation {Op{}, constant_diagonal_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
        else if constexpr (constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and
                           constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
          return scalar_constant_operation {arg.functor(), constant_diagonal_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
        else
          return std::monostate {};
      }
      else
      {
        if constexpr (std::is_default_constructible_v<Op> and
            constant_matrix<LhsType, CompileTimeStatus::known, Likelihood::maybe> and
            constant_matrix<RhsType, CompileTimeStatus::known, Likelihood::maybe>)
          return scalar_constant_operation {Op{}, constant_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
        else if constexpr (constant_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and
                           constant_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
        {
          return scalar_constant_operation {arg.functor(), constant_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
        }
        else
          return std::monostate {};
      }
    }


    template<typename Op, typename LhsType, typename RhsType, bool is_diag, typename Arg>
    static constexpr auto get_constant_sum_impl(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>)
      {
        if constexpr (is_diag) return constant_diagonal_coefficient {arg.rhs()};
        else return constant_coefficient {arg.rhs()};
      }
      else if constexpr (zero_matrix<RhsType>)
      {
        if constexpr (is_diag) return constant_diagonal_coefficient {arg.lhs()};
        else return constant_coefficient {arg.lhs()};
      }
      else return default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
    }


    template<typename Op, typename LhsType, typename RhsType, bool is_diag, typename Arg>
    static constexpr auto get_constant_product_impl(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>)
      {
        return constant_coefficient {arg.lhs()};
      }
      else if constexpr (zero_matrix<RhsType>)
      {
        return constant_coefficient {arg.rhs()};
      }
      else if constexpr (is_diag and constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and constant_matrix<RhsType>)
      {
        if constexpr (std::is_default_constructible_v<Op> and
            constant_diagonal_matrix<LhsType, CompileTimeStatus::known, Likelihood::maybe> and constant_matrix<RhsType, CompileTimeStatus::known>)
          return scalar_constant_operation {Op{}, constant_diagonal_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
        else
          return scalar_constant_operation {arg.functor(), constant_diagonal_coefficient {arg.lhs()}, constant_coefficient {arg.rhs()}};
      }
      else if constexpr (is_diag and constant_matrix<LhsType> and constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (std::is_default_constructible_v<Op> and
            constant_matrix<LhsType, CompileTimeStatus::known, Likelihood::maybe> and constant_diagonal_matrix<RhsType, CompileTimeStatus::known, Likelihood::maybe>)
          return scalar_constant_operation {Op{}, constant_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
        else
          return scalar_constant_operation {arg.functor(), constant_coefficient {arg.lhs()}, constant_diagonal_coefficient {arg.rhs()}};
      }
      else
      {
        return default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
      }
    }


    template<typename Arg1, typename Arg2, TriangleType t, Likelihood b>
    static constexpr bool is_triangular_sum =
      triangular_matrix<Arg1, t, Likelihood::maybe> and triangular_matrix<Arg2, t, Likelihood::maybe> and
      (t != TriangleType::any or
        (triangular_matrix<Arg1, TriangleType::upper, Likelihood::maybe> and triangular_matrix<Arg2, TriangleType::upper, Likelihood::maybe>) or
        (triangular_matrix<Arg1, TriangleType::lower, Likelihood::maybe> and triangular_matrix<Arg2, TriangleType::lower, Likelihood::maybe>)) and
      (b != Likelihood::definitely or triangular_matrix<Arg1, t, b> or triangular_matrix<Arg2, t, b>);


    template<typename Arg1, typename Arg2, TriangleType t, Likelihood b>
    static constexpr bool is_triangular_product =
      triangular_matrix<Arg1, t, b> or triangular_matrix<Arg2, t, b> or
      (((triangular_matrix<Arg1, TriangleType::lower, Likelihood::maybe> and triangular_matrix<Arg2, TriangleType::upper, Likelihood::maybe>) or
      (triangular_matrix<Arg1, TriangleType::upper, Likelihood::maybe> and triangular_matrix<Arg2, TriangleType::lower, Likelihood::maybe>))
        and (square_matrix<Arg1, b> or square_matrix<Arg2, b>));

  } // namespace detail


  // Default binary traits, if BinaryOp is not specifically matched.
  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct FunctorTraits<BinaryOp, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<BinaryOp, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = false;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_sum_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<std::plus<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_product_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<std::multiplies<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_product<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_conj_product_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return constexpr_conj(arg1) * arg2; }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_product<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_min_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return std::min(arg1, arg2); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_max_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return std::max(arg1, arg2); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
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

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else return detail::default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
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

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar, typename Exponent, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_pow_op<Scalar, Exponent>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar arg1, Exponent arg2) const noexcept { return internal::constexpr_pow(arg1, arg2); }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag) return std::monostate {};
      else if constexpr (zero_matrix<RhsType>) return ScalarConstant<Likelihood::definitely, Scalar, 1>{};
      else return detail::default_get_constant<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_difference_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (zero_matrix<LhsType>)
      {
        if constexpr (is_diag) return scalar_constant_operation {std::negate<>{}, constant_diagonal_coefficient {arg.rhs()}};
        else return scalar_constant_operation {std::negate<>{}, constant_coefficient {arg.rhs()}};
      }
      else if constexpr (zero_matrix<RhsType>)
      {
        if constexpr (is_diag) return constant_diagonal_coefficient {arg.lhs()};
        else return constant_coefficient {arg.lhs()};
      }
      else return detail::default_get_constant<std::minus<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_quotient_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (is_diag or zero_matrix<RhsType>) return std::monostate {};
      else return detail::default_get_constant<std::divides<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_and_op, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_product_impl<std::logical_and<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_product<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_or_op, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      if constexpr (not is_diag and
        constant_diagonal_matrix<LhsType, CompileTimeStatus::any, Likelihood::maybe> and
        constant_matrix<RhsType, CompileTimeStatus::known, Likelihood::maybe>)
      {
        if constexpr (constant_coefficient_v<RhsType> == true) return internal::ScalarConstant<Likelihood::maybe, bool, true>{};
        else return constant_diagonal_coefficient {arg.lhs()};
      }
      else if constexpr (not is_diag and
        constant_matrix<LhsType, CompileTimeStatus::known, Likelihood::maybe> and
        constant_diagonal_matrix<RhsType, CompileTimeStatus::any, Likelihood::maybe>)
      {
        if constexpr (constant_coefficient_v<LhsType> == true) return internal::ScalarConstant<Likelihood::maybe, bool, true>{};
        else return constant_diagonal_coefficient {arg.rhs()};
      }
      else
      {
        return detail::get_constant_sum_impl<std::logical_or<>, LhsType, RhsType, is_diag>(arg);
      }
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


  template<typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_boolean_xor_op, LhsType, RhsType>
  {
    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::default_get_constant<std::not_equal_to<>, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename Scalar1, typename Scalar2, typename LhsType, typename RhsType>
  struct FunctorTraits<EGI::scalar_absolute_difference_op<Scalar1, Scalar2>, LhsType, RhsType>
  {
    struct Op
    {
      constexpr auto operator()(Scalar1 arg1, Scalar2 arg2) const noexcept { return arg2 > arg1 ? arg2 - arg1 : arg1 - arg2; }
    };

    template<bool is_diag, typename Arg>
    static constexpr auto get_constant(const Arg& arg)
    {
      return detail::get_constant_sum_impl<Op, LhsType, RhsType, is_diag>(arg);
    }

    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = detail::is_triangular_sum<LhsType, RhsType, t, b>;

    static constexpr bool is_hermitian = hermitian_matrix<LhsType, Likelihood::maybe> and hermitian_matrix<RhsType, Likelihood::maybe>;
  };
#endif

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_TRAITS_FUNCTORS_BINARY_HPP
