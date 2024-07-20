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
 * \brief Trait details for Eigen binary functors used in PartialReduxExpr.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP
#define OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{

  // Default, if MemberOp is not handled below.
  template<typename MemberOp, std::size_t direction>
  struct ReduxTraits
  {
    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C&, const Factor& factor, const Dim& dim) noexcept
    {
      return std::monostate{};
    }

    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C&, const Factor& factor, const Dim& dim) noexcept
    {
      return std::monostate{};
    }
  };


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename C, typename = void>
    struct const_is_zero : std::false_type {};

    template<typename C>
    struct const_is_zero<C, std::enable_if_t<get_scalar_constant_value(C{}) == 0>> : std::true_type {};
  } // namespace detail
#endif


  /////////////
  //  Norms  //
  /////////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<int p, typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_lpnorm<p, ResultType, Scalar>, direction>
#else
  template<int p, typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_lpnorm<p, ResultType>, direction>
#endif
  {
  private:

    struct Op
    {
      template<typename X>
      constexpr Scalar operator()(X x, std::size_t dim) const noexcept
      {
        auto abs_x = internal::constexpr_abs(x);
        if constexpr (p == 1) return static_cast<Scalar>(dim) * abs_x;
        else if constexpr (p == 2) return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * abs_x;
        else return internal::constexpr_pow(static_cast<Scalar>(dim), static_cast<Scalar>(1)/p) * abs_x;
      }
    };


    struct OpInf
    {
      template<typename X>
      constexpr Scalar operator()(X x) const noexcept { return internal::constexpr_abs(x); }
    };


    template<bool diag, bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_impl(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      if constexpr (p == 0)
      {
        // Note: Eigen does not specifically define the l0 norm, so if p==0 the result is infinity.
        if constexpr (std::numeric_limits<ResultType>::has_infinity) return std::numeric_limits<ResultType>::infinity();
        else throw std::domain_error {"Domain error in lpnorm<0>: result is infinity"};
      }
#ifdef __cpp_concepts
      else if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      else if constexpr (detail::const_is_zero<C>::value)
#endif
      {
        return values::ScalarConstant<ResultType, 0>{};
      }
      else if constexpr (not at_least_square)
      {
        return std::monostate{};
      }
      else if constexpr (p == Eigen::Infinity)
      {
        return values::scalar_constant_operation{OpInf{}, c};
      }
      else if constexpr (diag)
      {
        return values::scalar_constant_operation{Op{}, c, factor};
      }
      else
      {
        return values::scalar_constant_operation{Op{}, c,
          values::scalar_constant_operation{std::multiplies<std::size_t>{}, factor, dim}};
      }
    }

  public:

    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return get_constant_impl<false, true>(c, factor, dim);
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return get_constant_impl<true, at_least_square>(c, factor, dim);
    }
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_stableNorm<ResultType, Scalar>, direction>
    : ReduxTraits<Eigen::internal::member_lpnorm<2, ResultType, Scalar>, direction> {};
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_stableNorm<ResultType>, direction>
    : ReduxTraits<Eigen::internal::member_lpnorm<2, ResultType>, direction> {};
#endif


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_hypotNorm<ResultType, Scalar>, direction>
    : ReduxTraits<Eigen::internal::member_lpnorm<2, ResultType, Scalar>, direction> {};
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_hypotNorm<ResultType>, direction>
    : ReduxTraits<Eigen::internal::member_lpnorm<2, ResultType>, direction> {};
#endif


# if not EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename...Args, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_squaredNorm<Args...>, direction>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = internal::constexpr_real(x);
          auto i = internal::constexpr_imag(x);
          if constexpr (constant_diagonal_matrix<XprType>) return r * r + i * i;
          else return dim * (r * r + i * i);
        }
        else return dim * x * x;
      }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<ResultType, 0>{};
      else
        return values::scalar_constant_operation{Op{}, c,
          values::scalar_constant_operation{std::multiplies<std::size_t>{}, factor, dim}};
    }


    template<bool at_least_square, typename C, typename Factor>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim&) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<ResultType, 0>{};
      else if constexpr (at_least_square)
        return values::scalar_constant_operation{Op{}, c, factor};
      else
        return std::monostate{};
    }
  };


  template<typename...Args, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_norm<Args...>, direction>
  {
    struct Op
    {
      template<typename Scalar>
      constexpr Scalar operator()(Scalar x, std::size_t dim) const noexcept
      {
        if constexpr (complex_number<Scalar>)
        {
          auto r = internal::constexpr_real(x);
          auto i = internal::constexpr_imag(x);
          if constexpr (constant_diagonal_matrix<XprType>) return r * r + i * i;
          return internal::constexpr_sqrt(dim * (r * r + i * i));
        }
        else
        {
          auto arg = internal::constexpr_abs(x);
          return internal::constexpr_sqrt(static_cast<Scalar>(dim)) * arg;
        }
      }
    };

    template<typename XprType, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<ResultType, 0>{};
      else
        return values::scalar_constant_operation{Op{}, c,
          values::scalar_constant_operation{std::multiplies<std::size_t>{}, factor, dim}};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      if constexpr (at_least_square)
        return values::scalar_constant_operation{Op{}, c, factor};
      else
        return std::monostate{};
    }
  };


  template<typename...Args, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_mean<Args...>, direction>
  {
    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim&) noexcept
    {
      return c;
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<ResultType, 0>{};
      else if constexpr (at_least_square)
        return (c * factor) / dim;
      else
        return std::monostate{};
    }
  };
# endif


  ///////////
  //  sum  //
  ///////////

  template<typename T, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<std::plus<T>, Scalar>, direction>
  {
    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
  #ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
  #else
        if constexpr (detail::const_is_zero<C>::value)
  #endif
        return values::ScalarConstant<Scalar, 0>{};
      else
        return values::scalar_constant_operation{std::multiplies<Scalar>{}, c,
          values::scalar_constant_operation{std::multiplies<Scalar>{}, factor, dim}};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
      if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<Scalar, 0>{};
      else if constexpr (at_least_square)
        return values::scalar_constant_operation{std::multiplies<Scalar>{}, c, factor};
      else
        return std::monostate{};
    }
  };


#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_sum<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_sum<ResultType>, direction>
#endif
    : ReduxTraits<Eigen::internal::member_redux<std::plus<Scalar>, Scalar>, direction> {};


  template<typename LhsScalar, typename RhsScalar, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_sum_op<LhsScalar, RhsScalar>, Scalar>, direction>
    : ReduxTraits<Eigen::internal::member_redux<std::plus<Scalar>, Scalar>, direction> {};


  ///////////
  //  min  //
  ///////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_minCoeff<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_minCoeff<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X, typename Dim>
      constexpr auto operator()(X x, Dim dim) const noexcept
      {
        if (dim > 1) return std::min<ResultType>(x, 0);
        else return x;
      }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return c;
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      if constexpr (at_least_square)
      {
        return values::scalar_constant_operation {Op{}, c, dim};
      }
      else if constexpr (scalar_constant<C, ConstantType::static_constant>)
      {
        if constexpr (C::value < 0) return std::monostate{};
        else return values::ScalarConstant<ResultType, 0>{};
      }
      else return std::monostate{};
    }
  };


  template<typename LhsScalar, typename RhsScalar, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_min_op<LhsScalar, RhsScalar>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_minCoeff<Scalar, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_minCoeff<Scalar>, direction> {};
#endif


  ///////////
  //  max  //
  ///////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_maxCoeff<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_maxCoeff<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X, typename Dim>
      constexpr auto operator()(X x, Dim dim) const noexcept
      {
        if (dim > 1) return std::max<ResultType>(x, 0);
        else return x;
      }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return c;
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      if constexpr (at_least_square)
      {
        return values::scalar_constant_operation {Op{}, c, dim};
      }
      else if constexpr (scalar_constant<C, ConstantType::static_constant>)
      {
        if constexpr (C::value > 0) return std::monostate{};
        else return values::ScalarConstant<ResultType, 0>{};
      }
      else return std::monostate{};
    }
  };


  template<typename LhsScalar, typename RhsScalar, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_max_op<LhsScalar, RhsScalar>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_maxCoeff<Scalar, Scalar>, direction> {};
#else
  : ReduxTraits<Eigen::internal::member_maxCoeff<Scalar>, direction> {};
#endif


  ///////////
  //  and  //
  ///////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_all<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_all<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X>
      constexpr bool operator()(X x) const noexcept { return static_cast<bool>(x); }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor&, const Dim&) noexcept
    {
      return values::scalar_constant_operation {Op{}, c};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C&, const Factor&, const Dim&) noexcept
    {
      return std::false_type{};
    }
  };


  template<typename T, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<std::logical_and<T>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_all<bool, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_all<bool>, direction> {};
#endif


  template<typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_boolean_and_op, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_all<bool, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_all<bool>, direction> {};
#endif


  //////////
  //  or  //
  //////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_any<ResultType, Scalar>, direction>
#else
    template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_any<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X>
      constexpr bool operator()(X x) const noexcept { return static_cast<bool>(x); }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return values::scalar_constant_operation {Op{}, c};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      if constexpr (at_least_square)
        return values::scalar_constant_operation {Op{}, c};
      else
        return std::monostate{};
    }
  };


  template<typename T, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<std::logical_or<T>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_any<bool, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_any<bool>, direction> {};
#endif


  template<typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_boolean_or_op, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_any<bool, Scalar>, direction> {};
#else
  : ReduxTraits<Eigen::internal::member_any<bool>, direction> {};
#endif


  /////////////
  //  count  //
  /////////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_count<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_count<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X>
      constexpr Eigen::Index operator()(X x, std::size_t dim) const noexcept
      {
        return static_cast<bool>(x) ? static_cast<Eigen::Index>(dim) : Eigen::Index{0};
      }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return values::scalar_constant_operation {Op{}, c,
        values::scalar_constant_operation{std::multiplies<std::size_t>{}, dim, factor}};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim&) noexcept
    {
      if constexpr (at_least_square)
        return values::scalar_constant_operation {Op{}, c, factor};
      else
        return std::monostate{};
    }
  };


  ///////////////
  //  product  //
  ///////////////

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  template<typename ResultType, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_prod<ResultType, Scalar>, direction>
#else
  template<typename ResultType, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_prod<ResultType>, direction>
#endif
  {
    struct Op
    {
      template<typename X>
      constexpr Scalar operator()(X x, std::size_t dim) const noexcept { return internal::constexpr_pow(x, dim); }
    };


    template<typename C, typename Factor, typename Dim>
    static constexpr auto get_constant(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
#ifdef __cpp_concepts
      if constexpr (requires { requires get_scalar_constant_value(C{}) == 0; })
#else
        if constexpr (detail::const_is_zero<C>::value)
#endif
        return values::ScalarConstant<ResultType, 0>{};
      else
        return values::scalar_constant_operation{Op{}, c,
          values::scalar_constant_operation{std::multiplies<std::size_t>{}, factor, dim}};
    }


    template<bool at_least_square, typename C, typename Factor, typename Dim>
    static constexpr auto get_constant_diagonal(const C& c, const Factor& factor, const Dim& dim) noexcept
    {
      return values::ScalarConstant<ResultType, 0>{};
    }
  };


  template<typename LhsScalar, typename RhsScalar, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<Eigen::internal::scalar_product_op<LhsScalar, RhsScalar>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_prod<Scalar, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_prod<Scalar>, direction> {};
#endif


  template<typename T, typename Scalar, std::size_t direction>
  struct ReduxTraits<Eigen::internal::member_redux<std::multiplies<T>, Scalar>, direction>
#if EIGEN_VERSION_AT_LEAST(3,4,0)
    : ReduxTraits<Eigen::internal::member_prod<Scalar, Scalar>, direction> {};
#else
    : ReduxTraits<Eigen::internal::member_prod<Scalar>, direction> {};
#endif

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN_TRAITS_FUNCTORS_REDUX_HPP
