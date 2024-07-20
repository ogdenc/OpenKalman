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
 * \file \internal
 * \brief Overloaded constexpr math functions.
 */

#ifndef OPENKALMAN_MATH_CONSTEXPR_HPP
#define OPENKALMAN_MATH_CONSTEXPR_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>


#ifdef __cpp_lib_is_constant_evaluated
#define NOTCONSTANTEVALUATED not std::is_constant_evaluated()
#else
#define NOTCONSTANTEVALUATED false
#endif

#ifdef __cpp_concepts
#define FUNCTIONEXISTSTEST(F)
#define FUNCTIONEXISTSTEST2(F)
#define IFFUNCTIONISINVOCABLEWITHARG(F) requires { F(arg); }
#define IFFUNCTIONISINVOCABLEWITHARG2(F) requires { F(arg1, arg2); }
#else
#define FUNCTIONEXISTSTEST(F)                               \
                                                            \
  template<typename Arg, typename = void>                   \
  struct F##_exists : std::false_type {};                   \
                                                            \
  using std::F;                                             \
                                                            \
  template<typename Arg>                                    \
  struct F##_exists<Arg, std::void_t<decltype(F(std::declval<const Arg&>()))>> : std::true_type {};

#define FUNCTIONEXISTSTEST2(F)                              \
  template<typename Arg1, typename Arg2, typename = void>   \
  struct F##_exists : std::false_type {};                   \
                                                            \
  using std::F;                                             \
                                                            \
  template<typename Arg1, typename Arg2>                    \
  struct F##_exists<Arg1, Arg2, std::void_t<decltype(F(std::declval<const Arg1&>(), std::declval<const Arg2&>()))>> : std::true_type {};

#define IFFUNCTIONISINVOCABLEWITHARG(F) detail::F##_exists<Arg>::value
#define IFFUNCTIONISINVOCABLEWITHARG2(F) detail::F##_exists<Arg1, Arg2>::value
#endif


#if __cpp_lib_constexpr_cmath >= 202306L
#define FUNCTIONISCONSTEXPRCALLABLE(F)                                                                       \
  namespace detail                                                                                           \
  {                                                                                                          \
    template<typename Arg>                                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg& arg)                                                 \
    {                                                                                                        \
      using std::F;                                                                                          \
      return std::tuple {true, F(arg)};                                                                      \
    }                                                                                                        \
  }
#else
#define FUNCTIONISCONSTEXPRCALLABLE(F)                                                                       \
  namespace detail                                                                                           \
  {                                                                                                          \
    FUNCTIONEXISTSTEST(F)                                                                                    \
                                                                                                             \
    template<typename Arg>                                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg& arg)                                                 \
    {                                                                                                        \
      using std::F;                                                                                          \
      if constexpr (IFFUNCTIONISINVOCABLEWITHARG(F))                                                         \
      {                                                                                                      \
        using Scalar = std::decay_t<decltype(F(std::declval<const Arg&>()))>;                                \
        struct Op { constexpr auto operator()(const Arg& arg) { using std::F; return F(arg); } };            \
                                                                                                             \
        if (NOTCONSTANTEVALUATED or constexpr_callable<Op, Arg> or arg != arg)                               \
          return std::tuple {true, F(arg)};                                                                  \
                                                                                                             \
        if constexpr (std::numeric_limits<Arg>::has_infinity)                                                \
          if (arg == std::numeric_limits<Arg>::infinity() or arg == -std::numeric_limits<Arg>::infinity())   \
            return std::tuple {true, F(arg)};                                                                \
                                                                                                             \
        return std::tuple {false, Scalar{}};                                                                 \
      }                                                                                                      \
      else                                                                                                   \
      {                                                                                                      \
        if constexpr (complex_number<Arg>) return std::tuple {false, Arg{}};                                 \
        else                                                                                                 \
        {                                                                                                    \
          using Scalar = std::decay_t<decltype(detail::convert_to_floating(std::declval<const Arg&>()))>;    \
          return std::tuple {false, Scalar{}};                                                               \
        }                                                                                                    \
      }                                                                                                      \
    }                                                                                                        \
  }
#endif


#if __cpp_lib_constexpr_cmath >= 202306L
#define FUNCTIONISCONSTEXPRCALLABLE2(F)                                                                      \
  namespace detail                                                                                           \
  {                                                                                                          \
    template<typename Arg>                                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg1& arg1, const Arg2& arg2)                             \
    {                                                                                                        \
      using std::F;                                                                                          \
      return std::tuple {true, F(arg1, arg2)};                                                               \
    }                                                                                                        \
  }
#else
#define FUNCTIONISCONSTEXPRCALLABLE2(F)                                                                      \
  namespace detail                                                                                           \
  {                                                                                                          \
    FUNCTIONEXISTSTEST2(F)                                                                                   \
                                                                                                             \
    template<typename Arg1, typename Arg2>                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg1& arg1, const Arg2& arg2)                             \
    {                                                                                                        \
      using std::F;                                                                                          \
      if constexpr (IFFUNCTIONISINVOCABLEWITHARG2(F))                                                        \
      {                                                                                                      \
        using Scalar = std::decay_t<decltype(F(std::declval<const Arg1&>(), std::declval<const Arg2&>()))>;  \
        struct Op                                                                                            \
        {                                                                                                    \
          constexpr auto operator()(const Arg1& arg1, const Arg2& arg2)                                      \
          { using std::F; return F(arg1, arg2); }                                                            \
        };                                                                                                   \
                                                                                                             \
        if (NOTCONSTANTEVALUATED or constexpr_callable<Op, Arg1, Arg2> or arg1 != arg1 or arg2 != arg2)      \
          return std::tuple {true, F(arg1, arg2)};                                                           \
                                                                                                             \
        if constexpr (std::numeric_limits<Arg1>::has_infinity)                                               \
          if (arg1 == std::numeric_limits<Arg1>::infinity() or arg1 == -std::numeric_limits<Arg1>::infinity()) \
            return std::tuple {true, F(arg1, arg2)};                                                         \
                                                                                                             \
        if constexpr (std::numeric_limits<Arg2>::has_infinity)                                               \
          if (arg2 == std::numeric_limits<Arg2>::infinity() or arg2 == -std::numeric_limits<Arg2>::infinity()) \
            return std::tuple {true, F(arg1, arg2)};                                                         \
                                                                                                             \
        return std::tuple {false, Scalar{}};                                                                 \
      }                                                                                                      \
      else                                                                                                   \
      {                                                                                                      \
        if constexpr (complex_number<Arg1>) return std::tuple {false, Arg1{}};                               \
        else if constexpr (complex_number<Arg2>) return std::tuple {false, Arg2{}};                          \
        else                                                                                                 \
        {                                                                                                    \
          using Scalar = std::decay_t<decltype(detail::convert_to_floating(                                  \
            std::declval<Arg1>() * std::declval<Arg2>()))>;                                                  \
          return std::tuple {false, Scalar{}};                                                               \
        }                                                                                                    \
      }                                                                                                      \
    }                                                                                                        \
  }
#endif


namespace OpenKalman::internal
{

  namespace detail
  {
    // Convert integral to floating or complex integral to complex floating
    template<typename T>
    constexpr decltype(auto) convert_to_floating(T&& x)
    {
      using std::real, std::imag;
      if constexpr (complex_number<T>)
      {
        using Scalar = std::decay_t<decltype(make_complex_number(convert_to_floating(real(x)), convert_to_floating(imag(x))))>;
        if constexpr (std::is_same_v<std::decay_t<T>, Scalar>) return std::forward<T>(x);
        else return make_complex_number<Scalar>(real(x), imag(x));
      }
      else return real(std::forward<T>(x));
    }


    // Convert a scalar type to an output typt
    template<typename Scalar, typename T>
    constexpr decltype(auto) convert_to_output(T&& x)
    {
      if constexpr (complex_number<Scalar>)
      {
        using std::real, std::imag;
        if constexpr (std::is_same_v<std::decay_t<T>, Scalar>) return std::forward<T>(x);
        else return make_complex_number<Scalar>(real(x), imag(x));
      }
      else if constexpr (std::is_same_v<Scalar, std::decay_t<T>>) return std::forward<T>(x);
      else return static_cast<Scalar>(std::forward<T>(x));
    }
  } // namespace detail


#ifndef __cpp_concepts
  namespace detail
  {
    // Determine whether operator F can be constant-evaluated.
    template<typename F, typename = void, typename...Args>
    struct is_constexpr_callable : std::false_type {};

    template<typename F, typename...Args>
    struct is_constexpr_callable<F, std::void_t<std::bool_constant<(F{}(Args{}...), true)>>, Args...> : std::true_type {};
  }
#endif


  template<typename F, typename...Args>
#ifdef __cpp_concepts
  concept constexpr_callable = requires { typename std::bool_constant<(F{}(Args{}...), true)>; };
#else
  static constexpr bool constexpr_callable = detail::is_constexpr_callable<F, void, Args...>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename = void>
    struct real_trait_exists : std::false_type {};

    template<typename Arg>
    struct real_trait_exists<Arg, std::void_t<decltype(interface::scalar_traits<Arg>::real(std::declval<const Arg&>()))>>
      : std::true_type {};
  }
#endif


  FUNCTIONISCONSTEXPRCALLABLE(real)

  /**
   * \internal
   * \brief A constexpr function to obtain the real part of a (complex) number.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_real(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_real(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_real(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else if constexpr (complex_number<Arg>)
    {
      using Arg_t = std::decay_t<Arg>;

      auto [is_callable, ret] = detail::real_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

  #ifdef __cpp_concepts
      if constexpr (requires(const Arg_t& a) { interface::scalar_traits<Scalar>::real(a); })
  #else
      if constexpr (detail::real_trait_exists<Arg_t>::value)
  #endif
        return static_cast<Scalar>(interface::scalar_traits<Arg_t>::real(std::forward<Arg>(arg)));
      else
      {
        using std::real;
        return real(std::forward<Arg>(arg));
      }
    }
    else return detail::convert_to_floating(std::forward<Arg>(arg));
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename = void>
    struct imag_trait_exists : std::false_type {};

    template<typename Arg>
    struct imag_trait_exists<Arg, std::void_t<decltype(interface::scalar_traits<Arg>::imag(std::declval<const Arg&>()))>>
      : std::true_type {};
  }
#endif


  FUNCTIONISCONSTEXPRCALLABLE(imag)

  /**
   * \internal
   * \brief A constexpr function to obtain the imaginary part of a (complex) number.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_imag(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_imag(T&& arg)
#endif
  {
    using Arg = decltype(arg);
    using Arg_t = std::decay_t<Arg>;

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_imag(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else if constexpr (complex_number<Arg>)
    {
      auto [is_callable, ret] = detail::imag_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

  #ifdef __cpp_concepts
      if constexpr (requires(const Arg_t& a) { interface::scalar_traits<Scalar>::imag(a); })
  #else
      if constexpr (detail::imag_trait_exists<Arg_t>::value)
  #endif
        return static_cast<Scalar>(interface::scalar_traits<Arg_t>::imag(std::forward<Arg>(arg)));
      else
      {
        using std::imag;
        return imag(std::forward<Arg>(arg));
      }
    }
    else return detail::convert_to_floating(Arg_t{0});
  }


  FUNCTIONISCONSTEXPRCALLABLE(conj)

  /**
   * \internal
   * \brief A constexpr function for the complex conjugate of a (complex) number.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_conj(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_conj(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_conj(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::conj_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if constexpr (complex_number<Scalar>)
      {
        return make_complex_number<Scalar>(constexpr_real(arg), -constexpr_imag(arg));
      }
      else
      {
        static_assert(not complex_number<Arg>);
        return std::forward<Arg>(arg);
      }
    }
  }


  /**
   * \internal
   * \brief Return a NaN in type T or raise an exception if Nan is not available.
   */
#ifdef __cpp_concepts
  template <scalar_type T>
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
#endif
  std::decay_t<T> constexpr_NaN()
  {
    if constexpr (complex_number<T>)
    {
      using R = std::decay_t<decltype(constexpr_real(std::declval<T>()))>;
      return make_complex_number<T>(constexpr_NaN<R>(), constexpr_NaN<R>());
    }
    else
    {
      using R = std::decay_t<T>;
      if constexpr (std::numeric_limits<R>::has_quiet_NaN) return std::numeric_limits<R>::quiet_NaN();
      else if constexpr (std::numeric_limits<R>::has_signaling_NaN) return std::numeric_limits<R>::signaling_NaN();
      else throw std::domain_error {"Domain error in arithmetic operation: result is not a number"};
    }
  }


  /**
   * \internal
   * \brief Return +infinity in type T or raise an exception if infinity is not available.
   */
#ifdef __cpp_concepts
  template <scalar_type T>
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
#endif
  std::decay_t<T> constexpr_infinity()
  {
    using R = std::decay_t<T>;
    if constexpr (std::numeric_limits<R>::has_infinity) return std::numeric_limits<R>::infinity();
    else throw std::domain_error {"Domain error in arithmetic operation: result is infinity"};
  }


  FUNCTIONISCONSTEXPRCALLABLE(signbit)

  /**
   * \internal
   * \brief A constexpr function for signbit.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_signbit(scalar_constant auto&& arg) requires (not complex_number<decltype(get_scalar_constant_value(arg))>)
#else
  template <typename T, std::enable_if_t<scalar_constant<T> and not complex_number<decltype(get_scalar_constant_value(std::declval<T>()))>, int> = 0>
  constexpr auto constexpr_signbit(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_signbit(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::signbit_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;
      static_assert(std::is_same_v<Scalar, bool>, "signbit function must return bool");

      return arg < 0;
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE2(copysign)

  /**
   * \internal
   * \brief A constexpr function for copysign.
   */
#ifdef __cpp_concepts
  template<scalar_constant Mag, scalar_constant Sgn> requires
    (not complex_number<decltype(get_scalar_constant_value(std::declval<Mag>()))>) and
    (not complex_number<decltype(get_scalar_constant_value(std::declval<Sgn>()))>) and
    (std::same_as<std::decay_t<decltype(get_scalar_constant_value(std::declval<Mag>()))>, std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>> or
      (std::numeric_limits<std::decay_t<decltype(get_scalar_constant_value(std::declval<Mag>()))>>::is_integer and
        std::numeric_limits<std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>>::is_integer))
#else
  template <typename Mag, typename Sgn, std::enable_if_t<scalar_constant<Mag> and scalar_constant<Sgn> and
    not complex_number<decltype(get_scalar_constant_value(std::declval<Sgn>()))> and not complex_number<decltype(get_scalar_constant_value(std::declval<Sgn>()))> and
    (std::is_same_v<std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>, std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>> or
      (std::numeric_limits<std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>>::is_integer and
        std::numeric_limits<std::decay_t<decltype(get_scalar_constant_value(std::declval<Sgn>()))>>::is_integer)), int> = 0>
#endif
  constexpr auto constexpr_copysign(Mag&& mag, Sgn&& sgn)
  {
    if constexpr (not scalar_type<Mag> or not scalar_type<Sgn>)
    {
      struct Op
      {
        using M = std::decay_t<decltype(get_scalar_constant_value(mag))>;
        using S = std::decay_t<decltype(get_scalar_constant_value(sgn))>;
        constexpr auto operator()(const M& m, const S& s) const { return constexpr_copysign(m, s); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Mag>(mag), std::forward<Sgn>(sgn)};
    }
    else
    {
      auto [is_callable, ret] = detail::copysign_is_constexpr_callable(mag, sgn);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      Scalar x = detail::convert_to_output<Scalar>(mag);
      return constexpr_signbit(x) == constexpr_signbit(sgn) ? x : -x;
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(sqrt)

  /**
   * \internal
   * \brief A constexpr square root function.
   * \details Uses the Newton-Raphson method
   * \tparam Scalar The scalar type.
   * \param x The operand.
   * \return The square root of x.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_sqrt(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_sqrt(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_sqrt(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::sqrt_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        // Find the principal square root
        auto a{detail::convert_to_floating(constexpr_real(arg))};
        auto b{detail::convert_to_floating(constexpr_imag(arg))};
        using R = std::decay_t<decltype(a)>;
        auto nx = constexpr_sqrt(a * a + b * b);
        auto sqp = constexpr_sqrt(R{0.5} * (nx + a));
        auto sqm = constexpr_sqrt(R{0.5} * (nx - a));
        return make_complex_number<Scalar>(sqp, b >= R{0} ? sqm : -sqm);
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if (x <= Scalar{0})
        {
          if (x == Scalar{0}) return std::forward<decltype(x)>(x);
          else return constexpr_NaN<Scalar>();
        }
        else
        {
          if constexpr (std::numeric_limits<Scalar>::has_infinity)
            if (x == std::numeric_limits<Scalar>::infinity()) return std::numeric_limits<Scalar>::infinity();

          Scalar next{Scalar{0.5} * std::forward<decltype(x)>(x)};
          Scalar previous{0};
          while (next != previous)
          {
            previous = next;
            next = Scalar{0.5} * (previous + std::forward<decltype(x)>(x) / previous);
          }
          return next;
        }
      }

      /** // Code for a purely integral version:
      T lo = 0 , hi = x / 2 + 1;
      while (lo != hi) { const T mid = (lo + hi + 1) / 2; if (x / mid < mid) hi = mid - 1; else lo = mid; }
      return lo;*/
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(abs)

  /**
   * \internal
   * \brief A constexpr function for the absolute value.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_abs(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_abs(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_abs(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::abs_is_constexpr_callable(arg);
      using Scalar = std::decay_t<decltype(ret)>;

      if constexpr (complex_number<Scalar>)
      {
        auto re{detail::convert_to_floating(constexpr_real(arg))};
        auto im{detail::convert_to_floating(constexpr_imag(arg))};
        using R = std::decay_t<decltype(constexpr_real(arg))>;
        return detail::convert_to_output<R>(constexpr_sqrt(re * re + im * im));
      }
      else
      {
        if (is_callable) return ret;
        else if constexpr (complex_number<decltype(arg)>)
        {
          auto re{constexpr_real(arg)};
          auto im{constexpr_imag(arg)};
          return detail::convert_to_output<Scalar>(constexpr_sqrt(re * re + im * im));
        }
        else
        {
          auto x = detail::convert_to_output<Scalar>(arg);
          return constexpr_signbit(arg) ? -x : x;
        }
      }
    }
  }


  namespace detail
  {
    // Taylor series expansion
    template <typename T>
    constexpr T sin_cos_impl(int i, const T& x, const T& sum, const T& term)
    {
      auto new_sum = sum + term;
      //if (are_within_tolerance(sum, new_sum)) return new_sum;
      if (sum == new_sum) return sum;
      else return sin_cos_impl(i + 2, x, new_sum, term * x * x / static_cast<T>(-i * (i + 1)));
    }


    // Scale a periodic function (e.g., sin or cos) to within ±pi
    template <typename T>
    constexpr T scale_periodic_function(const T& theta)
    {
      constexpr T pi2 {numbers::pi_v<T> * T{2}};
      constexpr T max {static_cast<T>(std::numeric_limits<std::intmax_t>::max())};
      constexpr T lowest {static_cast<T>(std::numeric_limits<std::intmax_t>::lowest())};
      if (theta > -pi2 and theta < pi2)
      {
        return theta;
      }
      else if (theta / pi2 >= lowest and theta / pi2 <= max)
      {
        return theta - static_cast<std::intmax_t>(theta / pi2) * pi2;
      }
      else if (theta > T{0})
      {
        T corr {pi2};
        while ((theta - corr) / pi2 > max) corr *= T{2};
        return scale_periodic_function(theta - corr);
      }
      else
      {
        T corr {pi2};
        while ((theta + corr) / pi2 < lowest) corr *= T{2};
        return scale_periodic_function(theta + corr);
      }
    }


    // Maclaurin series expansion
    template <typename T>
    constexpr T exp_impl(int i, const T& x, const T& sum, const T& term)
    {
      auto new_sum = sum + term;
      if (sum == new_sum) return sum;
      else return exp_impl(i + 1, x, new_sum, term * x / static_cast<T>(i + 1));
    }

    template <typename Scalar, typename X>
    constexpr Scalar integral_exp(X&& x)
    {
      using T = std::decay_t<X>;
      constexpr auto e = numbers::e_v<Scalar>;
      if (x == T{0}) return Scalar{1};
      else if (x == T{1}) return e;
      else if (x < T{0}) return Scalar{1} / integral_exp<Scalar>(-std::forward<X>(x));
      else if (x % T{2} == T{1}) return e * integral_exp<Scalar>(std::forward<X>(x) - T{1}); //< odd
      else { auto ehalf {integral_exp<Scalar>(std::forward<X>(x) / T{2})}; return ehalf * ehalf; } //< even
    }
  } // detail


  FUNCTIONISCONSTEXPRCALLABLE(exp)

  /**
   * \internal
   * \brief Exponential function.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_exp(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_exp(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_exp(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::exp_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (std::numeric_limits<std::decay_t<Arg>>::is_integer)
      {
        return detail::integral_exp<Scalar>(std::forward<Arg>(arg));
      }
      else
      {
        if constexpr (complex_number<Scalar>)
        {
          auto ea = constexpr_exp(constexpr_real(arg));
          auto b = detail::convert_to_floating(constexpr_imag(arg));
          using Rb = std::decay_t<decltype(b)>;

          if constexpr (std::numeric_limits<Rb>::has_infinity)
            if (b == std::numeric_limits<Rb>::infinity() or b == -std::numeric_limits<Rb>::infinity())
              return constexpr_NaN<Scalar>();

          auto theta{detail::scale_periodic_function(std::move(b))};
          auto sinb = detail::sin_cos_impl<Rb>(4, theta, theta, theta * theta * theta / Rb{-6.0});
          auto cosb = detail::sin_cos_impl<Rb>(3, theta, Rb{1}, Rb{-0.5} * theta * theta);
          return make_complex_number<Scalar>(ea * cosb, ea * sinb);
        }
        else
        {
          auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

          if constexpr (std::numeric_limits<Scalar>::has_infinity)
          {
            if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();
            else if (x == -std::numeric_limits<Scalar>::infinity()) return Scalar{0};
          }

          if (x >= Scalar{0} and x < Scalar{1}) return detail::exp_impl<Scalar>(1, x, Scalar{1}, x);
          else
          {
            int x_trunc = static_cast<int>(x) - (x < Scalar{0} ? 1 : 0);
            Scalar x_frac{std::forward<decltype(x)>(x) - static_cast<Scalar>(x_trunc)};
            return detail::integral_exp<Scalar>(x_trunc) * detail::exp_impl<Scalar>(1, x_frac, Scalar{1}, x_frac);
          }
        }
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(expm1)

  /**
   * \internal
   * \brief Exponential function minus 1.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_expm1(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_expm1(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_expm1(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::expm1_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (std::numeric_limits<std::decay_t<Arg>>::is_integer)
      {
        return detail::integral_exp<Scalar>(std::forward<Arg>(arg)) - Scalar{1};
      }
      else
      {
        if constexpr (complex_number<Scalar>)
        {
          auto ea = constexpr_expm1(constexpr_real(arg));
          auto b = detail::convert_to_floating(constexpr_imag(arg));
          using Rb = std::decay_t<decltype(b)>;

          if constexpr (std::numeric_limits<Rb>::has_infinity)
            if (b == std::numeric_limits<Rb>::infinity() or b == -std::numeric_limits<Rb>::infinity())
              return constexpr_NaN<Scalar>();

          auto theta{detail::scale_periodic_function(std::move(b))};
          auto sinb = detail::sin_cos_impl<Rb>(4, theta, theta, theta * theta * theta / Rb{-6.0});
          auto cosbm1 = detail::sin_cos_impl<Rb>(3, theta, Rb{0}, Rb{-0.5} * theta * theta);
          return make_complex_number<Scalar>(ea * (cosbm1 + Rb{1}) + cosbm1, ea * sinb + sinb);
        }
        else
        {
          auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

          if constexpr (std::numeric_limits<Scalar>::has_infinity)
          {
            if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();
            else if (x == -std::numeric_limits<Scalar>::infinity()) return Scalar{-1};
          }

          if (x >= Scalar{0} and x < Scalar{1})
          {
            if (x == Scalar{0}) return x;
            else return detail::exp_impl<Scalar>(1, x, Scalar{0}, x);
          }
          else
          {
            int x_trunc = static_cast<int>(x) - (x < Scalar{0} ? 1 : 0);
            Scalar x_frac = std::forward<decltype(x)>(x) - static_cast<Scalar>(x_trunc);
            auto et = detail::integral_exp<Scalar>(x_trunc) - Scalar{1};
            auto er = detail::exp_impl<Scalar>(1, x_frac, Scalar{0}, x_frac);
            return et * er + et + er;
          }
        }
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(sinh)

  /**
   * \internal
   * \brief Hyperbolic sine.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_sinh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_sinh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_sinh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::sinh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      {
        if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();
        else if (x == -std::numeric_limits<Scalar>::infinity()) return -constexpr_infinity<Scalar>();
      }

      if (x == Scalar{0}) return std::forward<decltype(x)>(x);
      else
      {
        auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        using Xf = std::decay_t<decltype(xf)>;
        auto ex = constexpr_exp(std::forward<decltype(xf)>(xf));

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        return detail::convert_to_output<Scalar>((ex - Xf{1}/ex) * Xf{0.5});
#else
        if constexpr (complex_number<Scalar>)
        {
          using R = std::decay_t<decltype(constexpr_real(xf))>;
          auto exr = constexpr_real(ex);
          auto exi = constexpr_imag(ex);
          if (exi == 0) return detail::convert_to_output<Scalar>((exr - R{1}/exr) * R{0.5});
          else
          {
            auto denom1 = R(1) / (exr*exr + exi*exi);
            return make_complex_number<Scalar>(R{0.5} * exr * (1 - denom1), R{0.5} * exi * (1 + denom1));
          }
        }
        else
        {
          return detail::convert_to_output<Scalar>((ex - Xf{1}/ex) * Xf{0.5});
        }
#endif
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(cosh)

  /**
   * \internal
   * \brief Hyperbolic cosine.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_cosh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_cosh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_cosh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::cosh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
          return std::numeric_limits<Scalar>::infinity();

      auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      using Xf = std::decay_t<decltype(xf)>;
      auto ex = constexpr_exp(std::forward<decltype(xf)>(xf));

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
      return detail::convert_to_output<Scalar>((ex + Xf{1}/ex) * Xf{0.5});
#else
      if constexpr (complex_number<Scalar>)
      {
        using R = std::decay_t<decltype(constexpr_real(xf))>;
        auto exr = constexpr_real(ex);
        auto exi = constexpr_imag(ex);
        if (exi == 0) return detail::convert_to_output<Scalar>((exr + R{1}/exr) * R{0.5});
        else
        {
          auto denom1 = R(1) / (exr*exr + exi*exi);
          return make_complex_number<Scalar>(R{0.5} * exr * (1 + denom1), R{0.5} * exi * (1 - denom1));
        }
      }
      else
      {
        return detail::convert_to_output<Scalar>((ex + Xf{1}/ex) * Xf{0.5});
      }
#endif
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(tanh)

  /**
   * \internal
   * \brief Hyperbolic tangent.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_tanh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_tanh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_tanh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::tanh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      {
        if (x == std::numeric_limits<Scalar>::infinity()) return Scalar{1};
        else if (x == -std::numeric_limits<Scalar>::infinity()) return Scalar{-1};
      }

      if (x == Scalar{0}) return x;
      else
      {
        auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        using Xf = std::decay_t<decltype(xf)>;
        auto ex = constexpr_exp(std::forward<decltype(xf)>(xf));

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        auto ex2 = ex * ex;
        return detail::convert_to_output<Scalar>((ex2 - Xf{1}) / (ex2 + Xf{1}));
#else
        if constexpr (complex_number<Scalar>)
        {
          auto er = constexpr_real(ex);
          auto ei = constexpr_imag(ex);
          auto d = er*er - ei*ei;
          auto b = 2*er*ei;
          auto denom = d*d + b*b + 2*d + 1;
          return make_complex_number<Scalar>((d*d + b*b - 1) / denom, 2 * b / denom);
        }
        else
        {
          auto ex2 = ex * ex;
          return detail::convert_to_output<Scalar>((ex2 - Xf{1}) / (ex2 + Xf{1}));
        }
#endif
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(sin)

#ifdef __cpp_concepts
  constexpr auto constexpr_sin(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_sin(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_sin(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::sin_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        auto re = detail::convert_to_floating(constexpr_real(arg));
        auto im = detail::convert_to_floating(constexpr_imag(arg));
        using Re = std::decay_t<decltype(re)>;

        if constexpr (std::numeric_limits<Re>::has_infinity)
          if (re == std::numeric_limits<Re>::infinity() or re == -std::numeric_limits<Re>::infinity())
            return constexpr_NaN<Scalar>();

        auto theta{detail::scale_periodic_function(std::move(re))};
        auto sinre = detail::sin_cos_impl<Re>(4, theta, theta, theta * theta * theta / Re{-6.0});
        auto cosre = detail::sin_cos_impl<Re>(3, theta, Re{1}, Re{-0.5} * theta * theta);
        return make_complex_number<Scalar>(sinre * constexpr_cosh(im), cosre * constexpr_sinh(im));
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
            return constexpr_NaN<Scalar>();

        auto theta{detail::scale_periodic_function(std::forward<decltype(x)>(x))};
        return detail::sin_cos_impl<Scalar>(4, theta, theta, theta * theta * theta / Scalar{-6.0});
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(cos)

#ifdef __cpp_concepts
  constexpr auto constexpr_cos(scalar_constant auto&& arg)
#else
  template <typename T>
  constexpr auto constexpr_cos(T&& arg, std::enable_if_t<scalar_constant<T>, int> = 0)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_cos(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::cos_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        auto re = detail::convert_to_floating(constexpr_real(arg));
        auto im = detail::convert_to_floating(constexpr_imag(arg));
        using Re = std::decay_t<decltype(re)>;

        if constexpr (std::numeric_limits<Re>::has_infinity)
          if (re == std::numeric_limits<Re>::infinity() or re == -std::numeric_limits<Re>::infinity())
            return constexpr_NaN<Scalar>();

        auto theta{detail::scale_periodic_function(std::move(re))};
        auto sinre = detail::sin_cos_impl<Re>(4, theta, theta, theta * theta * theta / Re{-6.0});
        auto cosre = detail::sin_cos_impl<Re>(3, theta, Re{1}, Re{-0.5} * theta * theta);
        return make_complex_number<Scalar>(cosre * constexpr_cosh(im), -sinre * constexpr_sinh(im));
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
            return constexpr_NaN<Scalar>();

        auto theta{detail::scale_periodic_function(std::forward<decltype(x)>(x))};
        return detail::sin_cos_impl<Scalar>(3, theta, Scalar{1}, Scalar{-0.5} * theta * theta);
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(tan)

#ifdef __cpp_concepts
  constexpr auto constexpr_tan(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_tan(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_tan(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::tan_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
          return constexpr_NaN<Scalar>();

      if (x == Scalar{0}) return x;
      else
      {
        auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        auto sx = constexpr_sin(xf);
        auto cx = constexpr_cos(xf);

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        return detail::convert_to_output<Scalar>(sx / cx);
#else
        if constexpr (complex_number<Scalar>)
        {
          auto sr = constexpr_real(sx);
          auto si = constexpr_imag(sx);
          auto cr = constexpr_real(cx);
          auto ci = constexpr_imag(cx);
          auto denom = cr*cr + ci*ci;
          return make_complex_number<Scalar>((sr*cr + si*ci) / denom, (si*cr - sr*ci) / denom);
        }
        else
        {
          return detail::convert_to_output<Scalar>(sx / cx);
        }
#endif
      }
    }
  }


  namespace detail
  {
    template <typename T>
    constexpr T asin_series(int n, const T& x, const T& sum, const T& term)
    {
      T new_sum {sum + term / static_cast<T>(n)};
      if (sum == new_sum) return sum;
      else return asin_series(n + 2, x, new_sum, term * x * x * static_cast<T>(n)/static_cast<T>(n+1));
    }


    template <typename T>
    constexpr T asin_impl(const T& x)
    {
      constexpr T pi2 = numbers::pi_v<T> * T{0.5};
      constexpr T invsq2 = numbers::sqrt2_v<T> * T{0.5};

      if (T{-invsq2} <= x and x <= T{invsq2}) return detail::asin_series<T>(3, x, x, T{0.5}*x*x*x);
      else if (T{invsq2} < x and x < T{1}) return pi2 - asin_impl(constexpr_sqrt(T{1} - x*x));
      else if (T{-1} < x and x < T{-invsq2}) return -pi2 + asin_impl(constexpr_sqrt(T{1} - x*x));
      else if (x == T{1}) return +pi2;
      else if (x == T{-1}) return -pi2;
      else return constexpr_NaN<T>();
    }


    template <typename T>
    constexpr T atan_impl(const T& x)
    {
      return detail::asin_impl(x / constexpr_sqrt(T{1} + x * x));
    }


    template <typename T>
    constexpr T atan2_impl(const T& y, const T& x)
    {
      constexpr auto pi = numbers::pi_v<T>;

      if constexpr (std::numeric_limits<T>::has_infinity)
      {
        constexpr auto inf = std::numeric_limits<T>::infinity();
        if (y == +inf)
        {
          if (x == +inf) return pi/4;
          else if (x == -inf) return 3*pi/4;
          else return pi/2;
        }
        else if (y == -inf)
        {
          if (x == +inf) return -pi/4;
          else if (x == -inf) return -3*pi/4;
          else return -pi/2;
        }
        else if (x == +inf)
        {
          return constexpr_copysign(T{0}, y);
        }
        else if (x == -inf)
        {
          return constexpr_copysign(pi, y);
        }
      }

      if (x > T{0})
      {
        return detail::atan_impl(static_cast<T>(y) / static_cast<T>(x));
      }
      else if (x < T{0})
      {
        if (y > T{0}) return detail::atan_impl(static_cast<T>(y) / static_cast<T>(x)) + pi;
        else if (y < T{0}) return detail::atan_impl(static_cast<T>(y) / static_cast<T>(x)) - pi;
        else return constexpr_copysign(pi, y);
      }
      else // if (x == Scalar{0})
      {
        if (y > T{0}) return pi/2;
        else if (y < T{0}) return -pi/2;
        else return constexpr_signbit(x) ? constexpr_copysign(pi, y) : constexpr_copysign(T{0}, y);
      }
    }
  } // detail


  namespace detail
  {
    // Halley's method
    template <typename T>
    constexpr T log_impl(const T& x, const T& y0 = 0, int cmp = 0)
    {
      auto expy0 = constexpr_exp(y0);
      auto y1 = y0 + T{2} * (x - expy0) / (x + expy0);
      if constexpr (complex_number<T>)
      {
        if (are_within_tolerance(y1, y0)) return y1;
        else return log_impl(x, y1);
      }
      else
      {
        // Detect when there is a change in direction.
        if (y1 == y0 or (cmp < 0 and y1 > y0) or (cmp > 0 and y1 < y0)) return y1;
        else return log_impl(x, y1, y1 > y0 ? +1 : y1 < y0 ? -1 : 0);
      }
    }


    template <typename T>
    constexpr std::tuple<T, T> log_scaling_gt(const T& x, const T& corr = T{0})
    {
      if (x < T{0x1p+4}) return {x, corr};
      else if (x < T{0x1p+16}) return log_scaling_gt<T>(x * T{0x1p-4}, corr + T{4} * numbers::ln2_v<T>);
      else if (x < T{0x1p+64}) return log_scaling_gt<T>(x * T{0x1p-16}, corr + T{16} * numbers::ln2_v<T>);
      else return log_scaling_gt<T>(x * T{0x1p-64}, corr + T{64} * numbers::ln2_v<T>);
    }


    template <typename T>
    constexpr std::tuple<T, T> log_scaling_lt(const T& x, const T& corr = T{0})
    {
      if (x > T{0x1p-4}) return {x, corr};
      else if (x > T{0x1p-16}) return log_scaling_lt<T>(x * T{0x1p+4}, corr - T{4} * numbers::ln2_v<T>);
      else if (x > T{0x1p-64}) return log_scaling_lt<T>(x * T{0x1p+16}, corr - T{16} * numbers::ln2_v<T>);
      else return log_scaling_lt<T>(x * T{0x1p+64}, corr - T{64} * numbers::ln2_v<T>);
    }
  } // namespace detail


  FUNCTIONISCONSTEXPRCALLABLE(log)

  /**
   * \internal
   * \brief Natural logarithm function.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_log(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_log(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_log(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::log_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        auto re = detail::convert_to_floating(constexpr_real(arg));
        auto im = detail::convert_to_floating(constexpr_imag(arg));
        using R = std::decay_t<decltype(re)>;
        return make_complex_number<Scalar>(R{0.5} * constexpr_log(re * re + im * im), detail::atan2_impl(im, re));
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity()) return std::forward<decltype(x)>(x);

        if (x == Scalar{1}) return Scalar{+0.};
        else if (x == Scalar{0}) return -constexpr_infinity<Scalar>();
        else if (x < Scalar{0}) return constexpr_NaN<Scalar>();
        auto [scaled, corr] = x >= Scalar{0x1p4} ?
          detail::log_scaling_gt(std::forward<decltype(x)>(x)) : detail::log_scaling_lt(std::forward<decltype(x)>(x));
        return detail::log_impl(scaled) + corr;
      }
    }
  }


  namespace detail
  {
    // Taylor series for log(1+x)
    template <typename T>
    constexpr T log1p_impl(int n, const T& x, const T& sum, const T& term)
    {
      T next_sum = sum + x * term / n;
      if (sum == next_sum) return sum;
      else return log1p_impl(n + 1, x, next_sum, term * -x);
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(log1p)

  /**
   * \internal
   * \brief The log1p function, where log1p(x) = log(x+1).
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_log1p(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_log1p(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_log1p(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::log1p_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        auto re = detail::convert_to_floating(constexpr_real(arg));
        auto im = detail::convert_to_floating(constexpr_imag(arg));
        using R = std::decay_t<decltype(re)>;
        return make_complex_number<Scalar>(
          R{0.5} * constexpr_log1p(re * re + R{2} * re + im * im),
            detail::atan2_impl(im, re + R{1}));
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity()) return std::forward<decltype(x)>(x);

        if (x == Scalar{0}) return std::forward<decltype(x)>(x);
        else if (x == Scalar{-1}) return -constexpr_infinity<Scalar>();
        else if (x < Scalar{-1}) return constexpr_NaN<Scalar>();

        if (Scalar{-0x1p-3} < x and x < Scalar{0x1p-3}) return detail::log1p_impl(2, x, x, -x);
        else
        {
          auto [scaled, corr] =
            x >= Scalar{0x1p4} ? detail::log_scaling_gt(std::forward<decltype(x)>(x) + Scalar{1}) :
              detail::log_scaling_lt(std::forward<decltype(x)>(x) + Scalar{1});
          return detail::log_impl(scaled) + corr;
        }
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(asinh)

#ifdef __cpp_concepts
  constexpr auto constexpr_asinh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_asinh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_asinh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::asinh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
        return std::forward<decltype(x)>(x);

      if (x == Scalar{0}) return std::forward<decltype(x)>(x);
      else
      {
        auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        using Xf = std::decay_t<decltype(xf)>;

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf * xf + Xf{1})));
#else
        if constexpr (complex_number<Scalar>)
        {
          auto xr = constexpr_real(xf);
          auto xi = constexpr_imag(xf);
          using R = std::decay_t<decltype(xr)>;
          auto sqt = constexpr_sqrt(make_complex_number(xr*xr - xi*xi + R{1}, R{2}*xr*xi));
          auto sqtr = constexpr_real(sqt);
          auto sqti = constexpr_imag(sqt);
          return detail::convert_to_output<Scalar>(constexpr_log(make_complex_number(xr + sqtr, xi + sqti)));
        }
        else
        {
          return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf * xf + Xf{1})));
        }
#endif
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(acosh)

#ifdef __cpp_concepts
  constexpr auto constexpr_acosh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_acosh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_acosh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::acosh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if (x == Scalar{1}) return static_cast<Scalar>(+0.);

      if constexpr (complex_number<Scalar>)
      {
        auto xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        using Xf = std::decay_t<decltype(xf)>;

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf + Xf{1}) * constexpr_sqrt(xf - Xf{1})));
#else
        if constexpr (complex_number<Scalar>)
        {
          auto xr = constexpr_real(xf);
          auto xi = constexpr_imag(xf);
          using R = std::decay_t<decltype(xr)>;
          auto sqtp = constexpr_sqrt(make_complex_number(xr + R{1}, xi));
          auto a = constexpr_real(sqtp);
          auto b = constexpr_imag(sqtp);
          auto sqtm = constexpr_sqrt(make_complex_number(xr - R{1}, xi));
          auto c = constexpr_real(sqtm);
          auto d = constexpr_imag(sqtm);
          return detail::convert_to_output<Scalar>(constexpr_log(make_complex_number(xr + a*c - b*d, xi + a*d + b*c)));
        }
        else
        {
          return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf + Xf{1}) * constexpr_sqrt(xf - Xf{1})));
        }
#endif
      }
      else
      {
        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();

        if (x < Scalar{1}) return constexpr_NaN<Scalar>();
        else return constexpr_log(x + constexpr_sqrt(x * x - Scalar{1}));
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(atanh)

#ifdef __cpp_concepts
  constexpr auto constexpr_atanh(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_atanh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_atanh(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::atanh_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      {
        if (x < Scalar{-1} or x > Scalar{1}) return constexpr_NaN<Scalar>();
        else if (x == Scalar{1}) return constexpr_infinity<Scalar>();
        else if (x == Scalar{-1}) return -constexpr_infinity<Scalar>();
      }

      if (x == Scalar{0}) return std::forward<decltype(x)>(x);
      else
      {
        auto&& xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
        using Xf = std::decay_t<decltype(xf)>;

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        return detail::convert_to_output<Scalar>(constexpr_log((Xf{1} + xf) / (Xf{1} - xf)) * Xf{0.5});
#else
        if constexpr (complex_number<Scalar>)
        {
          auto xr = constexpr_real(xf);
          auto xi = constexpr_imag(xf);
          using R = std::decay_t<decltype(xr)>;

          auto denom = R(1) - R(2)*xr + xr*xr + xi*xi;
          auto lg = constexpr_log(make_complex_number((R(1) - xr*xr - xi*xi) / denom, R(2) * xi / denom));
          return make_complex_number<Scalar>(R(0.5) * constexpr_real(lg), R(0.5) * constexpr_imag(lg));
        }
        else
        {
          return detail::convert_to_output<Scalar>(constexpr_log((Xf{1} + xf) / (Xf{1} - xf)) * Xf{0.5});
        }
#endif
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(asin)

#ifdef __cpp_concepts
  constexpr auto constexpr_asin(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_asin(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_asin(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::asin_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if constexpr (complex_number<Scalar>)
      {
        if (arg != arg) return constexpr_NaN<Scalar>();

        auto&& xf = detail::convert_to_floating(std::forward<Arg>(arg));
        using R = std::decay_t<decltype(constexpr_real(xf))>;

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        using Xf = std::decay_t<decltype(xf)>;
        constexpr auto i = make_complex_number<Xf>(R{0}, R{1});
        return detail::convert_to_output<Scalar>(i * constexpr_log(constexpr_sqrt(Xf{1} - xf * xf) - i * xf));
#else
        auto xr = constexpr_real(xf);
        auto xi = constexpr_imag(xf);
        auto sqt = constexpr_sqrt(make_complex_number(R{1} - xr*xr + xi*xi, -R(2)*xr*xi));
        auto lg = constexpr_log(make_complex_number(constexpr_real(sqt) + xi, constexpr_imag(sqt) - xr));
        return make_complex_number<Scalar>(-constexpr_imag(lg), constexpr_real(lg));
#endif
      }
      else return detail::asin_impl(detail::convert_to_output<Scalar>(std::forward<Arg>(arg)));
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(acos)

#ifdef __cpp_concepts
  constexpr auto constexpr_acos(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_acos(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_acos(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::acos_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();
      else if (arg == std::decay_t<Arg>{1}) return static_cast<Scalar>(+0.);

      auto&& xf = detail::convert_to_floating(std::forward<Arg>(arg));
      using Xf = std::decay_t<decltype(xf)>;
      using R = std::decay_t<decltype(constexpr_real(xf))>;

      auto s = constexpr_asin(xf);
      if (s != s) return constexpr_NaN<Scalar>();

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
      return detail::convert_to_output<Scalar>(Xf{numbers::pi_v<R> / 2} - std::move(s));
#else
      if constexpr (complex_number<Scalar>)
      {
        constexpr R pi2 {numbers::pi_v<R> / 2};
        return make_complex_number<Scalar>(pi2 - constexpr_real(s), - constexpr_imag(s));
      }
      else return detail::convert_to_output<Scalar>(Xf{numbers::pi_v<R> / 2} - std::move(s));
#endif
    }
  }


  namespace detail
  {
    template <typename T>
    constexpr T atan_impl_general(const T& x)
    {
      if constexpr (complex_number<T>)
      {
        using R = std::decay_t<decltype(constexpr_real(x))>;

#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        constexpr auto i = make_complex_number<T>(R{0}, R{1});
        return T{-0.5} * i * constexpr_log((T{1} + x*i)/(T{1} - x*i));
#else
        auto xr = constexpr_real(x);
        auto xi = constexpr_imag(x);
        auto ar = -xi;
        auto ai = xr;
        auto lar = R{0.5} * constexpr_log1p(ar * ar + R{2} * ar + ai * ai);
        auto lai = detail::atan2_impl(ai, ar + R{1});
        auto br = xi;
        auto bi = -xr;
        auto lbr = R{0.5} * constexpr_log1p(br * br + R{2} * br + bi * bi);
        auto lbi = detail::atan2_impl(bi, br + R{1});
        return make_complex_number<T>(R{-0.5} * (-lai + lbi), R{-0.5} * (lar - lbr));
#endif
      }
      else return atan_impl(x);
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(atan)

#ifdef __cpp_concepts
  constexpr auto constexpr_atan(scalar_constant auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
  constexpr auto constexpr_atan(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (not scalar_type<Arg>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        constexpr auto operator()(const A& a) const { return constexpr_atan(a); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::atan_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (arg != arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Scalar>)
      {
        auto x = detail::convert_to_floating(std::forward<Arg>(arg));
        return detail::convert_to_output<Scalar>(detail::atan_impl_general(x));
      }
      else
      {
        auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
        {
          if (x == std::numeric_limits<Scalar>::infinity()) return numbers::pi_v<Scalar> * Scalar{0.5};
          else if (x == -std::numeric_limits<Scalar>::infinity()) return numbers::pi_v<Scalar> * Scalar{-0.5};
        }

        if (x == Scalar{0}) return std::forward<decltype(x)>(x);
        else return detail::atan_impl(std::forward<decltype(x)>(x));
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE2(atan2)

#ifdef __cpp_concepts
  template <scalar_constant Y, scalar_constant X>
#else
  template <typename Y, typename X, std::enable_if_t<scalar_constant<Y> and scalar_constant<X>, int> = 0>
#endif
  constexpr auto constexpr_atan2(Y&& y_arg, X&& x_arg)
  {
    if constexpr (not scalar_type<Y> or not scalar_type<X>)
    {
      struct Op
      {
        using YA = std::decay_t<decltype(get_scalar_constant_value(y_arg))>;
        using XA = std::decay_t<decltype(get_scalar_constant_value(x_arg))>;
        constexpr auto operator()(const YA& y, const XA& x) const { return constexpr_atan2(y, x); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Y>(y_arg), std::forward<X>(x_arg)};
    }
    else
    {
      auto [is_callable, ret] = detail::atan2_is_constexpr_callable(y_arg, x_arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      if (y_arg != y_arg or x_arg != x_arg) return constexpr_NaN<Scalar>();

      if constexpr (complex_number<Y> or complex_number<X>)
      {
        auto&& yf = detail::convert_to_floating(std::forward<Y>(y_arg));
        auto&& xf = detail::convert_to_floating(std::forward<X>(x_arg));

        auto yr = constexpr_real(yf);

        using Sf = std::decay_t<decltype(yf)>;
        using R = std::decay_t<decltype(yr)>;

        constexpr auto pi = numbers::pi_v<R>;

        if (xf == Sf(0))
        {
          if (yr > 0) return Scalar(0.5*pi);
          else if (yr < 0) return Scalar(-0.5*pi);
          else return Scalar{0};
        }
        else if (yf == Sf(0))
        {
          if (constexpr_real(xf) < 0) return Scalar(pi);
          return Scalar(0);
        }
        else
#if __cpp_lib_constexpr_complex >= 201711L and (not defined(__clang__) or __clang_major__ >= 17) // Check this and later versions of clang
        {
          auto raw = detail::atan_impl_general(yf / xf);
          auto raw_r = constexpr_real(raw);
          if (raw_r > pi) return detail::convert_to_output<Scalar>(raw - Sf{pi});
          else if (raw_r < -pi) return detail::convert_to_output<Scalar>(raw + Sf{pi});
          else return detail::convert_to_output<Scalar>(raw);
        }
#else
        {
          auto yi = constexpr_imag(yf);
          auto xi = constexpr_imag(xf);
          auto xr = constexpr_real(xf);
          auto denom = xr*xr + xi*xi;
          auto raw = detail::atan_impl_general(make_complex_number((yr * xr + yi * xi) / denom, (yi * xr - yr * xi) / denom));
          auto raw_r = constexpr_real(raw);
          auto raw_i = constexpr_imag(raw);
          if (raw_r > pi) return make_complex_number<Scalar>(raw_r - pi, raw_i);
          else if (raw_r < -pi) return make_complex_number<Scalar>(raw_r + pi, raw_i);
          else return make_complex_number<Scalar>(raw_r, raw_i);
        }
#endif
      }
      else
        return detail::atan2_impl(
          detail::convert_to_output<Scalar>(std::forward<Y>(y_arg)),
          detail::convert_to_output<Scalar>(std::forward<X>(x_arg)));
    }
  }


  namespace detail
  {
    template <typename T, typename N>
    constexpr T pow_integral(const T& x, const N& n)
    {
      using R = std::conditional_t<complex_number<T>, std::decay_t<decltype(constexpr_real(x))>, T>;

      if (n == 0) return T{1};
      if (x != T{0} and n < 0) return T{1} / pow_integral(x, -n);
      else if (n % 2 == 1) // positive odd
      {
        if (x == T{0}) return x;
        else return x * pow_integral(x, n - 1);
      }
      else if (-n % 2 == 1) // negative odd
      {
        if (x == T{0}) return T{constexpr_copysign(constexpr_infinity<R>(), constexpr_real(x))};
        else return pow_integral(x, n + 1) / x;
      }
      else // positive even or negative even
      {
        if (x == T{0}) return n > 0 ? T{+0.} : T{+constexpr_infinity<R>()};
        else return pow_integral(x, n / 2) * pow_integral(x, n / 2);
      }
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE2(pow)

  /**
   * \internal
   * \brief A constexpr power function.
   * \param x The operand
   * \param n The power
   * \return x to the power of n.
   */
#ifdef __cpp_concepts
  template<scalar_constant Arg, scalar_constant Exponent>
#else
  template <typename Arg, typename Exponent, std::enable_if_t<scalar_constant<Arg> and scalar_constant<Exponent>, int> = 0>
#endif
  constexpr auto constexpr_pow(Arg&& arg, Exponent&& exponent)
  {
    if constexpr (not scalar_type<Arg> or not scalar_type<Exponent>)
    {
      struct Op
      {
        using A = std::decay_t<decltype(get_scalar_constant_value(arg))>;
        using E = std::decay_t<decltype(get_scalar_constant_value(exponent))>;
        constexpr auto operator()(const A& a, const E& e) const { return constexpr_pow(a, e); }
      };
      return values::scalar_constant_operation {Op{}, std::forward<Arg>(arg), std::forward<Exponent>(exponent)};
    }
    else
    {
      auto [is_callable, ret] = detail::pow_is_constexpr_callable(arg, exponent);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

      using T = std::decay_t<Arg>;
      using U = std::decay_t<Exponent>;

      if (arg == T{1} or exponent == U{0}) return Scalar{1};
      else if (arg != arg or exponent != exponent) return constexpr_NaN<Scalar>();

      if constexpr (std::numeric_limits<U>::is_integer)
      {
        if constexpr (std::numeric_limits<T>::has_infinity)
        {
          if (exponent % 2 == 1) // positive odd
          {
            if (arg == -std::numeric_limits<T>::infinity()) return -constexpr_infinity<T>();
            else if (arg == +std::numeric_limits<T>::infinity()) return +constexpr_infinity<T>();
          }
          else if (-exponent % 2 == 1) // negative odd
          {
            if (arg == -std::numeric_limits<T>::infinity()) return T{-0.};
            else if (arg == +std::numeric_limits<T>::infinity()) return T{+0.};
          }
          else if (arg == -std::numeric_limits<T>::infinity() or arg == +std::numeric_limits<T>::infinity())
          {
            if (exponent > 0) return +constexpr_infinity<T>(); // positive even
            else return T{+0.}; // negative even
          }
        }

        return detail::pow_integral(detail::convert_to_output<Scalar>(std::forward<Arg>(arg)), exponent);
      }
      else
      {
        if constexpr (complex_number<Scalar>)
        {
          auto&& xf = detail::convert_to_floating(std::forward<Arg>(arg));
          auto&& nf = detail::convert_to_floating(std::forward<Exponent>(exponent));

          return detail::convert_to_output<Scalar>(constexpr_exp(constexpr_log(std::forward<decltype(xf)>(xf)) * std::forward<decltype(nf)>(nf)));
        }
        else
        {
          auto&& x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));
          auto&& n = detail::convert_to_output<Scalar>(std::forward<Exponent>(exponent));

          if constexpr (std::numeric_limits<Scalar>::has_infinity)
          {
            if (x == -std::numeric_limits<Scalar>::infinity() or x == +std::numeric_limits<Scalar>::infinity())
            {
              // Note: en.cppreference.com/w/cpp/numeric/math/pow says that the sign of both these should be reversed,
              // but both GCC and clang return a result as follows:
              if (n < Scalar{0}) return Scalar{-0.}; else return -constexpr_infinity<Scalar>();
            }
            else if (n == -std::numeric_limits<Scalar>::infinity())
            {
              if (Scalar{-1} < x and x < Scalar{1}) return +constexpr_infinity<Scalar>();
              else if (x < Scalar{-1} or Scalar{1} < x) return Scalar{+0.};
              else return Scalar{1}; // x == -1 (x == 1 case handled above)
            }
            else if (n == +std::numeric_limits<Scalar>::infinity())
            {
              if (Scalar{-1} < x and x < Scalar{1}) return Scalar{+0.};
              else if (x < Scalar{-1} or Scalar{1} < x) return +constexpr_infinity<Scalar>();
              else return Scalar{1}; // x == -1 (x == 1 case handled above)
            }
          }

          if (x > Scalar{0})
          {
            if (n == Scalar{1}) return std::forward<decltype(x)>(x);
            else return constexpr_exp(constexpr_log(std::forward<decltype(x)>(x)) * std::forward<decltype(n)>(n));
          }
          else if (x == 0)
          {
            if (n < 0) return +constexpr_infinity<Scalar>();
            else return Scalar{+0.};
          }
          else return constexpr_NaN<Scalar>();
        }
      }
    }
  }


} // namespace OpenKalman::internal

#undef FUNCTIONISCONSTEXPRCALLABLE2
#undef FUNCTIONEXISTSTEST2
#undef IFFUNCTIONISINVOCABLEWITHARG2
#undef FUNCTIONISCONSTEXPRCALLABLE
#undef FUNCTIONEXISTSTEST
#undef IFFUNCTIONISINVOCABLEWITHARG
#undef NOTCONSTANTEVALUATED

#endif //OPENKALMAN_MATH_CONSTEXPR_HPP
