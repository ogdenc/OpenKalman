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
 * \brief Overloaded consteval math functions.
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


#define FUNCTIONISCONSTEXPRCALLABLE(F)                                                                       \
  namespace detail                                                                                           \
  {                                                                                                          \
    FUNCTIONEXISTSTEST(F)                                                                                    \
                                                                                                             \
    template<typename Arg>                                                                                   \
    struct Op_##F { constexpr auto operator()(const Arg& arg) { using std::F; return F(arg); } };            \
                                                                                                             \
    template<typename Arg>                                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg& arg)                                                 \
    {                                                                                                        \
      using std::F;                                                                                          \
      if constexpr (IFFUNCTIONISINVOCABLEWITHARG(F))                                                         \
      {                                                                                                      \
        using Scalar = std::decay_t<decltype(F(std::declval<const Arg&>()))>;                                \
        if constexpr (constexpr_callable<Scalar, Arg, Op_##F<Arg>>) return std::tuple {true, F(arg)};        \
        else if (NOTCONSTANTEVALUATED) return std::tuple {true, F(arg)};                                     \
        else return std::tuple {false, Scalar{}};                                                            \
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


#define FUNCTIONISCONSTEXPRCALLABLE2(F)                                                                      \
  namespace detail                                                                                           \
  {                                                                                                          \
    FUNCTIONEXISTSTEST2(F)                                                                                   \
                                                                                                             \
    template<typename Arg1, typename Arg2>                                                                   \
    struct Op_##F                                                                                            \
    {                                                                                                        \
      constexpr auto operator()(const Arg1& arg1, const Arg2& arg2) { using std::F; return F(arg1, arg2); }  \
    };                                                                                                       \
                                                                                                             \
    template<typename Arg1, typename Arg2>                                                                   \
    constexpr auto F##_is_constexpr_callable(const Arg1& arg1, const Arg2& arg2)                             \
    {                                                                                                        \
      using std::F;                                                                                          \
      if constexpr (IFFUNCTIONISINVOCABLEWITHARG2(F))                                                        \
      {                                                                                                      \
        using Scalar = std::decay_t<decltype(F(std::declval<const Arg1&>(), std::declval<const Arg2&>()))>;  \
        if constexpr (constexpr_callable<Scalar, Arg1, Arg2, Op_##F<Arg1, Arg2>>)                            \
          return std::tuple {true, F(arg1, arg2)};                                                           \
        else if (NOTCONSTANTEVALUATED) return std::tuple {true, F(arg1, arg2)};                              \
        else return std::tuple {false, Scalar{}};                                                            \
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


    // Determine whether operator F can be constant-evaluated.
    template<typename F, typename = void, typename...Args>
    struct is_constexpr_callable : std::false_type {};

    template<typename F, typename...Args>
    struct is_constexpr_callable<F, std::enable_if_t<(F{}(Args{}...), true)>, Args...> : std::true_type {};
  } // namespace detail


#ifdef __cpp_concepts
  template<typename F, typename...Args>
  concept constexpr_callable = detail::is_constexpr_callable<F, void, Args...>::value;
#else
  template<typename F, typename...Args>
  static constexpr bool constexpr_callable = detail::is_constexpr_callable<F, void, Args...>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename = void>
    struct real_trait_exists : std::false_type {};

    template<typename Arg>
    struct real_trait_exists<Arg, std::void_t<decltype(ScalarTraits<Arg>::real(std::declval<const Arg&>()))>>
      : std::true_type {};
  }
#endif


  FUNCTIONISCONSTEXPRCALLABLE(real)

  /**
   * \internal
   * \brief A constexpr function to obtain the real part of a (complex) number.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_real(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_real(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    if constexpr (complex_number<Arg>)
    {
      using Arg_t = std::decay_t<Arg>;

      auto [is_callable, ret] = detail::real_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

  #ifdef __cpp_concepts
      if constexpr (requires(const Arg_t& a) { ScalarTraits<Scalar>::real(a); })
  #else
      if constexpr (detail::real_trait_exists<Arg_t>::value)
  #endif
        return static_cast<Scalar>(ScalarTraits<Arg_t>::real(std::forward<Arg>(arg)));
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
    struct imag_trait_exists<Arg, std::void_t<decltype(ScalarTraits<Arg>::imag(std::declval<const Arg&>()))>>
      : std::true_type {};
  }
#endif


  FUNCTIONISCONSTEXPRCALLABLE(imag)

  /**
   * \internal
   * \brief A constexpr function to obtain the imaginary part of a (complex) number.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_imag(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_imag(T&& arg)
#endif
  {
    using Arg = decltype(arg);
    using Arg_t = std::decay_t<Arg>;

    if constexpr (complex_number<Arg>)
    {
      auto [is_callable, ret] = detail::imag_is_constexpr_callable(arg);
      if (is_callable) return ret;
      using Scalar = std::decay_t<decltype(ret)>;

  #ifdef __cpp_concepts
      if constexpr (requires(const Arg_t& a) { ScalarTraits<Scalar>::imag(a); })
  #else
      if constexpr (detail::imag_trait_exists<Arg_t>::value)
  #endif
        return static_cast<Scalar>(ScalarTraits<Arg_t>::imag(std::forward<Arg>(arg)));
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
  constexpr auto constexpr_conj(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_conj(T&& arg)
#endif
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
      return make_complex_number(constexpr_real(arg), -constexpr_imag(arg));
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
  constexpr std::decay_t<T> constexpr_NaN()
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
  constexpr std::decay_t<T> constexpr_infinity()
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
  constexpr bool constexpr_signbit(scalar_type auto&& arg) requires (not complex_number<decltype(arg)>)
#else
  template <typename T, std::enable_if_t<scalar_type<T> and not complex_number<T>, int> = 0>
  constexpr bool constexpr_signbit(T&& arg)
#endif
  {
    auto [is_callable, ret] = detail::signbit_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;
    static_assert(std::is_same_v<Scalar, bool>, "signbit function must return bool");

    return arg < 0;
  }


  FUNCTIONISCONSTEXPRCALLABLE2(copysign)

  /**
   * \internal
   * \brief A constexpr function for copysign.
   */
#ifdef __cpp_concepts
  template<scalar_type Mag, scalar_type Sgn> requires (not complex_number<Mag>) and (not complex_number<Sgn>) and
    (std::same_as<std::decay_t<Mag>, std::decay_t<Sgn>> or
      (std::numeric_limits<std::decay_t<Mag>>::is_integer and std::numeric_limits<std::decay_t<Sgn>>::is_integer))
#else
  template <typename Mag, typename Sgn, std::enable_if_t<scalar_type<Mag> and scalar_type<Sgn> and
    not complex_number<Mag> and not complex_number<Sgn> and
    (std::is_same_v<std::decay_t<Mag>, std::decay_t<Sgn>> or
      (std::numeric_limits<std::decay_t<Mag>>::is_integer and std::numeric_limits<std::decay_t<Sgn>>::is_integer)), int> = 0>
#endif
  constexpr auto constexpr_copysign(Mag&& mag, Sgn&& sgn)
  {
    auto [is_callable, ret] = detail::copysign_is_constexpr_callable(mag, sgn);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    Scalar x = detail::convert_to_output<Scalar>(mag);
    return constexpr_signbit(x) == constexpr_signbit(sgn) ? x : -x;
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
  constexpr auto constexpr_sqrt(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_sqrt(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::sqrt_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    if constexpr (complex_number<Scalar>)
    {
      // Find the principal square root
      auto a {detail::convert_to_floating(constexpr_real(arg))};
      auto b {detail::convert_to_floating(constexpr_imag(arg))};
      using R = std::decay_t<decltype(a)>;
      auto nx = constexpr_sqrt(a * a + b * b);
      auto sqp = constexpr_sqrt(R{0.5} * (nx + a));
      auto sqm = constexpr_sqrt(R{0.5} * (nx - a));
      return make_complex_number<Scalar>(sqp, b >= R{0} ? sqm : -sqm);
    }
    else
    {
      decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if (x <= Scalar{0})
      {
        if (x == Scalar{0}) return x;
        else return constexpr_NaN<Scalar>();
      }
      else
      {
        if constexpr (std::numeric_limits<Scalar>::has_infinity)
          if (x == std::numeric_limits<Scalar>::infinity()) return std::numeric_limits<Scalar>::infinity();

        Scalar next {Scalar{0.5} * x};
        Scalar previous {0};
        while (next != previous)
        {
          previous = next;
          next = Scalar{0.5} * (previous + x / previous);
        }
        return next;
      }
    }

    /** // Code for a purely integral version:
    T lo = 0 , hi = x / 2 + 1;
    while (lo != hi) { const T mid = (lo + hi + 1) / 2; if (x / mid < mid) hi = mid - 1; else lo = mid; }
    return lo;*/
  }


  FUNCTIONISCONSTEXPRCALLABLE(abs)

  /**
   * \internal
   * \brief A constexpr function for the absolute value.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_abs(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_abs(T&& arg)
#endif
  {
    auto [is_callable, ret] = detail::abs_is_constexpr_callable(arg);
    using Scalar = std::decay_t<decltype(ret)>;

    if constexpr (complex_number<Scalar>)
    {
      auto re {detail::convert_to_floating(constexpr_real(arg))};
      auto im {detail::convert_to_floating(constexpr_imag(arg))};
      using R = std::decay_t<decltype(constexpr_real(arg))>;
      return detail::convert_to_output<R>(constexpr_sqrt(re*re + im*im));
    }
    else
    {
      if (is_callable) return ret;
      else if constexpr (complex_number<decltype(arg)>)
      {
        auto re {constexpr_real(arg)};
        auto im {constexpr_imag(arg)};
        return detail::convert_to_output<Scalar>(constexpr_sqrt(re*re + im*im));
      }
      else
      {
        auto x = detail::convert_to_output<Scalar>(arg);
        return constexpr_signbit(arg) ? -x : x;
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
      else if (x < T{0}) return Scalar{1} / integral_exp<Scalar>(-std::forward<T>(x));
      else if (x % T{2} == T{1}) return e * integral_exp<Scalar>(std::forward<T>(x) - T{1}); //< odd
      else { auto ehalf {integral_exp<Scalar>(std::forward<T>(x) / T{2})}; return ehalf * ehalf; } //< even
    }
  } // detail


  FUNCTIONISCONSTEXPRCALLABLE(exp)

  /**
   * \internal
   * \brief Exponential function.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_exp(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_exp(T&& arg)
#endif
  {
    using Arg = decltype(arg);

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

        auto theta {detail::scale_periodic_function(std::move(b))};
        auto sinb = detail::sin_cos_impl<Rb>(4, theta, theta, theta * theta * theta / Rb{-6.0});
        auto cosb = detail::sin_cos_impl<Rb>(3, theta, Rb{1}, Rb{-0.5} * theta * theta);
        return make_complex_number<Scalar>(ea * cosb, ea * sinb);
      }
      else
      {
        decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
        {
          if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();
          else if (x == -std::numeric_limits<Scalar>::infinity()) return Scalar{0};
        }

        if (x >= Scalar{0} and x < Scalar{1}) return detail::exp_impl<Scalar>(1, x, Scalar{1}, x);
        else
        {
          int x_trunc = static_cast<int>(x) - (x < Scalar{0} ? 1 : 0);
          Scalar x_frac {x - static_cast<Scalar>(x_trunc)};
          return detail::integral_exp<Scalar>(x_trunc) * detail::exp_impl<Scalar>(1, x_frac, Scalar{1}, x_frac);
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
  constexpr auto constexpr_expm1(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_expm1(T&& arg)
#endif
  {
    using Arg = decltype(arg);

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

        auto theta {detail::scale_periodic_function(std::move(b))};
        auto sinb = detail::sin_cos_impl<Rb>(4, theta, theta, theta * theta * theta / Rb{-6.0});
        auto cosbm1 = detail::sin_cos_impl<Rb>(3, theta, Rb{0}, Rb{-0.5} * theta * theta);
        return make_complex_number<Scalar>(ea * (cosbm1 + Rb{1}) + cosbm1, ea * sinb + sinb);
      }
      else
      {
        decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

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
          Scalar x_frac = x - static_cast<Scalar>(x_trunc);
          auto et = detail::integral_exp<Scalar>(x_trunc) - Scalar{1};
          auto er = detail::exp_impl<Scalar>(1, x_frac, Scalar{0}, x_frac);
          return et * er + et + er;
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
  constexpr auto constexpr_sinh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_sinh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::sinh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
    {
      if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();
      else if (x == -std::numeric_limits<Scalar>::infinity()) return -constexpr_infinity<Scalar>();
    }

    if (x == Scalar{0}) return x;
    else
    {
      decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      using Xf = std::decay_t<decltype(xf)>;
      return detail::convert_to_output<Scalar>((constexpr_exp(xf) - constexpr_exp(-xf)) * Xf{0.5});
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(cosh)

  /**
   * \internal
   * \brief Hyperbolic cosine.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_cosh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_cosh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::cosh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
        return std::numeric_limits<Scalar>::infinity();

    decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
    using Xf = std::decay_t<decltype(xf)>;
    return detail::convert_to_output<Scalar>((constexpr_exp(xf) + constexpr_exp(-xf)) * Xf{0.5});
  }


  FUNCTIONISCONSTEXPRCALLABLE(tanh)

  /**
   * \internal
   * \brief Hyperbolic tangent.
   */
#ifdef __cpp_concepts
  constexpr auto constexpr_tanh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_tanh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::tanh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
    {
      if (x == std::numeric_limits<Scalar>::infinity()) return Scalar{1};
      else if (x == -std::numeric_limits<Scalar>::infinity()) return Scalar{-1};
    }

    if (x == Scalar{0}) return x;
    else
    {
      decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      auto em = constexpr_exp(-xf);
      auto e = constexpr_exp(std::forward<decltype(xf)>(xf));
      return detail::convert_to_output<Scalar>((e - em) / (e + em));
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(sin)

#ifdef __cpp_concepts
  constexpr auto constexpr_sin(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_sin(T&& arg)
#endif
  {
    using Arg = decltype(arg);

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

      auto theta {detail::scale_periodic_function(std::move(re))};
      auto sinre = detail::sin_cos_impl<Re>(4, theta, theta, theta * theta * theta / Re{-6.0});
      auto cosre = detail::sin_cos_impl<Re>(3, theta, Re{1}, Re{-0.5} * theta * theta);
      return make_complex_number<Scalar>(sinre * constexpr_cosh(im), cosre * constexpr_sinh(im));
    }
    else
    {
      decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
          return constexpr_NaN<Scalar>();

      auto theta {detail::scale_periodic_function(std::move(x))};
      return detail::sin_cos_impl<Scalar>(4, theta, theta, theta * theta * theta / Scalar{-6.0});
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(cos)

#ifdef __cpp_concepts
  constexpr auto constexpr_cos(scalar_type auto&& arg)
#else
  template <typename T>
  constexpr auto constexpr_cos(T&& arg, std::enable_if_t<scalar_type<T>, int> = 0)
#endif
  {
    using Arg = decltype(arg);

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

      auto theta {detail::scale_periodic_function(std::move(re))};
      auto sinre = detail::sin_cos_impl<Re>(4, theta, theta, theta * theta * theta / Re{-6.0});
      auto cosre = detail::sin_cos_impl<Re>(3, theta, Re{1}, Re{-0.5} * theta * theta);
      return make_complex_number<Scalar>(cosre * constexpr_cosh(im), -sinre * constexpr_sinh(im));
    }
    else
    {
      decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
          return constexpr_NaN<Scalar>();

      auto theta {detail::scale_periodic_function(std::move(x))};
      return detail::sin_cos_impl<Scalar>(3, theta, Scalar{1}, Scalar{-0.5} * theta * theta);
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(tan)

#ifdef __cpp_concepts
  constexpr auto constexpr_tan(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_tan(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::tan_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity())
        return constexpr_NaN<Scalar>();

    if (x == Scalar{0}) return x;
    else
    {
      decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      return detail::convert_to_output<Scalar>(constexpr_sin(xf) / constexpr_cos(xf));
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
  constexpr auto constexpr_log(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_log(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::log_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    if constexpr (complex_number<Scalar>)
    {
      auto re = detail::convert_to_floating(constexpr_real(arg));
      auto im = detail::convert_to_floating(constexpr_imag(arg));
      return make_complex_number<Scalar>(constexpr_log(constexpr_sqrt(re * re + im * im)), detail::atan2_impl(im, re));
    }
    else
    {
      decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if (x == Scalar{1}) return Scalar{+0.};

      if constexpr (std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity()) return x;

      if (x == 0) return -constexpr_infinity<Scalar>();
      else if (x < 0) return constexpr_NaN<Scalar>();
      auto [scaled, corr] = x >= Scalar{0x1p4} ? detail::log_scaling_gt(x) : detail::log_scaling_lt(x);
      return detail::log_impl(scaled) + corr;
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(asin)

#ifdef __cpp_concepts
  constexpr auto constexpr_asin(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_asin(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::asin_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if constexpr (complex_number<Scalar>)
    {
      if (arg != arg) return constexpr_NaN<Scalar>();

      decltype(auto) xf = detail::convert_to_floating(std::forward<Arg>(arg));
      using Xf = std::decay_t<decltype(xf)>;
      using R = std::decay_t<decltype(constexpr_real(xf))>;

      constexpr auto i = make_complex_number<Xf>(R{0}, R{1});
      return detail::convert_to_output<Scalar>(i * constexpr_log(constexpr_sqrt(Xf{1} - xf * xf) - i * xf));
    }
    else return detail::asin_impl(detail::convert_to_output<Scalar>(std::forward<Arg>(arg)));
  }


  FUNCTIONISCONSTEXPRCALLABLE(acos)

#ifdef __cpp_concepts
  constexpr auto constexpr_acos(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_acos(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::acos_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();
    else if (arg == std::decay_t<Arg>{1}) return static_cast<Scalar>(+0.);

    decltype(auto) xf = detail::convert_to_floating(std::forward<Arg>(arg));
    using Xf = std::decay_t<decltype(xf)>;
    using R = std::decay_t<decltype(constexpr_real(xf))>;

    auto s = constexpr_asin(xf);
    if (s != s) return constexpr_NaN<Scalar>();
    else return detail::convert_to_output<Scalar>(Xf{numbers::pi_v<R>/2} - std::move(s));
  }


  namespace detail
  {
    template <typename T>
    constexpr T atan_impl_general(const T& x)
    {
      if constexpr (complex_number<T>)
      {
        using R = std::decay_t<decltype(constexpr_real(x))>;
        constexpr auto i = make_complex_number<T>(R{0}, R{1});
        return T{-0.5} * i * constexpr_log((T{1} + x*i)/(T{1} - x*i));
      }
      else return atan_impl(x);
    }
  };


  FUNCTIONISCONSTEXPRCALLABLE(atan)

#ifdef __cpp_concepts
  constexpr auto constexpr_atan(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_atan(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::atan_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    if constexpr (complex_number<Scalar>)
    {
      return detail::convert_to_output<Scalar>(detail::atan_impl_general(detail::convert_to_floating(std::forward<Arg>(arg))));
    }
    else
    {
      decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

      if constexpr (std::numeric_limits<Scalar>::has_infinity)
      {
        if (x == std::numeric_limits<Scalar>::infinity()) return numbers::pi_v<Scalar> * Scalar{0.5};
        else if (x == -std::numeric_limits<Scalar>::infinity()) return numbers::pi_v<Scalar> * Scalar{-0.5};
      }

      if (x == Scalar{0}) return x;
      else return detail::atan_impl(x);
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE2(atan2)

#ifdef __cpp_concepts
  template <scalar_type Y, scalar_type X>
#else
  template <typename Y, typename X, std::enable_if_t<scalar_type<Y> and scalar_type<X>, int> = 0>
#endif
  constexpr auto constexpr_atan2(Y&& y_arg, X&& x_arg)
  {
    auto [is_callable, ret] = detail::atan2_is_constexpr_callable(y_arg, x_arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (y_arg != y_arg or x_arg != x_arg) return constexpr_NaN<Scalar>();

    if constexpr (complex_number<Y> or complex_number<X>)
    {
      decltype(auto) yf = detail::convert_to_floating(std::forward<Y>(y_arg));
      decltype(auto) xf = detail::convert_to_floating(std::forward<X>(x_arg));

      auto yp = constexpr_real(yf);
      auto xp = constexpr_real(xf);

      using Sf = std::decay_t<decltype(yf)>;
      using R = std::decay_t<decltype(yp)>;

      constexpr auto pi = numbers::pi_v<R>;

      if (xp >= R{0})
      {
        if (xf == Sf{0}) return Scalar{0};
        else return detail::convert_to_output<Scalar>(detail::atan_impl_general(yf / xf));
      }
      else if (yp >= R{0})
        return detail::convert_to_output<Scalar>(detail::atan_impl_general(yf / xf) + Sf{pi});
      else
        return detail::convert_to_output<Scalar>(detail::atan_impl_general(yf / xf) - Sf{pi});
    }
    else return detail::atan2_impl(detail::convert_to_output<Scalar>(std::forward<Y>(y_arg)),
      detail::convert_to_output<Scalar>(std::forward<X>(x_arg)));
  }


  FUNCTIONISCONSTEXPRCALLABLE(asinh)

#ifdef __cpp_concepts
  constexpr auto constexpr_asinh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_asinh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::asinh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
      if (x == std::numeric_limits<Scalar>::infinity() or x == -std::numeric_limits<Scalar>::infinity()) return x;

    if (x == Scalar{0}) return x;
    else
    {
      decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      using Xf = std::decay_t<decltype(xf)>;
      return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf * xf + Xf{1})));
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(acosh)

#ifdef __cpp_concepts
  constexpr auto constexpr_acosh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_acosh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::acosh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if (x == Scalar{1}) return static_cast<Scalar>(+0.);

    if constexpr (complex_number<Scalar>)
    {
      auto xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      using Xf = std::decay_t<decltype(xf)>;
      return detail::convert_to_output<Scalar>(constexpr_log(xf + constexpr_sqrt(xf + Xf{1}) * constexpr_sqrt(xf - Xf{1})));
    }
    else
    {
      if constexpr (std::numeric_limits<Scalar>::has_infinity)
        if (x == std::numeric_limits<Scalar>::infinity()) return constexpr_infinity<Scalar>();

      if (x < Scalar{1}) return constexpr_NaN<Scalar>();
      else return constexpr_log(x + constexpr_sqrt(x * x - Scalar{1}));
    }
  }


  FUNCTIONISCONSTEXPRCALLABLE(atanh)

#ifdef __cpp_concepts
  constexpr auto constexpr_atanh(scalar_type auto&& arg)
#else
  template <typename T, std::enable_if_t<scalar_type<T>, int> = 0>
  constexpr auto constexpr_atanh(T&& arg)
#endif
  {
    using Arg = decltype(arg);

    auto [is_callable, ret] = detail::atanh_is_constexpr_callable(arg);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

    if (arg != arg) return constexpr_NaN<Scalar>();

    decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<Arg>(arg));

    if constexpr (not complex_number<Scalar> and std::numeric_limits<Scalar>::has_infinity)
    {
      if (x < Scalar{-1} or x > Scalar{1}) return constexpr_NaN<Scalar>();
      else if (x == Scalar{1}) return constexpr_infinity<Scalar>();
      else if (x == Scalar{-1}) return -constexpr_infinity<Scalar>();
    }

    if (x == Scalar{0}) return x;
    else
    {
      decltype(auto) xf = detail::convert_to_floating(std::forward<decltype(x)>(x));
      using Xf = std::decay_t<decltype(xf)>;
      return detail::convert_to_output<Scalar>(constexpr_log((Xf{1} + xf)/(Xf{1} - xf)) * Xf{0.5});
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
  template<scalar_type T, scalar_type U>
#else
  template <typename T, typename U, std::enable_if_t<scalar_type<T> and scalar_type<U>, int> = 0>
#endif
  constexpr auto constexpr_pow(T&& arg, U&& exponent)
  {
    auto [is_callable, ret] = detail::pow_is_constexpr_callable(arg, exponent);
    if (is_callable) return ret;
    using Scalar = std::decay_t<decltype(ret)>;

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

      return detail::pow_integral(detail::convert_to_output<Scalar>(std::forward<T>(arg)), exponent);
    }
    else
    {
      if constexpr (complex_number<Scalar>)
      {
        decltype(auto) xf = detail::convert_to_floating(std::forward<T>(arg));
        decltype(auto) nf = detail::convert_to_floating(std::forward<U>(exponent));

        return detail::convert_to_output<Scalar>(constexpr_exp(constexpr_log(xf) * nf));
      }
      else
      {
        decltype(auto) x = detail::convert_to_output<Scalar>(std::forward<T>(arg));
        decltype(auto) n = detail::convert_to_output<Scalar>(std::forward<U>(exponent));

        if constexpr (std::numeric_limits<Scalar>::has_infinity)
        {
          if (x == -std::numeric_limits<Scalar>::infinity() or x == +std::numeric_limits<Scalar>::infinity())
          {
            if (n < Scalar{0}) return Scalar{+0.}; else return +constexpr_infinity<Scalar>();
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
          if (n == Scalar{1}) return x;
          else return constexpr_exp(constexpr_log(x) * n);
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

} // namespace OpenKalman::internal

#undef FUNCTIONISCONSTEXPRCALLABLE2
#undef FUNCTIONEXISTSTEST2
#undef IFFUNCTIONISINVOCABLEWITHARG2
#undef FUNCTIONISCONSTEXPRCALLABLE
#undef FUNCTIONEXISTSTEST
#undef IFFUNCTIONISINVOCABLEWITHARG
#undef NOTCONSTANTEVALUATED

#endif //OPENKALMAN_MATH_CONSTEXPR_HPP
