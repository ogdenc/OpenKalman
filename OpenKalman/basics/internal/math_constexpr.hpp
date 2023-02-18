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

#include<cstdint>
#include<limits>


namespace OpenKalman::internal
{

  namespace detail
  {
    // Convert integral to floating or complex integral to complex floating and then call a function with the result
    template<typename F, typename T, typename...Ts>
    constexpr decltype(auto) convert_floating(const F& f, T&& x, Ts&&...xs)
    {
      if constexpr ((complex_number<T> or ... or complex_number<Ts>))
      {
        using R = std::decay_t<decltype(make_complex_number(real_part(x), imaginary_part(x)))>;
        if constexpr ((std::is_same_v<std::decay_t<T>, R> and ... and std::is_same_v<std::decay_t<Ts>, R>))
          return f(std::forward<T>(x), std::forward<Ts>(xs)...);
        else return f(make_complex_number(real_part(std::forward<T>(x)), imaginary_part(x)),
          make_complex_number(real_part(std::forward<Ts>(xs)), imaginary_part(xs))...);
      }
      else return f(real_part(std::forward<T>(x)), real_part(std::forward<Ts>(xs))...);
    }
  } // detail


  /**
   * \internal
   * \brief A constexpr square root function.
   * \details Uses the Newton-Raphson method
   * \tparam Scalar The scalar type.
   * \param x The operand.
   * \return The square root of x.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>
  constexpr auto constexpr_sqrt(const T& x)
#else
  template <typename T>
  constexpr auto constexpr_sqrt(const T& x, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>, int> = 0)
#endif
  {
    /** // Code for purely integral version:
    T lo = 0 , hi = x / 2 + 1;
    while (lo != hi) { const T mid = (lo + hi + 1) / 2; if (x / mid < mid) hi = mid - 1; else lo = mid; }
    return lo;*/
    auto f = [](const auto& fx) {
      using Fx = std::decay_t<decltype(fx)>;
      if constexpr (complex_number<Fx>)
      {
        // Find the principal square root
        auto a {real_part(fx)};
        auto b {imaginary_part(fx)};
        auto afx = constexpr_sqrt(a * a + b * b);
        auto sqp = constexpr_sqrt((afx + a) / 2);
        auto sqm = constexpr_sqrt((afx - a) / 2);
        return make_complex_number(sqp, b >= 0 ? sqm : -sqm);
      }
      else
      {
        if (fx <= Fx{0})
        {
          if (fx == Fx{0}) return Fx{0};
          else if (std::numeric_limits<Fx>::has_quiet_NaN) return std::numeric_limits<Fx>::quiet_NaN();
          else if (std::numeric_limits<Fx>::has_signaling_NaN) return std::numeric_limits<Fx>::signaling_NaN();
          else return 1 / (fx - fx); // error
        }
        else
        {
          Fx next = Fx{0.5} * fx;
          Fx previous = Fx{0};
          while (next != previous)
          {
            previous = next;
            next = Fx{0.5} * (previous + fx / previous);
          }
          return next;
        }
      }
    };
    return detail::convert_floating(f, x);
  }


  namespace detail
  {
    // Taylor series expansion
    template <typename T>
    constexpr T sin_cos_impl(int i, T x, T sum, T term)
    {
      auto new_sum = sum + term;
      //if (are_within_tolerance(sum, new_sum)) return new_sum;
      if (sum == new_sum) return new_sum;
      else return sin_cos_impl(i + 2, x, new_sum, -term / static_cast<T>(i * (i + 1)) * x * x);
    }


    template <typename T>
    constexpr T scale_periodic_function(T theta)
    {
      constexpr auto pi2 = numbers::pi_v<T> * 2;
      constexpr auto max = static_cast<T>(std::numeric_limits<std::intmax_t>::max());
      constexpr auto lowest = static_cast<T>(std::numeric_limits<std::intmax_t>::lowest());
      if (theta > -pi2 and theta < pi2) return theta;
      else if (theta / pi2 >= lowest and theta / pi2 <= max) return theta - static_cast<std::intmax_t>(theta / pi2) * pi2;
      else if (theta > 0)
      {
        T corr = pi2;
        while ((theta - corr) / pi2 > max) corr *= 2;
        return scale_periodic_function(theta - corr);
      }
      else
      {
        T corr = pi2;
        while ((theta + corr) / pi2 < lowest) corr *= 2;
        return scale_periodic_function(theta + corr);
      }
    }
  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>>
  constexpr auto constexpr_sin(T&& x)
#else
  template <typename T>
  constexpr auto constexpr_sin(T&& x, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Fx = std::decay_t<decltype(x)>;
      Fx theta = detail::scale_periodic_function(std::forward<decltype(x)>(x));
      if (theta == Fx{0}) return Fx{0};
      else return detail::sin_cos_impl<Fx>(4, theta, theta, -theta * theta * theta / 6.0);
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>>
  constexpr auto constexpr_cos(T&& x)
#else
  template <typename T>
  constexpr auto constexpr_cos(T&& x, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Fx = std::decay_t<decltype(x)>;
      Fx theta = detail::scale_periodic_function(std::forward<decltype(x)>(x));
      if (theta == Fx{0}) return Fx{1.0};
      else return detail::sin_cos_impl<Fx>(3, theta, 1.0, -theta * theta / 2);
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


  namespace detail
  {
    // Maclaurin series expansion
    template <typename T>
    constexpr auto exp_impl(int i, const T& x, const T& sum, const T& term)
    {
      auto new_sum = sum + term;
      if (sum == new_sum) return new_sum;
      else return exp_impl(i + 1, x, new_sum, term * x / static_cast<T>(i + 1));
    }
  } // detail


  /**
   * \internal
   * \brief Exponential function, taking an integral argument.
   * \tparam Scalar The result type
   */
#ifdef __cpp_concepts
  template <floating_scalar_type Scalar = double, std::integral T>
#else
  template <typename Scalar = double, typename T, std::enable_if_t<
    floating_scalar_type<Scalar> and std::is_integral_v<std::decay_t<T>>, int> = 0>
#endif
  constexpr std::decay_t<Scalar> constexpr_exp(T x)
  {
    constexpr auto e = numbers::e_v<Scalar>;
    if (x == 0) return Scalar{1};
    else if (x == 1) return e;
    else if (x < 0) return Scalar{1} / constexpr_exp<Scalar>(-x);
    else if (x % 2 == 1) return e * constexpr_exp<Scalar>(x - 1); //< odd
    else return constexpr_exp<Scalar>(x / 2) * constexpr_exp<Scalar>(x / 2); //< even
  }


  /**
   * \internal
   * \brief Exponential function.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_floating_point_v<std::decay_t<T>> or complex_number<T>
#else
  template <typename T, std::enable_if_t<std::is_floating_point_v<std::decay_t<T>> or complex_number<T>, int> = 0>
#endif
  constexpr auto constexpr_exp(T&& x)
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      if constexpr (complex_number<Scalar>)
      {
        auto ea = constexpr_exp(real_part(x));
        auto b = imaginary_part(x);
        return make_complex_number(ea * constexpr_cos(b), ea * constexpr_sin(b));
      }
      else
      {
        if (x >= 0 and x < 1) return detail::exp_impl<Scalar>(1, x, 1, x);
        else
        {
          int x_trunc = static_cast<int>(x) - (x < Scalar{0} ? 1 : 0);
          Scalar x_frac = x - static_cast<Scalar>(x_trunc);
          return constexpr_exp<Scalar, int>(x_trunc) * detail::exp_impl<Scalar>(1, x_frac, 1, x_frac);
        }
      }
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


  /**
   * \internal
   * \brief Exponential function minus 1, taking an integral argument.
   * \tparam Scalar The result type
   */
#ifdef __cpp_concepts
  template <floating_scalar_type Scalar = double, std::integral T>
#else
  template <typename Scalar = double, typename T, std::enable_if_t<
    floating_scalar_type<Scalar> and std::is_integral_v<std::decay_t<T>>, int> = 0>
#endif
  constexpr std::decay_t<Scalar> constexpr_expm1(T x)
  {
    return constexpr_exp<Scalar>(x) - Scalar{1};
  }


  /**
   * \internal
   * \brief Exponential function minus 1.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_floating_point_v<std::decay_t<T>> or complex_number<T>
  constexpr auto constexpr_expm1(T&& x)
#else
  template <typename T>
  constexpr auto constexpr_expm1(T&& x, std::enable_if_t<std::is_floating_point_v<std::decay_t<T>> or complex_number<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      if constexpr (complex_number<Scalar>)
      {
        auto ea = constexpr_expm1(real_part(x));
        auto b = imaginary_part(x);
        auto sinb2 = constexpr_sin(b/2);
        auto sinb = constexpr_sin(b);
        return make_complex_number(ea * constexpr_cos(b) - 2*sinb2*sinb2, ea * sinb + sinb);
      }
      else
      {
        if (x >= 0 and x < 1) return detail::exp_impl<Scalar>(1, x, 0, x);
        else
        {
          int x_trunc = static_cast<int>(x) - (x < Scalar{0} ? 1 : 0);
          Scalar x_frac = x - static_cast<Scalar>(x_trunc);
          auto et = constexpr_expm1<Scalar, int>(x_trunc);
          auto er = detail::exp_impl<Scalar>(1, x_frac, 0, x_frac);
          return et * er + et + er;
        }
      }
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


  /**
   * \internal
   * \brief Hyperbolic sine.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>
#else
  template <typename T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>, int> = 0>
#endif
  constexpr auto constexpr_sinh(T&& x)
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      return (internal::constexpr_exp(x) - internal::constexpr_exp(-x)) / Scalar{2};
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


  /**
   * \internal
   * \brief Hyperbolic cosine.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>
#else
  template <typename T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>, int> = 0>
#endif
  constexpr auto constexpr_cosh(T&& x)
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      return (internal::constexpr_exp(x) + internal::constexpr_exp(-x)) / Scalar{2};
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


  /**
   * \internal
   * \brief Hyperbolic tangent.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>
#else
  template <typename T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>, int> = 0>
#endif
  constexpr auto constexpr_tanh(T&& x)
  {
    auto e = internal::constexpr_exp(x);
    auto em = internal::constexpr_exp(-x);
    return (e - em) / (e + em);
  }


#ifdef __cpp_concepts
  template <complex_number T>
  constexpr auto constexpr_sin(T&& x)
#else
  template <typename T>
  constexpr auto constexpr_sin(T&& x, std::enable_if_t<complex_number<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      auto re = real_part(x);
      auto im = imaginary_part(x);
      return make_complex_number(constexpr_sin(re) * constexpr_cosh(im), constexpr_cos(re) * constexpr_sinh(im));
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <complex_number T> requires (not std::is_arithmetic_v<std::decay_t<T>>)
  constexpr auto constexpr_cos(T&& x)
#else
  template <typename T>
  constexpr auto constexpr_cos(T&& x, std::enable_if_t<complex_number<T> and not std::is_arithmetic_v<std::decay_t<T>>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      auto re = real_part(x);
      auto im = imaginary_part(x);
      return make_complex_number(constexpr_cos(re) * constexpr_cosh(im), -constexpr_sin(re) * constexpr_sinh(im));
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <typename X> requires std::is_arithmetic_v<X> or complex_number<X>
  constexpr auto constexpr_tan(X x)
#else
  template <typename X>
  constexpr auto constexpr_tan(X x, std::enable_if_t<std::is_arithmetic_v<X> or complex_number<X>, int> = 0)
#endif
  {
    return constexpr_sin(x) / constexpr_cos(x);
  }


  namespace detail
  {
    template <typename T>
    constexpr T asin_impl(int n, T x, T sum, T term)
    {
      auto new_sum = sum + term / static_cast<T>(n);
      if (sum == new_sum) return new_sum;
      else return asin_impl(n + 2, x, new_sum, term * static_cast<T>(n)/static_cast<T>(n+1) * x * x);
    }
  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<T>
  constexpr auto constexpr_asin(T x)
#else
  template <typename T>
  constexpr auto constexpr_asin(T x, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      constexpr auto pi = numbers::pi_v<Scalar>;
      if (-1 < x and x < 1) return detail::asin_impl<Scalar>(3, x, x, x * x * x / Scalar{2});
      else if (x == 1) return +pi / 2;
      else if (x == -1) return -pi / 2;
      else return 1 / (x - x); // error
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<T>
  constexpr auto constexpr_acos(T x)
#else
  template <typename T>
  constexpr auto constexpr_acos(T x, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      constexpr auto pi = numbers::pi_v<Scalar>;
      if (x == -1) return pi;
      else if (x == 1) return Scalar{0};
      else if (x > -1 and x < 1) return pi/2 - constexpr_asin<Scalar>(x);
      else return 1 / (x - x); // error
    };
    return detail::convert_floating(f, std::forward<T>(x));

  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<T>
  constexpr auto constexpr_atan(T x)
#else
  template <typename T>
  constexpr auto constexpr_atan(T x, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      if (x == Scalar{0}) return Scalar{0};
      else return constexpr_asin(x / constexpr_sqrt(Scalar{1} + x * x));
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<T>
  constexpr auto constexpr_atan2(T y, T x)
#else
  template <typename T>
  constexpr auto constexpr_atan2(T y, T x, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0)
#endif
  {
    auto f = [](auto&& y, auto&& x) {
      using Scalar = std::conditional_t<std::is_integral_v<T>, double, T>;
      constexpr auto pi = numbers::pi_v<Scalar>;

      if constexpr (std::numeric_limits<Scalar>::has_signaling_NaN)
      {
        constexpr auto nan = std::numeric_limits<Scalar>::signaling_NaN();
        if (x == nan or y == nan) return nan;
      }

      if constexpr (std::numeric_limits<Scalar>::has_quiet_NaN)
      {
        constexpr auto nan = std::numeric_limits<Scalar>::quiet_NaN();
        if (x == nan or y == nan) return nan;
      }

      if constexpr (std::numeric_limits<Scalar>::has_infinity)
      {
        constexpr auto inf = std::numeric_limits<Scalar>::infinity();
        if (y == +inf)
        {
          if (x == +inf) return pi / 4;
          else if (x == -inf) return 3 * pi / 4;
          else return pi / 2;
        }
        else if (y == -inf)
        {
          if (x == +inf) return -pi / 4;
          else if (x == -inf) return -3 * pi / 4;
          else return -pi / 2;
        }
        else if (x == +inf) // should copy sign of y onto 0.
        {
          if (y >= 0) return Scalar{0};
          else return -Scalar{0};
        }
        else if (x == -inf) // should copy sign of y onto pi.
        {
          if (y >= 0) return pi;
          else return -pi;
        }
      }

      if (x > 0)
      {
        return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x));
      }
      else if (x < 0)
      {
        if (y > 0) return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x)) + pi;
        else if (y < 0) return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x)) - pi;
        else return pi; // sign should depend on y
      }
      else if (x == 0)
      {
        if (y > 0) return pi / 2;
        else if (y < 0) return -pi / 2;
        else return Scalar{0};  // sign should depend on y
      }
      else return 1 / (y - y); // error
    };

    return detail::convert_floating(f, y, x);
  }


  namespace detail
  {
    template <typename T>
    constexpr T arithmetic_geometric_mean(const T& a, const T& g)
    {
      T nexta = (a + g) / T{2};
      T nextg = constexpr_sqrt(a * g);
      if constexpr (complex_number<T>) {if (are_within_tolerance(nexta, nextg)) return nexta;}
      else if (nexta <= nextg) return nexta;
      return arithmetic_geometric_mean(nexta, nextg);
    }


    template <typename T>
    constexpr T log_halley_impl(const T& x, const T& y0, int cmp = 0)
    {
      auto expy0 = constexpr_exp(y0);
      auto y1 = y0 + T{2} * (x - expy0) / (x + expy0);
      if constexpr (complex_number<T>)
      {
        if (are_within_tolerance(y1, y0)) return y1;
        else return log_halley_impl(x, y1);
      }
      else
      {
        // Detect when there is a change in direction.
        if (y1 == y0 or (cmp < 0 and y1 > y0) or (cmp > 0 and y1 < y0)) return y1;
        else return log_halley_impl(x, y1, y1 > y0 ? +1 : y1 < y0 ? -1 : 0);
      }
    }


    template <typename T>
    constexpr T log_impl(const T& x)
    {
      using R = std::decay_t<decltype(real_part(std::declval<T>()))>;

      // Get a close estimation using the agm (arithmetic-geometric mean) function.
      constexpr auto pi = numbers::pi_v<R>;
      constexpr auto ln2 = numbers::ln2_v<R>;
      constexpr T m = 14 + 2;
      constexpr T m2 = 0x1p-14;
      auto approx = pi/(T{2} * detail::arithmetic_geometric_mean<T>(1, m2/x)) - m*ln2;

      // Finish off with Halley's method for the last few digits.
      return detail::log_halley_impl(x, approx);
    }


    template <typename T>
    constexpr std::tuple<T, T> log_scaling_gt(T x, T corr = 0)
    {
      if (x < T{0x1p+4}) return {x, corr};
      else if (x < T{0x1p+16}) return log_scaling_gt<T>(x * T{0x1p-4}, corr + T{4} * numbers::ln2_v<T>);
      else if (x < T{0x1p+64}) return log_scaling_gt<T>(x * T{0x1p-16}, corr + T{16} * numbers::ln2_v<T>);
      else return log_scaling_gt<T>(x * T{0x1p-64}, corr + T{64} * numbers::ln2_v<T>);
    }


    template <typename T>
    constexpr std::tuple<T, T> log_scaling_lt(T x, T corr = 0)
    {
      if (x > T{0x1p-4}) return {x, corr};
      else if (x > T{0x1p-16}) return log_scaling_lt<T>(x * T{0x1p+4}, corr - T{4} * numbers::ln2_v<T>);
      else if (x > T{0x1p-64}) return log_scaling_lt<T>(x * T{0x1p+16}, corr - T{16} * numbers::ln2_v<T>);
      else return log_scaling_lt<T>(x * T{0x1p+64}, corr - T{64} * numbers::ln2_v<T>);
    }
  } // namespace detail


  /**
   * \internal
   * \brief Natural logarithm function.
   */
#ifdef __cpp_concepts
  template <typename T> requires std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>
  constexpr auto constexpr_log(const T& x)
#else
  template <typename T>
  constexpr auto constexpr_log(const T& x, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>, int> = 0)
#endif
  {
    auto f = [](const auto& fx) {
      using Fx = std::decay_t<decltype(fx)>;
      if (fx == Fx{1}) return Fx{0};

      if constexpr (complex_number<Fx>)
      {
        auto re = real_part(fx);
        auto im = imaginary_part(fx);
        return make_complex_number(constexpr_log(constexpr_sqrt(re * re + im * im)), constexpr_atan2(im, re));
      }
      else
      {
        auto [scaled, corr] = fx >= Fx{0x1p4} ? detail::log_scaling_gt(fx) : detail::log_scaling_lt(fx);
        return detail::log_impl(scaled) + corr;
      }
    };
    return detail::convert_floating(f, x);
  }


#ifdef __cpp_concepts
  template <complex_number T>
  constexpr auto constexpr_asin(T x)
#else
  template <typename T>
  constexpr auto constexpr_asin(T x, std::enable_if_t<complex_number<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      using R = std::decay_t<decltype(real_part(x))>;
      constexpr auto i = make_complex_number(R{0}, R{-1});
      return i * constexpr_log(constexpr_sqrt(Scalar{1} - x * x) - i * x);
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <complex_number T>
  constexpr auto constexpr_acos(T x)
#else
  template <typename T>
  constexpr auto constexpr_acos(T x, std::enable_if_t<complex_number<T>, int> = 0)
#endif
  {
    auto f = [](auto&& x) {
      using R = std::decay_t<decltype(real_part(x))>;
      constexpr auto pi = numbers::pi_v<R>;
      return pi/2 - constexpr_asin(x);
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <complex_number T>
  constexpr auto constexpr_atan(T x)
#else
  template <typename T, std::enable_if_t<complex_number<T>, int> = 0>
  constexpr auto constexpr_atan(T x)
#endif
  {
    auto f = [](auto&& x) {
      using Scalar = std::decay_t<decltype(x)>;
      return constexpr_asin(x / constexpr_sqrt(Scalar{1} + x * x));
    };
    return detail::convert_floating(f, std::forward<T>(x));
  }


#ifdef __cpp_concepts
  template <complex_number T>
  constexpr auto constexpr_atan2(T y, T x)
#else
  template <typename T>
  constexpr auto constexpr_atan2(T y, T x, std::enable_if_t<complex_number<T>, int> = 0)
#endif
  {
    auto f = [](auto&& y, auto&& x) {
      auto yp = real_part(y);
      auto xp = real_part(x);

      using RealScalar = std::decay_t<decltype(xp)>;
      using Scalar = std::complex<RealScalar>;
      constexpr auto pi = numbers::pi_v<RealScalar>;

      if (xp >= 0)
      {
        if (xp == 0 and imaginary_part(x) == 0) return make_complex_number(RealScalar{0}, RealScalar{0});
        else return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x));
      }
      else if (yp >= 0) return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x)) + pi;
      else return constexpr_atan(static_cast<Scalar>(y) / static_cast<Scalar>(x)) - pi;
    };
    return detail::convert_floating(f, y, x);
  }


  /**
   * \internal
   * \brief A constexpr power function.
   * \param x The operand
   * \param n The power
   * \return x to the power of n.
   */
#ifdef __cpp_concepts
  template <typename T, typename U> requires (std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and
    (std::is_floating_point_v<U> or complex_number<U>)
  constexpr auto constexpr_pow(T x, U n)
#else
  template <typename T, typename U>
  constexpr auto constexpr_pow(T x, U n, std::enable_if_t<(std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and
    (std::is_floating_point_v<U> or complex_number<U>), int> = 0)
#endif
  {
    auto f = [](auto&& x, auto&& n) {
      using Scalar = std::decay_t<decltype(x)>;
      return constexpr_exp(constexpr_log<Scalar>(x) * static_cast<Scalar>(n));
    };
    return detail::convert_floating(f, x, n);
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template <typename T, std::integral U>
    requires (std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and std::is_unsigned_v<U>
  constexpr T constexpr_pow(T x, U n)
#else
  template <typename T, typename U>
  constexpr T constexpr_pow(T x, U n, std::enable_if_t<(std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and
    std::is_integral_v<U> and std::is_unsigned_v<U>, int> = 0)
#endif
  {
    if (n == 0) return 1;
    else if (n == 1) return x;
    else if (n % 2 == 1) return x * constexpr_pow(x, n - 1);
    else return constexpr_pow(x, n / 2) * constexpr_pow(x, n / 2);
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template <typename T, std::integral U>
    requires (std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and (not std::is_unsigned_v<U>)
  constexpr auto constexpr_pow(T x, U n)
#else
  template <typename T, typename U>
  constexpr auto constexpr_pow(T x, U n, std::enable_if_t<
    (std::is_arithmetic_v<std::decay_t<T>> or complex_number<T>) and
    (std::is_integral_v<U> and not std::is_unsigned_v<U>), int> = 0)
#endif
  {
    using R = std::decay_t<decltype(detail::convert_floating(std::negate<>{}, x))>;
    if (n == 0) return R {1};
    else if (n == 1) return static_cast<R>(x);
    else if (n < 0) return R{1} / constexpr_pow<T, U>(x, -n);
    else if (n % 2 == 1) return static_cast<R>(x) * constexpr_pow<T, U>(x, n - 1);
    else return constexpr_pow<T, U>(x, n / 2) * constexpr_pow<T, U>(x, n / 2);
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_MATH_CONSTEXPR_HPP
