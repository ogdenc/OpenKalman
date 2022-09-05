/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts and traits for scalar types.
 */

#ifndef OPENKALMAN_SCALAR_TYPES_HPP
#define OPENKALMAN_SCALAR_TYPES_HPP

#include <complex>

namespace OpenKalman
{

  // -------------------- //
  //    complex_number    //
  // -------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_complex_number : std::false_type {};


    template<typename T>
    struct is_complex_number<std::complex<T>> : std::true_type {};
  }


  /**
   * \brief T is a std::complex.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept complex_number = detail::is_complex_number<std::decay_t<T>>::value;
#else
  constexpr bool complex_number = detail::is_complex_number<std::decay_t<T>>::value;
#endif


  // --------------------------------------------- //
  //    ScalarTraits for arithmetic and complex    //
  // --------------------------------------------- //

  namespace interface
  {
#ifdef __cpp_concepts
    template<typename T> requires std::is_arithmetic_v<std::decay_t<T>>
    struct ScalarTraits<T>
#else
    template<typename T>
    struct ScalarTraits<T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>>>
#endif
    {
    private:

      using Real = std::conditional_t<std::is_floating_point_v<std::decay_t<T>>, std::decay_t<T>, double>;

    public:

      template<typename Arg>
      static constexpr auto real_projection(Arg&& arg)
      {
        if constexpr (std::is_floating_point_v<std::decay_t<Arg>>) return std::forward<Arg>(arg);
        else return double(std::forward<Arg>(arg));
      }

      template<typename Arg, typename RealProj>
      static constexpr auto inverse_real_projection(Arg&&, RealProj p) { return p; }

      static auto sin(std::decay_t<T> t) { return std::sin(t); }
      static auto cos(std::decay_t<T> t) { return std::cos(t); }

      static auto sqrt(std::decay_t<T> t) { return std::sqrt(t); }

      static auto asin2(std::decay_t<T> y, std::decay_t<T> r) {
        // This is so that a zero-radius or faulty spherical coordinate has horizontal inclination:
        if (r == 0 or r < y or y < -r) return std::decay_t<decltype(std::asin(y/r))>(0);
        else return std::asin(y/r);
      }

      static auto atan2(std::decay_t<T> y, std::decay_t<T> x) {
        if constexpr (std::numeric_limits<std::decay_t<T>>::is_iec559) return std::atan2(y, x);
        else
        {
          using R = std::decay_t<decltype(std::atan2(y, x))>;
          if (y == 0)
            return std::copysign(std::signbit(x) ? R(0) : numbers::pi_v<R>, y);
          else if (not std::isfinite(y))
            return std::copysign(numbers::pi_v<R> * (std::isfinite(x) ? 0.5 : std::signbit(x) ? 0.25 : 0.75), y);
          else if (not std::isfinite(x)) {
            if (std::signbit(x)) return std::copysign(numbers::pi_v<R>, y);
            else return std::copysign(R(0), y);
          }
          else if (x == 0)
            return std::copysign(numbers::pi_v<R>/2, y);
          else
            return std::atan2(y, x);
        }
      }
    };


#ifdef __cpp_concepts
    template<complex_number T>
    struct ScalarTraits<T>
#else
    template<typename T>
    struct ScalarTraits<T, std::enable_if_t<complex_number<std::decay_t<T>>>>
#endif
    {
    private:

      using Real = std::decay_t<decltype(std::real(std::declval<T>()))>;
      static constexpr auto pi = numbers::pi_v<Real>;

    public:

      template<typename Arg>
      static constexpr auto real_projection(Arg&& arg)
      {
        return std::real(std::forward<Arg>(arg));
      }

      template<typename Arg, typename RealProj>
      static constexpr auto inverse_real_projection(Arg&& arg, RealProj p)
      {
        return std::complex {p, std::imag(std::forward<Arg>(arg))};
      }

      static auto sin(std::decay_t<T> t) { return std::sin(t); }
      static auto cos(std::decay_t<T> t) { return std::cos(t); }

      static auto sqrt(std::decay_t<T> t) { return std::sqrt(t);
      }

      static auto asin2(std::decay_t<T> y, std::decay_t<T> r) {
        if (r == Real(0)) return std::decay_t<T>(Real(0));
        else return std::asin(y/r);
      }

      static auto atan2(std::decay_t<T> y, std::decay_t<T> x) {
        if (y == Real(0)) {
          return std::decay_t<T> {std::copysign(std::signbit(x.real()) ? 0 : pi, y.real())};
        }
        else if (not std::isfinite(y.real())) {
          Real k = pi * (std::isfinite(x.real()) ? 0.5 : std::signbit(x.real()) ? 0.25 : 0.75);
          return std::decay_t<T> {std::copysign(k, y.real())};
        }
        else if (not std::isfinite(x.real())) {
          if (std::signbit(x.real())) return std::decay_t<T> {std::copysign(pi, y.real())};
          else return std::decay_t<T> {std::copysign(Real(0), y.real())};
        }
        else if (x.real() > 0) return std::atan(y/x);
        else if (x.real() < 0) return std::atan(y/x) + std::copysign(pi, y.real());
        else return std::decay_t<T> {std::copysign(pi/2, y.real())}; // x.real() == 0
      }
    };

  } // namespace interface


  // -------------------------- //
  //    floating_scalar_type    //
  // -------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_floating_scalar_type : std::false_type {};

    template<typename T>
    struct is_floating_scalar_type<T, std::enable_if_t<
      std::is_default_constructible_v<T> and
      std::is_floating_point_v<std::decay_t<decltype(interface::ScalarTraits<T>::real_projection(std::declval<T>()))>> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::inverse_real_projection(std::declval<T>(),
        interface::ScalarTraits<T>::real_projection(std::declval<T>()))), const T&> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::sin(std::declval<T>())), const T&> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::cos(std::declval<T>())), const T&> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::sqrt(std::declval<T>())), const T&> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::asin2(std::declval<T>(), std::declval<T>())), const T&> and
      std::is_convertible_v<decltype(interface::ScalarTraits<T>::atan2(std::declval<T>(), std::declval<T>())), const T&> and
      std::is_convertible_v<decltype(std::declval<T>() + std::declval<T>()), const T&> and
      std::is_convertible_v<decltype(std::declval<T>() - std::declval<T>()), const T&> and
      std::is_convertible_v<decltype(std::declval<T>() * std::declval<T>()), const T&> and
      std::is_convertible_v<decltype(std::declval<T>() / std::declval<T>()), const T&> and
      std::is_convertible_v<decltype(std::declval<T>() == std::declval<T>()), const T&>
      >>: std::true_type {};
  }
#endif


  /**
   * \brief T is a scalar angle type.
   * \details T must be a floating-point scalar type which may include std::floating_point, std::complex,
   * or a custom-defined scalar type in which certain traits in \ref interface::ScalarTraits are defined and
   * typical math operations (+, -, *, /, and ==) are also defined.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept floating_scalar_type = std::floating_point<std::decay_t<T>> or complex_number<T> or
    requires(T t1, T t2) {
      requires std::default_initializable<std::decay_t<T>>;
      requires std::floating_point<std::decay_t<decltype(interface::ScalarTraits<T>::real_projection(t1))>>;
      {interface::ScalarTraits<T>::inverse_real_projection(t1,
        interface::ScalarTraits<T>::real_projection(t2))} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<T>::sin(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<T>::cos(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<T>::sqrt(t1)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<T>::asin2(t1, t2)} -> std::convertible_to<const std::decay_t<T>&>;
      {interface::ScalarTraits<T>::atan2(t1, t2)} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 + t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 - t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 * t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 / t2} -> std::convertible_to<const std::decay_t<T>&>;
      {t1 == t2} -> std::convertible_to<const bool>;
    };
#else
  constexpr bool floating_scalar_type =
    std::is_floating_point_v<std::decay_t<T>> or complex_number<T> or detail::is_floating_scalar_type<std::decay_t<T>>::value;
#endif


  // ----------------- //
  //    scalar_type    //
  // ----------------- //

  /**
   * \brief T is a scalar type (i.e., a \ref floating_scalar_type or std::is_arithmetic.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept scalar_type =
#else
  constexpr bool scalar_type =
#endif
    floating_scalar_type<T> or std::is_arithmetic_v<std::decay_t<T>>;


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_TYPES_HPP
