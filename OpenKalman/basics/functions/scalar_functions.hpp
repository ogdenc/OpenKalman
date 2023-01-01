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
 * \brief Overloaded general functions relating to scalar types.
 */

#ifndef OPENKALMAN_SCALAR_FUNCTIONS_HPP
#define OPENKALMAN_SCALAR_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;


  /**
   * \brief Determine whether two numbers are within a rounding tolerance
   * \tparam Arg1 The first argument
   * \tparam Arg2 The second argument
   * \tparam epsilon_factor A factor to be multiplied by the epsilon
   * \return true if within the rounding tolerance, otherwise false
   */
  template<unsigned int epsilon_factor = 2, typename Arg1, typename Arg2>
  constexpr bool are_within_tolerance(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (complex_number<Arg1> or complex_number<Arg2>)
    {
      return are_within_tolerance<epsilon_factor>(std::real(arg1), std::real(arg2)) and
        are_within_tolerance<epsilon_factor>(std::imag(arg1), std::imag(arg2));
    }
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      constexpr auto ep = epsilon_factor * std::numeric_limits<Diff>::epsilon();
      return -static_cast<Diff>(ep) <= diff and diff <= static_cast<Diff>(ep);
    }

  }


  namespace internal
  {

    /**
     * \internal
     * \brief A constexpr square root function.
     * \tparam Scalar The scalar type.
     * \param x The operand.
     * \return The square root of x.
     */
    template<typename Scalar>
# ifdef __cpp_consteval
    consteval
# else
    constexpr
# endif
    Scalar constexpr_sqrt(Scalar x)
    {
      if constexpr(std::is_integral_v<Scalar>)
      {
        Scalar lo = 0;
        Scalar hi = x / 2 + 1;
        while (lo != hi)
        {
          const Scalar mid = (lo + hi + 1) / 2;
          if (x / mid < mid) hi = mid - 1;
          else lo = mid;
        }
        return lo;
      }
      else
      {
        Scalar cur = 0.5 * x;
        Scalar old = 0.0;
        while (cur != old)
        {
          old = cur;
          cur = 0.5 * (old + x / old);
        }
        return cur;
      }
    }


    /**
     * \internal
     * \brief A constexpr power function.
     * \tparam Scalar The scalar type.
     * \param a The operand
     * \param n The power
     * \return a to the power of n.
     */
    template<typename Scalar>
//# ifdef __cpp_consteval
//    consteval
//# else
    constexpr
//# endif
    Scalar constexpr_pow(Scalar a, std::size_t n)
    {
      return n == 0 ? 1 : constexpr_pow(a, n / 2) * constexpr_pow(a, n / 2) * (n % 2 == 0 ?  1 : a);
    }


  } // namespace internal


  /**
   * \brief Project to a real number of a \ref std::floating_point type that depends on the argument.
   * \tparam Arg a \ref scalar_type
   * \details For example, if the argument is a complex number, the function will convert to the type of its real part.
   */
#ifdef __cpp_concepts
  constexpr decltype(auto)
  real_projection(scalar_type auto&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_type<Arg>, int> = 0>
  constexpr decltype(auto) real_projection(Arg&& arg)
#endif
  {
    if constexpr (complex_number<decltype(arg)>)
      return interface::ScalarTraits<std::decay_t<decltype(arg)>>::real_projection(std::forward<decltype(arg)>(arg));
    else
      return std::forward<decltype(arg)>(arg);
  }


  /**
   * \overload
   * \brief Project to a real number of type Scalar.
   * \details This is used for wrapping the real part of angles or other modular scalar values. If the argument is
   * already a real number (as opposed, for example, to a complex number), this will be an identity function
   * (if already \ref std::floating_point) or a numerical conversion from a \ref std::integral or custom scalar type to
   * \ref std::floating_point.
   * \tparam Scalar a std::floating_point type to convert to.
   * \tparam Arg a \ref scalar_type
   * \return A number of type Scalar. This will be the real part of a complex number, or the argument converted to Scalar.
   */
#ifdef __cpp_concepts
  template<std::floating_point Scalar>
  constexpr std::floating_point decltype(auto)
  real_projection(scalar_type auto&& arg)
  requires std::convertible_to<std::decay_t<decltype(real_projection(arg))>, Scalar>
#else
  template<typename Scalar, typename Arg, std::enable_if_t<std::is_floating_point_v<Scalar> and scalar_type<Arg> and
    std::is_convertible_v<decltype(real_projection(std::declval<Arg&&>())), std::decay_t<Scalar>>, int> = 0>
  constexpr decltype(auto) real_projection(Arg&& arg)
#endif
  {
    if constexpr (complex_number<decltype(arg)>)
      return static_cast<std::decay_t<Scalar>>(real_projection(std::forward<decltype(arg)>(arg)));
    else
      return std::forward<decltype(arg)>(arg);
  }


  /**
   * \brief The inverse of \ref real_projection.
   * \details This takes a real number (\ref std::floating_point) and recovers a corresponding scalar value
   * from which it would have been a projection. This function must obey the following identity for all
   * <code>x</code> of type Scalar: <code>x == inverse_real_projection(x, real_projection(x))</code>.
   * For example, if the argument is a complex number, the result of this function is a complex number whose real
   * part is updated with the value p of floating type RealProj.
   * \tparam Scalar a \ref scalar_type
   * \param real_projection A \ref std::floating_point argument representing a hypothetical result of \ref real_projection.
   */
#ifdef __cpp_concepts
  constexpr /*floating_scalar_type*/ decltype(auto)
  inverse_real_projection(floating_scalar_type auto&& arg, std::decay_t<decltype(real_projection(arg))> real_projection)
#else
  template<typename Scalar, std::enable_if_t<floating_scalar_type<Scalar>, int> = 0>
  constexpr decltype(auto) inverse_real_projection(Scalar&& arg, std::decay_t<decltype(real_projection(arg))> real_projection)
#endif
  {
    return interface::ScalarTraits<std::decay_t<decltype(arg)>>::template inverse_real_projection(
      std::forward<decltype(arg)>(arg), real_projection);
  }


  /**
   * \brief Return the imaginary part of a complex number.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar>
#else
  template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto) imaginary_part(Scalar&& scalar)
  {
    if constexpr (complex_number<Scalar>)
      return ScalarTraits<std::decay_t<Scalar>>::imag(std::forward<Scalar>(scalar));
    else
      return static_cast<std::decay_t<Scalar>>(0);
  }


  /**
   * \brief Return the complex conjugate of a number.   */
#ifdef __cpp_concepts
  template<scalar_type Scalar>
#else
  template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto) conjugate(Scalar&& scalar)
  {
    if constexpr (complex_number<Scalar>)
      return ScalarTraits<std::decay_t<Scalar>>::conj(std::forward<Scalar>(scalar));
    else
      return std::forward<Scalar>(scalar);
  }


  /**
   * \brief Return the sine of a number.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar>
#else
  template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto) sine(Scalar&& scalar)
  {
    return ScalarTraits<std::decay_t<Scalar>>::sin(std::forward<Scalar>(scalar));
  }


  /**
   * \brief Return the cosine of a number.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar>
#else
  template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto) cosine(Scalar&& scalar)
  {
    return ScalarTraits<std::decay_t<Scalar>>::cos(std::forward<Scalar>(scalar));
  }


  /**
   * \brief Return the square root of a number.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar>
#else
  template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
  constexpr decltype(auto) square_root(Scalar&& scalar)
  {
    return ScalarTraits<std::decay_t<Scalar>>::sqrt(std::forward<Scalar>(scalar));
  }


  /**
   * \brief Return the arcsine of the ratio Y / R, taking account the correct quadrant.
   * \tparam Y A y-axis coordinate.
   * \tparam R A radius (distance from origin).
   */
#ifdef __cpp_concepts
  template<scalar_type Y, scalar_type R>
#else
  template<typename Y, typename R, std::enable_if_t<scalar_type<Y> and scalar_type<R>, int> = 0>
#endif
  constexpr decltype(auto) arcsine2(Y&& y, R&& r)
  {
    return ScalarTraits<std::decay_t<Y>>::asin2(std::forward<Y>(y), std::forward<R>(r));
  }


  /**
   * \brief Return the arctangent of the ratio Y / X, taking account the correct quadrant.
   * \tparam Y A y-axis coordinate.
   * \tparam X An x-axis coordinate.
   */
#ifdef __cpp_concepts
  template<scalar_type Y, scalar_type X>
#else
  template<typename Y, typename X, std::enable_if_t<scalar_type<Y> and scalar_type<X>, int> = 0>
#endif
  constexpr decltype(auto) arctangent2(Y&& y, X&& x)
  {
    return ScalarTraits<std::decay_t<Y>>::atan2(std::forward<Y>(y), std::forward<X>(x));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_FUNCTIONS_HPP
