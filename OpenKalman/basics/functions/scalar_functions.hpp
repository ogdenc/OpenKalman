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
   * \brief Make a complex number from real and imaginary parts.
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  constexpr complex_number auto
  make_complex_number(scalar_type auto&& re, scalar_type auto&& im = 0)
    requires (not complex_number<decltype(re)>) and std::same_as<std::decay_t<decltype(re)>, std::decay_t<decltype(im)>>
#else
  template<typename Re, typename Im, std::enable_if_t<scalar_type<Re> and (not complex_number<Re>) and
    std::is_same_v<std::decay_t<Re>, std::decay_t<Im>>, int> = 0>
  constexpr auto make_complex_number(Re&& re, Im&& im)
#endif
  {
    using R = std::decay_t<decltype(re)>;
    return interface::ScalarTraits<R>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
  }


  /**
   * \brief Make a complex number of type T from real and imaginary parts.
   * \param re The real part.
   * \param im The imaginary part.
   */
#ifdef __cpp_concepts
  template<typename T>
  constexpr complex_number auto
  make_complex_number(scalar_type auto&& re, scalar_type auto&& im = 0)
  requires std::same_as<std::decay_t<decltype(re)>, std::decay_t<decltype(im)>>
#else
  template<typename T, typename Re, typename Im, std::enable_if_t<scalar_type<Re> and
    std::is_same_v<std::decay_t<Re>, std::decay_t<Im>>, int> = 0>
  constexpr auto make_complex_number(Re&& re, Im&& im)
#endif
  {
    return interface::ScalarTraits<std::decay_t<T>>::make_complex(std::forward<decltype(re)>(re), std::forward<decltype(im)>(im));
  }


  /**
   * \brief Return the imaginary part of a complex number.
   */
#ifdef __cpp_concepts
  constexpr floating_scalar_type decltype(auto)
  real_part(scalar_type auto&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_type<Arg>, int> = 0>
  constexpr decltype(auto) real_part(Arg&& arg)
#endif
  {
    return interface::ScalarTraits<std::decay_t<decltype(arg)>>::real_part(std::forward<decltype(arg)>(arg));
  }


  /**
   * \brief Return the imaginary part of a complex number.
   */
#ifdef __cpp_concepts
  constexpr floating_scalar_type decltype(auto)
  imaginary_part(scalar_type auto&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_type<Arg>, int> = 0>
  constexpr decltype(auto) imaginary_part(Arg&& arg)
#endif
  {
    return interface::ScalarTraits<std::decay_t<decltype(arg)>>::imaginary_part(std::forward<decltype(arg)>(arg));
  }


  namespace internal
  {
    /**
     * \internal
     * \brief The inverse of \ref real_part.
     * \details This takes a real number (\ref std::floating_point) and recovers a corresponding scalar value
     * from which it would have been a projection. This function must obey the following identity for all
     * <code>x</code> of type Scalar: <code>x == inverse_real_projection(x, real_part(x))</code>.
     * For example, if the argument is a complex number, the result of this function is a complex number whose real
     * part is updated with the value p of floating type RealProj.
     * \tparam Scalar a \ref scalar_type
     * \param re A \ref std::floating_point argument representing a hypothetical result of \ref real_part.
     */
  #ifdef __cpp_concepts
    constexpr scalar_type decltype(auto)
    inverse_real_projection(scalar_type auto&& arg, floating_scalar_type auto&& re)
      requires std::same_as<std::decay_t<decltype(real_part(arg))>, std::decay_t<decltype(re)>>
  #else
    template<typename Arg, typename Re, std::enable_if_t<scalar_type<Arg> and floating_scalar_type<Re> and
      std::is_same_v<std::decay_t<decltype(real_part(std::declval<Arg>()))>, std::decay_t<Re>>, int> = 0>
    constexpr decltype(auto) inverse_real_projection(Arg&& arg, Re&& re)
  #endif
    {
      using S = std::decay_t<decltype(arg)>;
      if constexpr (complex_number<S>)
      {
        auto im = imaginary_part(std::forward<decltype(arg)>(arg));
        auto ret = make_complex_number(static_cast<std::decay_t<decltype(im)>>(re), std::move(im));
        return static_cast<S>(std::move(ret));
      }
      else return std::forward<decltype(re)>(re);
    }
  }


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
      return are_within_tolerance<epsilon_factor>(real_part(arg1), real_part(arg2)) and
        are_within_tolerance<epsilon_factor>(imaginary_part(arg1), imaginary_part(arg2));
    }
    else
    {
      auto diff = arg1 - arg2;
      using Diff = decltype(diff);
      constexpr auto ep = epsilon_factor * std::numeric_limits<Diff>::epsilon();
      return -static_cast<Diff>(ep) <= diff and diff <= static_cast<Diff>(ep);
    }

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
