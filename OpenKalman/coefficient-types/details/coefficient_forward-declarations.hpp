/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COEFFICIENT_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_COEFFICIENT_FORWARD_DECLARATIONS_HPP

#include <type_traits>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  // -------------------------------- //
  //   Composite coefficient groups   //
  // -------------------------------- //

  /**
   * A set of coefficient types to be associated with the rows or columns of a matrix.
   *
   * Each coefficient Cs can be a single coefficient (e.g., Axis, Angle, Distance, Inclination),
   * an atomic coefficient group (e.g., Polar, Spherical) or a composite coefficient (e.g., Coefficient<Axis, angle::Radians>).
   * \tparam Cs Any types within the concept coefficients (internal bool variable coefficients in c++17).
   */
#ifdef __cpp_concepts
  template<coefficients ... Cs>
#else
  template<typename ... Cs>
#endif
  struct Coefficients;


  namespace internal
  {
    template<typename T>
    struct is_composite_coefficients : std::false_type {};

    template<typename...C>
    struct is_composite_coefficients<Coefficients<C...>> : std::true_type {};
  }


  // ----------------------------- //
  //   Atomic coefficient groups   //
  // ----------------------------- //

  /**
   * \brief A real or integral number, [&minus;&infin;,&infin].
   */
  struct Axis;


  /**
   * \brief A non-negative real or integral number, [0,&infin], representing a distance.
   */
  struct Distance;


  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [max,min) when it increases or decreases outside that range.
   * \tparam Limits A class template defining the real values <code>min</code> and <code>max</code>, representing
   * minimum and maximum values, respectively, beyond which wrapping occurs. Scalar is a scalar type
   * (e.g., <code>double</code>).
   */
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::min)> and
    std::floating_point<decltype(Limits<double>::max)> and (Limits<double>::min < Limits<double>::max)
#endif
  struct Angle;


  /**
   * \brief A positive or negative real number &phi; representing an inclination or declination from the horizon.
   * \details &phi;<sub>down</sub>&le;&phi;&le;&phi;<sub>up</sub>, where &phi;<sub>down</sub> is a real number
   * representing down, and &phi;<sub>up</sub> is a real number representing up. Normally, the horizon will be zero and
   * &phi;<sub>down</sub>=&minus;&phi;<sub>up</sub>, but in general, the horizon is at
   * &half;(&phi;<sub>down</sub>+&minus;&phi;<sub>up</sub>).
   * \tparam Limits A class template defining the real values <code>down</code> and <code>up</code>, where
   * <code>down</code>=&phi;<sub>down</sub> and <code>up</code>=&phi;<sub>up</sub>. Scalar is a scalar type
   * (e.g., <code>double</code>).
   */
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::down)> and
    std::floating_point<decltype(Limits<double>::up)> and (Limits<double>::down < Limits<double>::up)
#endif
  struct Inclination;


  /**
   * \brief An atomic coefficient group reflecting polar coordinates.
   * \details C1 and C2 are coefficients, and must be some combination of Distance and Angle, such as
   * Polar<Distance, angle::Radians> or Polar<angle::Degrees, Distance>.
   * \tparam C1 Either Distance or Angle.
   * \tparam C2 Either Distance or Angle.
   */
  template<typename C1, typename C2>
  struct Polar;


  /**
   * A group reflecting spherical coordinates.
   *
   * Coefficient1, Coefficient2, and Coefficient3 must be some combination of Distance, Inclination, and Angle/angle::Radians,
   * reflecting the distance, inclination, and azimuth, respectively.
   * Examples: Spherical<Distance, Inclination, angle::Radians>, Spherical<angle::Radians, Distance, Inclination>.
   * \tparam Coefficient1 Distance, inclination, or Angle (e.g., angle::Radians).
   * \tparam Coefficient2 Distance, inclination, or Angle (e.g., angle::Radians).
   * \tparam Coefficient3 Distance, inclination, or Angle (e.g., angle::Radians).
   */
  template<typename C1, typename C2, typename C3>
  struct Spherical;


  namespace internal
  {
    template<typename T>
    struct is_atomic_coefficient_group : std::false_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Axis
     * \interface coefficients<> \sa Axis
     */
    template<>
    struct is_atomic_coefficient_group<Axis> : std::true_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Distance
     * \interface coefficients<> \sa Distance
     */
    template<>
    struct is_atomic_coefficient_group<Distance> : std::true_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Angle<>
     * \interface coefficients<> \sa Angle<>
     */
    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Angle<Limits>> : std::true_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Inclination<>
     * \interface coefficients<> \sa Inclination<>
     */
    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Inclination<Limits>> : std::true_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Polar<>
     * \interface coefficients<> \sa Polar<>
     */
    template<typename C1, typename C2>
    struct is_atomic_coefficient_group<Polar<C1, C2>> : std::true_type {};

    /**
     * \internal \struct is_atomic_coefficient_group<> \sa Spherical<>
     * \interface coefficients<> \sa Spherical<>
     */
    template<typename C1, typename C2, typename C3>
    struct is_atomic_coefficient_group<Spherical<C1, C2, C3>> : std::true_type {};


    // -------------------- //
    //   is_equivalent_to   //
    // -------------------- //

#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable>
#endif
    struct is_equivalent_to : std::false_type {};

    /**
     * \interface equivalent_to<>
     * \note Any coefficient or group of coefficients is equivalent to itself.
     */

    template<>
    struct is_equivalent_to<Axis, Axis> : std::true_type {};


    template<>
    struct is_equivalent_to<Distance, Distance> : std::true_type {};


    template<template<typename Scalar> typename Limits>
    struct is_equivalent_to<Angle<Limits>, Angle<Limits>> : std::true_type {};


    template<template<typename Scalar> typename Limits>
    struct is_equivalent_to<Inclination<Limits>, Inclination<Limits>> : std::true_type {};


    template<typename C1a, typename C2a, typename C1b, typename C2b>
    struct is_equivalent_to<Polar<C1a, C2a>, Polar<C1b, C2b>>
      : std::bool_constant<equivalent_to<C1a, C1b> and equivalent_to<C2a, C2b>> {};


    template<typename C1a, typename C2a, typename C3a, typename C1b, typename C2b, typename C3b>
    struct is_equivalent_to<Spherical<C1a, C2a, C3a>, Spherical<C1b, C2b, C3b>>
      : std::bool_constant<equivalent_to<C1a, C1b> and
        equivalent_to<C2a, C2b> and equivalent_to<C3a, C3b>> {};


    /**
     * \interface equivalent_to<>
     * \note Coefficient<Ts...> is equivalent to Coefficient<Us...>, if each Ts is equivalent to its respective Us.
     */
#ifdef __cpp_concepts
    template<coefficients...C1, coefficients...C2> requires
      (sizeof...(C1) == 0 and sizeof...(C2) == 0) or
      (sizeof...(C1) > 1 and sizeof...(C2) > 1 and (equivalent_to<C1, C2> and ...))
    struct is_equivalent_to<Coefficients<C1...>, Coefficients<C2...>>
#else
    template<typename...C1, typename...C2>
    struct is_equivalent_to<Coefficients<C1...>, Coefficients<C2...>, std::enable_if_t<
      (sizeof...(C1) == 0 and sizeof...(C2) == 0) or
      (sizeof...(C1) > 1 and sizeof...(C2) > 1 and (equivalent_to<C1, C2> and ...))>>
#endif
      : std::true_type {};

    /**
     * \interface equivalent_to<>
     * \note Coefficient<T> is equivalent to T, and vice versa.
     */

#ifdef __cpp_concepts
    template<typename T, typename U> requires equivalent_to<T, U>
    struct is_equivalent_to<T, Coefficients<U>>
#else
    template<typename T, typename U>
    struct is_equivalent_to<T, Coefficients<U>, std::enable_if_t<equivalent_to<T, U>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename T, typename U> requires equivalent_to<T, U> and internal::is_atomic_coefficient_group<U>::value
    struct is_equivalent_to<Coefficients<T>, U>
#else
    template<typename T, typename U>
    struct is_equivalent_to<Coefficients<T>, U, std::enable_if_t<
      equivalent_to<T, U> and internal::is_atomic_coefficient_group<U>::value>>
#endif
    : std::true_type {};


    // ---------------- //
    //   is_prefix_of   //
    // ---------------- //

#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable>
#endif
    struct is_prefix_of : std::false_type {};


    /**
     * \interface prefix_of<>
     * \note T is a prefix of U if equivalent_to<T, U>.
     */
#ifdef __cpp_concepts
    template<coefficients C1, coefficients C2> requires equivalent_to<C1, C2>
    struct is_prefix_of<C1, C2>
#else
    template<typename C1, typename C2>
    struct is_prefix_of<C1, C2, std::enable_if_t<equivalent_to<C1, C2>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<coefficients Ca, coefficients Cb, coefficients...C1, coefficients...C2> requires
      equivalent_to<Ca, Cb> and
      prefix_of<Coefficients<C1...>, Coefficients<C2...>> and
      (not equivalent_to<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>)
    struct is_prefix_of<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>
#else
    template<typename Ca, typename Cb, typename...C1, typename...C2>
    struct is_prefix_of<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>,
      std::enable_if_t<equivalent_to<Ca, Cb> and prefix_of<Coefficients<C1...>, Coefficients<C2...>> and
        not equivalent_to<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>>>
#endif
      : std::true_type {};


    /**
     * \interface prefix_of<>
     * \note Coefficients<> is a prefix of any set of coefficients.
     */
#ifdef __cpp_concepts
    template<coefficients C> requires (not equivalent_to<Coefficients<>, C>)
    struct is_prefix_of<Coefficients<>, C>
#else
    template<typename C>
    struct is_prefix_of<Coefficients<>, C,
      std::enable_if_t<coefficients<C> and not equivalent_to<Coefficients<>, C>>>
#endif
      : std::true_type {};


    /**
     * \interface prefix_of<>
     * \note C is a prefix of Coefficients<C, Cs...> for any coefficients Cs.
     */
#ifdef __cpp_concepts
    template<coefficients C, coefficients...Cs>
    struct is_prefix_of<C, Coefficients<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct is_prefix_of<C, Coefficients<C, Cs...>, std::enable_if_t<coefficients<C>>>
#endif
      : std::true_type {};

  } // namespace internal


  // --------------------------------- //
  //   Concatenation of coefficients   //
  // --------------------------------- //

  namespace detail
  {
    template<typename ...>
    struct ConcatenateImpl;

    template<>
    struct ConcatenateImpl<>
    {
      using type = Coefficients<>;
    };

    template<typename Cs1, typename ... Coeffs>
    struct ConcatenateImpl<Cs1, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1>;
    };

    template<typename ... Cs1, typename ... Coeffs>
    struct ConcatenateImpl<Coefficients<Cs1...>, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1...>;
    };
  }


  /**
   * \brief Concatenate any number of Coefficients<...> types.
   */
#ifdef __cpp_concepts
  template<coefficients ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#else
  template<typename ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_COEFFICIENT_FORWARD_DECLARATIONS_HPP
