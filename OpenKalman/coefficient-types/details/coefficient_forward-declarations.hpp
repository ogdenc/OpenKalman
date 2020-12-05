/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward definitions for coefficients.
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
   * \brief A set of coefficient types.
   * \details This is the key to the wrapping functionality of OpenKalman. Each of the coefficients Cs... matches-up with
   * one or more of the rows or columns of a matrix. The number of coefficients per coefficient depends on the size
   * of the coefficient. For example, Axis, Distance, Angle, and Inclination are size 1, and each correspond to a
   * single coefficient. Polar is size 2 and corresponds to two coefficients (e.g., a distance and an angle).
   * Spherical is size 3 and corresponds to three coefficients.
   * Example: <code>Coefficients&lt;Axis, angle::Radians&gt;</code>
   * \sa Specializations: Coefficients<>, \ref CoefficientsCCs "Coefficients<C, Cs...>"
   * \tparam Cs Any types within the concept coefficients (internal bool variable coefficients in c++17).
   */
#ifdef __cpp_concepts
  template<coefficients...Cs>
#else
  template<typename...Cs>
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
   * \struct Axis
   * \brief A real or integral number, (&minus;&infin;,&infin;).
   * \details This is the default coefficient type. No wrapping occurs, and matrices operate as usual.
   */
  struct Axis;


  /**
   * \struct Distance
   * \brief A non-negative real or integral number, [0,&infin;], representing a distance.
   * \details This is similar to Axis, but wrapping occurs to ensure that values are never negative.
   */
  struct Distance;


  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [max,min) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * \tparam Limits A class template defining the real values <code>min</code> and <code>max</code>, representing
   * minimum and maximum values, respectively, beyond which wrapping occurs. Scalar is a std::floating_point type.
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
   * The inclinations inclination::Radians and inclination::Degrees are predefined.
   * \tparam Limits A class template defining the real values <code>down</code> and <code>up</code>, where
   * <code>down</code>=&phi;<sub>down</sub> and <code>up</code>=&phi;<sub>up</sub>.
   * Scalar is a std::floating_point type.
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
   * <code>Polar&lt;Distance, angle::Radians&gt; or Polar&lt;angle::Degrees, Distance&gt;</code>.
   * Polar coordinates span two adjacent coefficients in a matrix.
   * \tparam C1, C2 Distance and Angle, in either order. By default, they are Distance and angle::Radians, respectively.
   */
  template<typename C1, typename C2>
  struct Polar;


  /**
   * \brief An atomic coefficient group reflecting spherical coordinates.
   * \details Coefficient1, Coefficient2, and Coefficient3 must be some combination of Distance, Inclination, and Angle
   * in any order, reflecting the distance, inclination, and azimuth, respectively.
   * Spherical coordinates span three adjacent coefficients in a matrix.
   * Examples: <code>Spherical&lt;Distance, inclination::Degrees, angle::Radians&gt;,
   * Spherical&lt;angle::PositiveDegrees, Distance, inclination::Radians&gt;</code>
   * \tparam Coefficient1, Coefficient2, Coefficient3 Distance, inclination, and Angle, in any order.
   * By default, they are Distance, angle::Radians, and inclination::Radians, respectively.
   */
  template<typename C1, typename C2, typename C3>
  struct Spherical;


  namespace internal
  {
    template<typename T>
    struct is_atomic_coefficient_group : std::false_type {};

    template<>
    struct is_atomic_coefficient_group<Axis> : std::true_type {};

    template<>
    struct is_atomic_coefficient_group<Distance> : std::true_type {};

    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Angle<Limits>> : std::true_type {};

    template<template<typename Scalar> typename Limits>
    struct is_atomic_coefficient_group<Inclination<Limits>> : std::true_type {};

    template<typename C1, typename C2>
    struct is_atomic_coefficient_group<Polar<C1, C2>> : std::true_type {};

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


#ifdef __cpp_concepts
    template<coefficients C> requires (not equivalent_to<Coefficients<>, C>)
    struct is_prefix_of<Coefficients<>, C>
#else
    template<typename C>
    struct is_prefix_of<Coefficients<>, C,
      std::enable_if_t<coefficients<C> and not equivalent_to<Coefficients<>, C>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<coefficients C, coefficients...Cs>
    struct is_prefix_of<C, Coefficients<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct is_prefix_of<C, Coefficients<C, Cs...>, std::enable_if_t<coefficients<C>>>
#endif
      : std::true_type {};


    /**
     * \internal
     * \brief A sufficiently-defined class representing an atomic or composite group of coefficients.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept coefficient_class =
    std::integral<decltype(T::size)> and std::integral<decltype(T::dimension)> and
      (T::axes_only or not T::axes_only) and
      (internal::is_atomic_coefficient_group<typename T::difference_type>::value or
        internal::is_composite_coefficients<typename T::difference_type>::value) and
      requires {T::template to_euclidean_array<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template from_euclidean_array<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template wrap_array_get<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template wrap_array_set<double, 0>[0](0., std::function<void(const double, const std::size_t)>(),
          std::function<double(const std::size_t)>());} and
      (std::tuple_size_v<decltype(T::template to_euclidean_array<double, 0>)> == T::dimension) and
      (std::tuple_size_v<decltype(T::template from_euclidean_array<double, 0>)> == T::size) and
      (std::tuple_size_v<decltype(T::template wrap_array_get<double, 0>)> == T::size) and
      (std::tuple_size_v<decltype(T::template wrap_array_set<double, 0>)> == T::size);
#else
    inline constexpr bool coefficient_class = true;
#endif

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
   * \details Example: \code Concatenate<Coefficients<angle::Radians>, Coefficients<Axis, Distance>> ==
   * Coefficients<angle::Radians, Axis, Distance> \endcode.
   */
#ifdef __cpp_concepts
  template<coefficients ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#else
  template<typename ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_COEFFICIENT_FORWARD_DECLARATIONS_HPP
