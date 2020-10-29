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

namespace OpenKalman
{
  // ----------------------------- //
  //   Atomic coefficient groups   //
  // ----------------------------- //

  /**
   * A real number, [&minus;&infin;,&infin].
   */
  struct Axis;

  /**
   * A modular value, such as an angle or wrapping unit circle.
   * @tparam Traits A class defining the real values <code>wrap_max</code> and <code>wrap_min</code>, which are the
   * minimum and maximum values, respectively, beyond which wrapping occurs.
   */
  template<typename Traits>
  struct Circle;

  /**
   * A non-negative real number, representing a distance.
   */
  struct Distance;

  /**
   * A positive or negative real number representing an inclination or declination from the horizon.
   * @tparam Traits A class defining the real value <code>max</code> and <code>min</code>, which are the
   * maximum and minimum values reflecting up and down, respectively. Normally, the horizon will be zero,
   * but in any event, the horizon will be <code>(max+min)/2</code>
   */
  template<typename Traits>
  struct Inclination;

  /**
   * A group reflecting polar coordinates.
   *
   * Coefficient1 and Coefficient2 must be some combination of Distance and Circle/Angle, such as
   * Polar<Distance, Angle> or Polar<Angle, Distance>.
   * @tparam Coefficient1 Either Distance or Circle (e.g., Angle).
   * @tparam Coefficient2 Either Distance or Circle (e.g., Angle).
   */
  template<typename Coefficient1, typename Coefficient2>
  struct Polar;

  /**
   * A group reflecting spherical coordinates.
   *
   * Coefficient1, Coefficient2, and Coefficient3 must be some combination of Distance, Inclination, and Circle/Angle,
   * reflecting the distance, inclination, and azimuth, respectively.
   * Examples: Spherical<Distance, Inclination, Angle>, Spherical<Angle, Distance, Inclination>.
   * @tparam Coefficient1 Distance, inclination, or Circle (e.g., Angle).
   * @tparam Coefficient2 Distance, inclination, or Circle (e.g., Angle).
   * @tparam Coefficient3 Distance, inclination, or Circle (e.g., Angle).
   */
  template<typename Coefficient1, typename Coefficient2, typename Coefficient3>
  struct Spherical;

  namespace internal
  {
    template<typename T>
    struct is_atomic_coefficient_group : std::false_type {};

    template<>
    struct is_atomic_coefficient_group<Axis> : std::true_type {};

    template<typename Traits>
    struct is_atomic_coefficient_group<Circle<Traits>> : std::true_type {};

    template<>
    struct is_atomic_coefficient_group<Distance> : std::true_type {};

    template<typename Traits>
    struct is_atomic_coefficient_group<Inclination<Traits>> : std::true_type {};

    template<typename Coeff1, typename Coeff2>
    struct is_atomic_coefficient_group<Polar<Coeff1, Coeff2>> : std::true_type {};

    template<typename Coeff1, typename Coeff2, typename Coeff3>
    struct is_atomic_coefficient_group<Spherical<Coeff1, Coeff2, Coeff3>> : std::true_type {};
  }


  // -------------------------------- //
  //   Composite coefficient groups   //
  // -------------------------------- //

  /**
   * A set of coefficient types to be associated with the rows or columns of a matrix.
   *
   * Each coefficient Cs can be a single coefficient (e.g., Axis, Circle, Distance, Inclination),
   * an atomic coefficient group (e.g., Polar, Spherical) or a composite coefficient (e.g., Coefficient<Axis, Angle>).
   * @tparam Cs Any types within the concept coefficients (type trait is_coefficients in c++17).
   */
#ifdef __cpp_concepts
  template<coefficients ... Cs>
#else
  template<typename ... Cs>
#endif
  struct Coefficients;


  namespace internal
  {
    /**
     * A type trait testing whether T is a composite set of coefficient groups.
     *
     * This corresponds to any specialization of the class Coefficients. Composite coefficients can, themselves,
     * comprise groups of other composite components. For example, Coefficients<Axis, Coefficients<Axis, Angle>>
     * tests positive for is_composite_coefficients.
     */
    template<typename T>
    struct is_composite_coefficients : std::false_type {};

    template<typename...C>
    struct is_composite_coefficients<Coefficients<C...>> : std::true_type {};
  }


  // ----------------------------- //
  //   equivalent, is_equivalent   //
  // ----------------------------- //

  // General case.
#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U, typename Enable>
#endif
  struct is_equivalent : std::false_type {};

  template<>
  struct is_equivalent<Axis, Axis> : std::true_type {};

  template<typename Traits>
  struct is_equivalent<Circle<Traits>, Circle<Traits>> : std::true_type {};

  template<>
  struct is_equivalent<Distance, Distance> : std::true_type {};

  template<typename Traits>
  struct is_equivalent<Inclination<Traits>, Inclination<Traits>> : std::true_type {};

  template<typename Coeff1a, typename Coeff2a, typename Coeff1b, typename Coeff2b>
  struct is_equivalent<Polar<Coeff1a, Coeff2a>, Polar<Coeff1b, Coeff2b>>
    : std::bool_constant<is_equivalent_v<Coeff1a, Coeff1b> and is_equivalent_v<Coeff2a, Coeff2b>> {};

  template<typename Coeff1a, typename Coeff2a, typename Coeff3a, typename Coeff1b, typename Coeff2b, typename Coeff3b>
  struct is_equivalent<Spherical<Coeff1a, Coeff2a, Coeff3a>, Spherical<Coeff1b, Coeff2b, Coeff3b>>
    : std::bool_constant<is_equivalent_v<Coeff1a, Coeff1b> and
      is_equivalent_v<Coeff2a, Coeff2b> and is_equivalent_v<Coeff3a, Coeff3b>> {};


#ifdef __cpp_concepts
  template<coefficients...C1, coefficients...C2> requires
    (sizeof...(C1) == 0 and sizeof...(C2) == 0) or
    (sizeof...(C1) > 1 and sizeof...(C2) > 1 and (is_equivalent_v<C1, C2> and ...))
  struct is_equivalent<Coefficients<C1...>, Coefficients<C2...>>
#else
  template<typename...C1, typename...C2>
  struct is_equivalent<Coefficients<C1...>, Coefficients<C2...>, std::enable_if_t<
    (sizeof...(C1) == 0 and sizeof...(C2) == 0) or
    (sizeof...(C1) > 1 and sizeof...(C2) > 1 and (is_equivalent_v<C1, C2> and ...))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<typename T, typename U> requires is_equivalent_v<T, U>
  struct is_equivalent<T, Coefficients<U>>
#else
  template<typename T, typename U>
  struct is_equivalent<T, Coefficients<U>, std::enable_if_t<is_equivalent_v<T, U>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<typename T, typename U> requires is_equivalent_v<T, U> and internal::is_atomic_coefficient_group<U>::value
  struct is_equivalent<Coefficients<T>, U>
#else
  template<typename T, typename U>
  struct is_equivalent<Coefficients<T>, U, std::enable_if_t<
    is_equivalent_v<T, U> and internal::is_atomic_coefficient_group<U>::value>>
#endif
  : std::true_type {};


  // ------------- //
  //   is_prefix   //
  // ------------- //

#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U, typename Enable>
#endif
  struct is_prefix : std::false_type {};

#ifdef __cpp_concepts
  template<coefficients C1, coefficients C2> requires equivalent<C1, C2>
  struct is_prefix<C1, C2>
#else
  template<typename C1, typename C2>
  struct is_prefix<C1, C2, std::enable_if_t<is_equivalent_v<C1, C2>>>
#endif
  : std::true_type {};

#ifdef __cpp_concepts
  template<coefficients Ca, coefficients Cb, coefficients...C1, coefficients...C2> requires
    equivalent<Ca, Cb> and
    is_prefix_v<Coefficients<C1...>, Coefficients<C2...>> and
    (not equivalent<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>)
  struct is_prefix<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>
#else
  template<typename Ca, typename Cb, typename...C1, typename...C2>
  struct is_prefix<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>,
    std::enable_if_t<is_equivalent_v<Ca, Cb> and is_prefix_v<Coefficients<C1...>, Coefficients<C2...>> and
      not is_equivalent_v<Coefficients<Ca, C1...>, Coefficients<Cb, C2...>>>>
#endif
  : std::true_type {};

#ifdef __cpp_concepts
  template<coefficients C> requires (not equivalent<Coefficients<>, C>)
  struct is_prefix<Coefficients<>, C>
#else
  template<typename C>
  struct is_prefix<Coefficients<>, C,
    std::enable_if_t<is_coefficients_v<C> and not is_equivalent_v<Coefficients<>, C>>>
#endif
  : std::true_type {};

#ifdef __cpp_concepts
  template<coefficients C, coefficients...C1>
  struct is_prefix<C, Coefficients<C, C1...>>
#else
  template<typename C, typename...C1>
  struct is_prefix<C, Coefficients<C, C1...>, std::enable_if_t<is_coefficients_v<C>>>
#endif
  : std::true_type {};


  // --------------------------------- //
  //   Concatenation of coefficients   //
  // --------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<coefficients ...>
#else
    template<typename ...>
#endif
    struct ConcatenateImpl;

    template<>
    struct ConcatenateImpl<>
    {
      using type = Coefficients<>;
    };

#ifdef __cpp_concepts
    template<typename Cs1, coefficients ... Coeffs> requires internal::is_atomic_coefficient_group<Cs1>::value
#else
    template<typename Cs1, typename ... Coeffs>
#endif
    struct ConcatenateImpl<Cs1, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1>;
    };

#ifdef __cpp_concepts
    template<coefficients ... Cs1, coefficients ... Coeffs>
#else
    template<typename ... Cs1, typename ... Coeffs>
#endif
    struct ConcatenateImpl<Coefficients<Cs1...>, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Cs1...>;
    };
  }

  /**
   * Concatenate any number of Coefficients<...> types.
   */
#ifdef __cpp_concepts
  template<coefficients ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#else
  template<typename ... Coeffs> using Concatenate = typename detail::ConcatenateImpl<Coeffs...>::type;
#endif

}

#endif //OPENKALMAN_COEFFICIENT_FORWARD_DECLARATIONS_HPP
