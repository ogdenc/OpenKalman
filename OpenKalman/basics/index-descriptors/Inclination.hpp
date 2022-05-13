/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Inclination class and related limits.
 */

#ifndef OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
#define OPENKALMAN_COEFFICIENTS_INCLINATION_HPP

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::down)> and
    std::floating_point<decltype(Limits<double>::up)> and (Limits<double>::down < Limits<double>::up)
#endif
  struct Inclination;


  /// Namespace for definitions relating to coefficients representing an inclination.
  namespace inclination
  {
    /// Namespace for classes describing the numerical limits for an Inclination.
    namespace limits
    {
      /**
       * The limits of an inclination measured in radians: [-½&pi;,½&pi;].
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Radians
      {
        static constexpr Scalar up = std::numbers::pi_v<Scalar> / 2;
        static constexpr Scalar down = -std::numbers::pi_v<Scalar> / 2;
      };


      /**
       * The limits of an inclination measured in degrees: [-90,90].
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Degrees
      {
        static constexpr Scalar up = 90;
        static constexpr Scalar down = -90;
      };

    } // namespace limits

    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<limits::Radians>;

    /// An inclination measured in degrees [-90,90].
    using Degrees = Inclination<limits::Degrees>;

  } // namespace inclination


  /**
   * \brief A positive or negative real number &phi; representing an inclination or declination from the horizon.
   * \details &phi;<sub>down</sub>&le;&phi;&le;&phi;<sub>up</sub>, where &phi;<sub>down</sub> is a real number
   * representing down, and &phi;<sub>up</sub> is a real number representing up. Normally, the horizon will be zero and
   * &phi;<sub>down</sub>=&minus;&phi;<sub>up</sub>, but in general, the horizon is at
   * &frac12;(&phi;<sub>down</sub>+&minus;&phi;<sub>up</sub>).
   * The inclinations inclination::Radians and inclination::Degrees are predefined.
   * \tparam Limits A class template defining the real values <code>down</code> and <code>up</code>, where
   * <code>down</code>=&phi;<sub>down</sub> and <code>up</code>=&phi;<sub>up</sub>.
   * Scalar is a std::floating_point type.
   */
  template<template<typename Scalar> typename Limits = inclination::limits::Radians>
#ifdef __cpp_concepts
    requires std::floating_point<decltype(Limits<double>::down)> and
      std::floating_point<decltype(Limits<double>::up)> and (Limits<double>::down < Limits<double>::up)
#endif
  struct Inclination
  {
#ifndef __cpp_concepts
    static_assert(std::is_floating_point_v<decltype(Limits<double>::down)>)
    static_assert(std::is_floating_point_v<decltype(Limits<double>::up)>);
    static_assert(Limits<double>::down < Limits<double>::up);
#endif

  private:

    template<typename Scalar>
    static constexpr Scalar cf = std::numbers::pi_v<Scalar> / (Limits<Scalar>::up - Limits<Scalar>::down);

    template<typename Scalar>
    static constexpr Scalar horiz = (Limits<Scalar>::up + Limits<Scalar>::down) / 2;

  public:

    /**
     * \brief Maps an inclination angle element to coordinates in Euclidean space.
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
     * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean space,
     * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
     * so Limits<Scalar>::max wraps back to the point (-1, 0).
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index accessing either the x (if 0) or y (if 1) coordinate in Euclidean space
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    to_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t euclidean_local_index, std::size_t start)
    {
      auto theta = cf<Scalar> * (g(start) - horiz<Scalar>); // Convert to radians
      if (euclidean_local_index == 0)
        return std::cos(theta);
      else
        return std::sin(theta);
    }


    /**
     * \brief Return a functor mapping coordinates in Euclidean space to an inclination angle element.
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param euclidean_start The starting location of the x and y coordinates within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    from_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t euclidean_start)
    {
      const auto x = std::abs(g(euclidean_start)); // In Euclidean space, x must be non-negative since the inclination is in range [-½pi,½pi].
      const auto y = g(euclidean_start + 1);
      if constexpr (not std::numeric_limits<Scalar>::is_iec559)
        if (x == 0) return (y == 0) ? horiz<Scalar> : std::signbit(y) ? Limits<Scalar>::down : Limits<Scalar>::up;
      return std::atan2(y, x) / cf<Scalar> + horiz<Scalar>;
    }

  private:

    template<typename Scalar>
    static constexpr Scalar wrap_impl(const Scalar s)
    {
      constexpr Scalar up = Limits<Scalar>::up;
      constexpr Scalar down = Limits<Scalar>::down;
      if (s >= down and s <= up)
      {
        return s;
      }
      else
      {
        constexpr Scalar range = up - down;
        constexpr Scalar period = range * 2;
        Scalar a = std::fmod(s - down, period);
        if (a < 0) a += period;
        if (a > range) a = period - a;
        return a + down;
      }
    }

  public:

    /**
     * \brief Return a functor wrapping an inclination angle.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
     * \param start The starting location of the inclination angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr auto
    wrap_get_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start)
    {
      return wrap_impl(g(start));
    };


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the inclination angle and then mapping to, and then back from,
     * Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
     * \param start The starting location of the inclination angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<std::floating_point Scalar>
#else
    template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
    static constexpr void
    wrap_set_element(const std::function<void(Scalar, std::size_t)>& s, const std::function<Scalar(std::size_t)>& g,
      Scalar x, std::size_t local_index, std::size_t start)
    {
      s(wrap_impl(x), start);
    }

  };


  /**
   * \brief Inclination is represented by one coordinate.
   */
  template<template<typename Scalar> typename Limits>
  struct dimension_size_of<Inclination<Limits>> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Inclination<Limits>&) { return 1; }
  };


  /**
   * \brief Inclination is represented by two coordinates in Euclidean space.
   */
  template<template<typename Scalar> typename Limits>
  struct euclidean_dimension_size_of<Inclination<Limits>> : std::integral_constant<std::size_t, 2>
  {
    constexpr static std::size_t get(const Inclination<Limits>&) { return 2; }
  };


  /**
   * \brief The number of atomic components.
   */
  template<template<typename Scalar> typename Limits>
  struct index_descriptor_components_of<Inclination<Limits>> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Inclination<Limits>&) { return 1; }
  };


  /**
   * \brief A difference between two Inclination values does not wrap, and is treated as Axis.
   * \details See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1555 (2015).
   */
  template<template<typename Scalar> typename Limits>
  struct dimension_difference_of<Inclination<Limits>> { using type = Axis; };

} // namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
