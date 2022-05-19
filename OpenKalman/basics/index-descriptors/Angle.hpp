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
 * \brief Definition of the Angle class and related limits.
 */

#ifndef OPENKALMAN_ANGLE_HPP
#define OPENKALMAN_ANGLE_HPP

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman
{
  template<template<typename Scalar> typename Limits>
#ifdef __cpp_concepts
  requires std::floating_point<decltype(Limits<double>::min)> and
    std::floating_point<decltype(Limits<double>::max)> and (Limits<double>::min < Limits<double>::max) and
    (Limits<double>::min <= 0) and (0 <= Limits<double>::max)
#endif
  struct Angle;


  /// Namespace for definitions relating to typed_index_descriptors representing an angle.
  namespace angle
  {
    /// Namespace for classes describing the numerical limits for an angle.
    namespace limits
    {
      /**
       * The limits of an angle measured in radians: [-&pi;,&pi;).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Radians
      {
        static constexpr Scalar max = std::numbers::pi_v<Scalar>;
        static constexpr Scalar min = -std::numbers::pi_v<Scalar>;
      };


      /**
       * The limits of an angle measured in positive radians: [0,2&pi;).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct PositiveRadians
      {
        static constexpr Scalar max = 2 * std::numbers::pi_v<Scalar>;
        static constexpr Scalar min = 0;
      };


      /**
       * The limits of an angle measured in positive or negative degrees: [-180,180).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Degrees
      {
        static constexpr Scalar max = 180;
        static constexpr Scalar min = -180;
      };


      /**
       * The limits of an angle measured in positive degrees: [0,360).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct PositiveDegrees
      {
        static constexpr Scalar max = 360;
        static constexpr Scalar min = 0;
      };


      /**
       * The limits of a wrapping circle, such as the wrapping interval [0,1).
       * \tparam Scalar The scalar type (e.g., <code>double</code>).
       */
      template<typename Scalar = double>
#ifdef __cpp_concepts
      requires std::floating_point<Scalar>
#endif
      struct Circle
      {
        static constexpr Scalar min = 0;
        static constexpr Scalar max = 1;
      };

    } // namespace limits


    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<limits::Radians>;


    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<limits::PositiveRadians>;


    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<limits::PositiveDegrees>;


    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<limits::Degrees>;


    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<limits::Circle>;

  } // namespace angle


  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [max,min) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam Limits A class template defining the real values <code>min</code> and <code>max</code>, representing
   * minimum and maximum values, respectively, beyond which wrapping occurs. This range must include 0.
   * <code>Scalar</code> is a <code>std::floating_point</code> type.
   */
  template<template<typename Scalar> typename Limits = angle::limits::Radians>
#ifdef __cpp_concepts
    requires std::floating_point<decltype(Limits<double>::min)> and
      std::floating_point<decltype(Limits<double>::max)> and (Limits<double>::min < Limits<double>::max) and
      (Limits<double>::min <= 0) and (0 <= Limits<double>::max)
#endif
  struct Angle
  {
#ifndef __cpp_concepts
    static_assert(std::is_floating_point_v<decltype(Limits<double>::min)>);
    static_assert(std::is_floating_point_v<decltype(Limits<double>::max)>);
    static_assert(Limits<double>::min < Limits<double>::max);
    static_assert((Limits<double>::min <= 0) and (0 <= Limits<double>::max));
#endif

  private:

    template<typename Scalar>
    static constexpr Scalar cf = 2 * std::numbers::pi_v<Scalar> / (Limits<Scalar>::max - Limits<Scalar>::min);

    template<typename Scalar>
    static constexpr Scalar mid = (Limits<Scalar>::max + Limits<Scalar>::min) / 2;

  public:

    /**
     * \brief Maps an angle element to coordinates in Euclidean space.
     * \details The angle corresponds to x and y coordinates on a unit circle.
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
      const auto theta = cf<Scalar> * (g(start) - mid<Scalar>); // Convert to radians
      if (euclidean_local_index == 0)
        return std::cos(theta);
      else
        return std::sin(theta);
    }


    /**
     * \brief Return a functor mapping coordinates in Euclidean space to an angle element.
     * \details The angle corresponds to x and y coordinates on a unit circle.
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
      const auto x = g(euclidean_start);
      const auto y = g(euclidean_start + 1);
      if constexpr (not std::numeric_limits<Scalar>::is_iec559)
        if (x == 0) return (y == 0) ? mid<Scalar> : std::signbit(y) ? Limits<Scalar>::down : Limits<Scalar>::up;
      return std::atan2(y, x) / cf<Scalar> + mid<Scalar>;
    }


  private:

    template<typename Scalar>
    static constexpr Scalar wrap_impl(const Scalar a)
    {
      constexpr Scalar max = Limits<Scalar>::max;
      constexpr Scalar min = Limits<Scalar>::min;
      if (a >= min and a < max)
      {
        return a;
      }
      else
      {
        constexpr Scalar period = max - min;
        Scalar ar = std::fmod(a - min, period);
        if (ar < 0) ar += period;
        return ar + min;
      }
    }

  public:

    /**
     * \brief Return a functor wrapping an angle.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
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
    }


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
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


  namespace interface
  {
    /**
     * \internal
     * \brief Angle is represented by one coordinate.
     */
    template<template<typename Scalar> typename Limits>
    struct IndexDescriptorSize<Angle<Limits>> : std::integral_constant<std::size_t, 1>
    {
      constexpr static std::size_t get(const Angle<Limits>&) { return 1; }
    };


    /**
     * \internal
     * \brief Angle is represented by two coordinates in Euclidean space.
     */
    template<template<typename Scalar> typename Limits>
    struct EuclideanIndexDescriptorSize<Angle<Limits>> : std::integral_constant<std::size_t, 2>
    {
      constexpr static std::size_t get(const Angle<Limits>&) { return 2; }
    };


    /**
     * \internal
     * \brief The number of atomic components.
     */
    template<template<typename Scalar> typename Limits>
    struct IndexDescriptorComponentCount<Angle<Limits>> : std::integral_constant<std::size_t, 1>
    {
      constexpr static std::size_t get(const Angle<Limits>&) { return 1; }
    };


    /**
     * \internal
     * \brief The type of the result when subtracting two Angle values.
     * \details A distance between two points on a circle cannot be more than the circumference of the circle,
     * so it must be wrapped as an Angle.
     * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
     * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
     */
    template<template<typename Scalar> typename Limits>
    struct IndexDescriptorDifferenceType<Angle<Limits>> { using type = Angle<Limits>; };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_ANGLE_HPP
