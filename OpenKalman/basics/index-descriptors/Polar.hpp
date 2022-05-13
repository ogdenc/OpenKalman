/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of Polar class and associated details.
 */

#ifndef OPENKALMAN_POLAR_HPP
#define OPENKALMAN_POLAR_HPP


namespace OpenKalman
{
  /**
   * \brief An atomic coefficient group reflecting polar coordinates.
   * \details C1 and C2 are coefficients, and must be some combination of Distance and Angle, such as
   * <code>Polar&lt;Distance, angle::Radians&gt; or Polar&lt;angle::Degrees, Distance&gt;</code>.
   * Polar coordinates span two adjacent coefficients in a matrix.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam C1, C2 Distance and Angle, in either order. By default, they are Distance and angle::Radians, respectively.
   * \internal
   */
#ifdef __cpp_concepts
  template<atomic_fixed_index_descriptor C1 = Distance, atomic_fixed_index_descriptor C2 = angle::Radians>
#else
  template<typename C1 = Distance, typename C2 = angle::Radians, typename = void>
#endif
  struct Polar;


  namespace detail
  {
    // Implementation of polar coordinates.
    template<template<typename Scalar> typename Limits,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
    private:

      template<typename Scalar>
      static constexpr Scalar cf = 2 * std::numbers::pi_v<Scalar> / (Limits<Scalar>::max - Limits<Scalar>::min);

      template<typename Scalar>
      static constexpr Scalar mid = (Limits<Scalar>::max + Limits<Scalar>::min) / 2;

    public:

      /**
       * \brief Maps a polar coordinate to coordinates in Euclidean space.
       * \details This function takes a set of polar coordinates and converts them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
       * \tparam Scalar The scalar type (e.g., double).
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
       * \param start The starting index within the index descriptor
       */
#ifdef __cpp_concepts
      template<std::floating_point Scalar>
#else
      template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
      static constexpr auto
      to_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t euclidean_local_index, std::size_t start)
      {
        auto d = g(start + d_i);
        switch(euclidean_local_index)
        {
          case x_i: return std::cos(cf<Scalar> * g(start + a_i) - mid<Scalar>);
          case y_i: return std::sin(cf<Scalar> * g(start + a_i) - mid<Scalar>);
          default:  return d; // case d2_i
        }
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes x, y, and z Cartesian coordinates representing a location on a
       * unit half-cylinder, and converts them to polar coordinates.
       * \tparam Scalar The scalar type (e.g., double).
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting index within the Euclidean-transformed indices
       */
#ifdef __cpp_concepts
      template<std::floating_point Scalar>
#else
      template<typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
#endif
      static constexpr auto
      from_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t euclidean_start)
      {
        auto d = g(euclidean_start + d2_i);
        if (local_index == d_i)
        {
          return std::abs(d); // A negative distance is reflected to the positive axis.
        }
        else
        {
          // If distance is negative, flip 180 degrees:
          const auto x = std::signbit(d) ? -g(euclidean_start + x_i) : g(euclidean_start + x_i);
          const auto y = std::signbit(d) ? -g(euclidean_start + y_i) : g(euclidean_start + y_i);
          if constexpr (not std::numeric_limits<Scalar>::is_iec559)
            if (x == 0) return (y == 0) ? mid<Scalar> : std::signbit(y) ? Limits<Scalar>::down : Limits<Scalar>::up;
          return std::atan2(y, x) / cf<Scalar> + mid<Scalar>;
        }
      }

    private:

      template<typename Scalar>
      static constexpr Scalar polar_angle_wrap_impl(bool distance_is_negative, Scalar s)
      {
        constexpr Scalar max = Limits<Scalar>::max;
        constexpr Scalar min = Limits<Scalar>::min;
        constexpr Scalar period = max - min;

        Scalar a = distance_is_negative ? s + period * 0.5 : s;

        if (a >= min and a < max) // Check if the angle doesn't need wrapping.
        {
          return a;
        }
        else // Wrap the angle.
        {
          Scalar ar = std::fmod(a - min, period);
          if (ar < 0) ar += period;
          return ar + min;
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of polar coordinates.
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
        const auto d = g(start + d_i);
        if (local_index == d_i)
          return std::abs(d);
        else // local_index == a_i
          return polar_angle_wrap_impl<Scalar>(std::signbit(d), g(start + a_i));
      }


      /**
       * \brief Set an element and then perform any necessary modular wrapping.
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
        if (local_index == d_i)
        {
          s(std::abs(x), start + d_i);
          s(polar_angle_wrap_impl<Scalar>(std::signbit(x), g(start + a_i)), start + a_i); //< Possibly reflect angle
        }
        else // local_index == a_i
        {
          s(polar_angle_wrap_impl<Scalar>(false, x), start + a_i);
        }
      }

    };

  } // namespace detail


  // (Radius, Angle).
  template<template<typename Scalar> typename Limits>
  struct Polar<Distance, Angle<Limits>> : detail::PolarBase<Limits, 0, 1,  0, 1, 2> {};


  // (Angle, Radius).
  template<template<typename Scalar> typename Limits>
  struct Polar<Angle<Limits>, Distance> : detail::PolarBase<Limits, 1, 0,  2, 0, 1> {};


  /**
   * \internal
   * \brief Polar is represented by two coordinates.
   */
   template<typename T1, typename T2>
   struct dimension_size_of<Polar<T1, T2>> : std::integral_constant<std::size_t, 2>
   {
     constexpr static std::size_t get(const Polar<T1, T2>&) { return 2; }
   };


  /**
   * \brief The number of atomic components.
   */
  template<typename T1, typename T2>
  struct index_descriptor_components_of<Polar<T1, T2>> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Polar<T1, T2>&) { return 1; }
  };


  /**
   * \internal
   * \brief Polar is represented by three coordinates in Euclidean space.
   */
   template<typename T1, typename T2>
   struct euclidean_dimension_size_of<Polar<T1, T2>> : std::integral_constant<std::size_t, 3>
   {
     constexpr static std::size_t get(const Polar<T1, T2>&) { return 3; }
   };


  /**
   * \internal
   * \brief The type of the result when subtracting two Polar vectors.
   * \details For differences, each coordinate behaves as if it were Distance or Angle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   */
  template<typename T1, typename T2>
  struct dimension_difference_of<Polar<T1, T2>>
  {
    using type = Concatenate<dimension_difference_of_t<T1>, dimension_difference_of_t<T2>>;
  };


}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_HPP
