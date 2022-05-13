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
 * \brief Definition of Spherical class and associated details.
 */

#ifndef OPENKALMAN_SPHERICAL_HPP
#define OPENKALMAN_SPHERICAL_HPP


namespace OpenKalman
{
  /**
   * \brief An atomic coefficient group reflecting spherical coordinates.
   * \details Coefficient1, Coefficient2, and Coefficient3 must be some combination of Distance, Inclination, and Angle
   * in any order, reflecting the distance, inclination, and azimuth, respectively.
   * Spherical coordinates span three adjacent coefficients in a matrix.<br/>
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \par Examples
   * <code>Spherical&lt;Distance, inclination::Degrees, angle::Radians&gt;,<br/>
   * Spherical&lt;angle::PositiveDegrees, Distance, inclination::Radians&gt;</code>
   * \tparam C1, C2, C3 Distance, inclination, and Angle, in any order.
   * By default, they are Distance, angle::Radians, and inclination::Radians, respectively.
   */
#ifdef __cpp_concepts
  template<atomic_fixed_index_descriptor C1 = Distance, atomic_fixed_index_descriptor C2 = angle::Radians,
    atomic_fixed_index_descriptor C3 = inclination::Radians>
#else
  template<typename C1 = Distance, typename C2 = angle::Radians, typename C3 = inclination::Radians, typename = void>
#endif
  struct Spherical;


  namespace detail
  {
    // Implementation of polar coordinates.
    template<template<typename Scalar> typename CircleLimits, template<typename Scalar> typename InclinationLimits,
      std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
    private:

      static constexpr std::size_t d2_i = 0, x_i = 1, y_i = 2, z_i = 3;

      template<typename Scalar>
      static constexpr Scalar cf_cir = 2 * std::numbers::pi_v<Scalar> /
        (CircleLimits<Scalar>::max - CircleLimits<Scalar>::min);

      template<typename Scalar>
      static constexpr Scalar mid = (CircleLimits<Scalar>::max + CircleLimits<Scalar>::min) / 2;

      template<typename Scalar>
      static constexpr Scalar cf_inc = std::numbers::pi_v<Scalar> /
        (InclinationLimits<Scalar>::up - InclinationLimits<Scalar>::down);

      template<typename Scalar>
      static constexpr Scalar horiz = (InclinationLimits<Scalar>::up + InclinationLimits<Scalar>::down) / 2;

    public:

      /**
       * \brief Maps an element to coordinates in Euclidean space.
       * \details This function takes a set of spherical coordinates and converts them to d, x, y, and z
       * Cartesian coordinates representing a location on a unit 4D half-cylinder.
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
        switch(euclidean_local_index)
        {
          case x_i:
          {
            const auto theta = cf_cir<Scalar> * (g(start + a_i) - mid<Scalar>);
            const auto phi = cf_inc<Scalar> * (g(start + i_i) - horiz<Scalar>);
            return std::cos(theta) * std::cos(phi);
          }
          case y_i:
          {
            const auto theta = cf_cir<Scalar> * (g(start + a_i) - mid<Scalar>);
            const auto phi = cf_inc<Scalar> * (g(start + i_i) - horiz<Scalar>);
            return std::sin(theta) * std::cos(phi);
          }
          case z_i:
          {
            const auto phi = cf_inc<Scalar> * (g(start + i_i) - horiz<Scalar>);
            return std::sin(phi);
          }
          default:
          {
            return g(start + d_i); // case d2_i
          }
        }
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes d, x, y, and z Cartesian coordinates representing a location on a
       * 4D unit half-cylinder, and converts them to spherical coordinates.
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
        switch(local_index)
        {
          case a_i:
          {
            // If distance is negative, flip x and y axes 180 degrees:
            const auto x = std::signbit(d) ? -g(euclidean_start + x_i) : g(euclidean_start + x_i);
            const auto y = std::signbit(d) ? -g(euclidean_start + y_i) : g(euclidean_start + y_i);
            if constexpr (not std::numeric_limits<Scalar>::is_iec559)
              if (x == 0) return (y == 0) ? mid<Scalar> : std::signbit(y) ? CircleLimits<Scalar>::down : CircleLimits<Scalar>::up;
            return std::atan2(y, x) / cf_cir<Scalar> + mid<Scalar>;
          }
          case i_i:
          {
            const auto z = g(euclidean_start + z_i);
            const auto r = std::hypot(g(euclidean_start + x_i), g(euclidean_start + y_i), z);
            if constexpr (std::numeric_limits<Scalar>::is_iec559)
            {
              const auto phi = std::asin(z / r);
              if (std::isnan(phi)) return horiz<Scalar>; // A NaN result is converted to the horizontal inclination.
              else return (std::signbit(d) ? -phi : phi) / cf_inc<Scalar> + horiz<Scalar>;
            }
            else
            {
              if (r == 0 or z > r or z < -r) return horiz<Scalar>;
              else return std::asin((std::signbit(d) ? -z : z) / r) / cf_inc<Scalar> + horiz<Scalar>;
            }
          }
          default: // case d_i
          {
            return std::abs(d);
          }
        }
      }

    private:

      template<typename Scalar>
      static constexpr std::tuple<Scalar, bool>
      inclination_wrap_impl(Scalar a)
      {
        constexpr Scalar max = InclinationLimits<Scalar>::up;
        constexpr Scalar min = InclinationLimits<Scalar>::down;
        constexpr Scalar range = max - min;
        constexpr Scalar period = 2 * range;
        if (a >= min and a <= max) // A shortcut, for the easy case.
        {
          return { a, false };
        }
        else
        {
          Scalar ar = std::fmod(a - min, period);
          if (ar < 0) ar += period;
          if (ar > range)
          {
            // Do a mirror reflection about vertical axis.
            return { period + min - ar, true };
          }
          else
          {
            return { min + ar, false };
          }
        }
      }


      template<typename Scalar>
      static constexpr Scalar
      azimuth_wrap_impl(bool reflect_azimuth, Scalar s)
      {
        constexpr Scalar max = CircleLimits<Scalar>::max;
        constexpr Scalar min = CircleLimits<Scalar>::min;
        constexpr Scalar period = max - min;

        Scalar a = reflect_azimuth ? s + period * 0.5 : s;

        if (a >= min and a < max) // Check if angle doesn't need wrapping.
        {
          return a;
        }
        else // Wrap the angle.
        {
          Scalar ar = std::fmod(a - min, period);
          if (ar < 0)
          {
            ar += period;
          }
          return ar + min;
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of spherical coordinates.
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
        switch(local_index)
        {
          case a_i:
          {
            const auto b = std::get<1>(inclination_wrap_impl<Scalar>(g(start + i_i)));
            const bool reflect_azimuth = b != std::signbit(d);
            return azimuth_wrap_impl<Scalar>(reflect_azimuth, g(start + a_i));
          }
          case i_i:
          {
            const auto new_i = std::get<0>(inclination_wrap_impl<Scalar>(g(start + i_i)));
            return std::signbit(d) ? -new_i : new_i;
          }
          default: // case d_i
          {
            return std::abs(d);
          }
        }
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
        switch(local_index)
        {
          case a_i:
          {
            s(azimuth_wrap_impl<Scalar>(false, x), start + a_i);
            break;
          }
          case i_i:
          {
            const auto [new_x, b] = inclination_wrap_impl<Scalar>(x);
            s(new_x, start + i_i); // Adjust inclination.
            const auto i_angle = start + a_i;
            s(azimuth_wrap_impl<Scalar>(b, g(i_angle)), i_angle); // Adjust azimuth.
            break;
          }
          default: // case d_i
          {
            s(std::abs(x), start + d_i);
            if (std::signbit(x)) // If new distance is negative
            {
              const auto i_angle = start + a_i;
              const auto i_inclination = start + i_i;
              s(azimuth_wrap_impl<Scalar>(true, g(i_angle)), i_angle); // Adjust azimuth.
              s(-g(i_inclination), i_inclination); // Adjust inclination.
            }
            break;
          }
        }
      }

    };

  } // namespace detail


  // Distance, Angle, Inclination.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Distance, Angle<ALimits>, Inclination<ILimits>> : detail::SphericalBase<ALimits, ILimits, 0, 1, 2> {};


  // Distance, Inclination, Angle.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Distance, Inclination<ILimits>, Angle<ALimits>> : detail::SphericalBase<ALimits, ILimits, 0, 2, 1> {};


  // Angle, Distance, Inclination.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Angle<ALimits>, Distance, Inclination<ILimits>> : detail::SphericalBase<ALimits, ILimits, 1, 0, 2> {};


  // Inclination, Distance, Angle.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Inclination<ILimits>, Distance, Angle<ALimits>> : detail::SphericalBase<ALimits, ILimits, 1, 2, 0> {};


  // Angle, Inclination, Distance.
  template<template<typename Scalar> typename ALimits, template<typename Scalar> typename ILimits>
  struct Spherical<Angle<ALimits>, Inclination<ILimits>, Distance> : detail::SphericalBase<ALimits, ILimits, 2, 0, 1> {};


  // Inclination, Angle, Distance.
  template<template<typename Scalar> typename ILimits, template<typename Scalar> typename ALimits>
  struct Spherical<Inclination<ILimits>, Angle<ALimits>, Distance> : detail::SphericalBase<ALimits, ILimits, 2, 1, 0> {};


  /**
   * \internal
   * \brief Spherical is represented by three coordinates.
   */
   template<typename T1, typename T2, typename T3>
   struct dimension_size_of<Spherical<T1, T2, T3>> : std::integral_constant<std::size_t, 3>
   {
     constexpr static std::size_t get(const Spherical<T1, T2, T3>& t) { return 3; }
   };


  /**
   * \internal
   * \brief Spherical is represented by four coordinates in Euclidean space.
   */
   template<typename T1, typename T2, typename T3>
   struct euclidean_dimension_size_of<Spherical<T1, T2, T3>> : std::integral_constant<std::size_t, 4>
   {
     constexpr static std::size_t get(const Spherical<T1, T2, T3>& t) { return 4; }
   };


  /**
   * \brief The number of atomic components.
   */
  template<typename T1, typename T2, typename T3>
  struct index_descriptor_components_of<Spherical<T1, T2, T3>> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Spherical<T1, T2, T3>&) { return 1; }
  };


  /**
   * \internal
   * \brief The type of the result when subtracting two Spherical vectors.
   * \details For differences, each coordinate behaves as if it were Distance, Angle, or Inclination.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1555 (2015).
   */
  template<typename T1, typename T2, typename T3>
  struct dimension_difference_of<Spherical<T1, T2, T3>>
  {
    using type = Concatenate<dimension_difference_of_t<T1>, dimension_difference_of_t<T2>, dimension_difference_of_t<T3>>;
  };

}// namespace OpenKalman

#endif //OPENKALMAN_SPHERICAL_HPP
