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
   * \details C1, C2, and C3 must be some combination of Distance, Inclination, and Angle
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
  template<typename C1 = Distance, typename C2 = angle::Radians, typename C3 = inclination::Radians>
#ifdef __cpp_concepts
  requires
    (distance_descriptor<C1> and angle_descriptor<C2> and inclination_descriptor<C3>) or
    (distance_descriptor<C1> and angle_descriptor<C3> and inclination_descriptor<C2>) or
    (distance_descriptor<C2> and angle_descriptor<C1> and inclination_descriptor<C3>) or
    (distance_descriptor<C2> and angle_descriptor<C3> and inclination_descriptor<C1>) or
    (distance_descriptor<C3> and angle_descriptor<C1> and inclination_descriptor<C2>) or
    (distance_descriptor<C3> and angle_descriptor<C2> and inclination_descriptor<C1>)
#endif
  struct Spherical
  {
#ifndef __cpp_concepts
    static_assert(
      (distance_descriptor<C1> and angle_descriptor<C2> and inclination_descriptor<C3>) or
      (distance_descriptor<C1> and angle_descriptor<C3> and inclination_descriptor<C2>) or
      (distance_descriptor<C2> and angle_descriptor<C1> and inclination_descriptor<C3>) or
      (distance_descriptor<C2> and angle_descriptor<C3> and inclination_descriptor<C1>) or
      (distance_descriptor<C3> and angle_descriptor<C1> and inclination_descriptor<C2>) or
      (distance_descriptor<C3> and angle_descriptor<C2> and inclination_descriptor<C1>));
#endif
  };


  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename CircleLimits, typename InclinationLimits, std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
      static constexpr std::size_t size = 3;
      static constexpr std::size_t euclidean_size = 4;
      static constexpr std::size_t component_count = 1;
      static constexpr bool always_euclidean = false;

    private:

      static constexpr std::size_t d2_i = 0, x_i = 1, y_i = 2, z_i = 3;

    public:

      /**
       * \brief Maps an element to coordinates in Euclidean space.
       * \details This function takes a set of spherical coordinates and converts them to d, x, y, and z
       * Cartesian coordinates representing a location on a unit 4D half-cylinder.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
       * \param start The starting index within the index descriptor
       */
#ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
#else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
      {
        if (euclidean_local_index == d2_i)
        {
          return g(start + d_i);
        }
        else
        {
          using Scalar = decltype(g(std::declval<std::size_t>()));
          using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
          const Scalar cf_inc {numbers::pi_v<R> / (InclinationLimits::up - InclinationLimits::down)};
          const Scalar horiz {R{InclinationLimits::up + InclinationLimits::down} * R{0.5}};

          Scalar phi = cf_inc * (g(start + i_i) - horiz);
          if (euclidean_local_index == z_i)
          {
            return interface::ScalarTraits<Scalar>::sin(phi);
          }
          else
          {
            const Scalar cf_cir {2 * numbers::pi_v<R> / (CircleLimits::max - CircleLimits::min)};
            const Scalar mid {R{CircleLimits::max + CircleLimits::min} * R{0.5}};
            Scalar theta = cf_cir * (g(start + a_i) - mid);
            if (euclidean_local_index == x_i)
              return interface::ScalarTraits<Scalar>::cos(theta) * interface::ScalarTraits<Scalar>::cos(phi);
            else // euclidean_local_index == y_i
              return interface::ScalarTraits<Scalar>::sin(theta) * interface::ScalarTraits<Scalar>::cos(phi);
          }
        }
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes d, x, y, and z Cartesian coordinates representing a location on a
       * 4D unit half-cylinder, and converts them to spherical coordinates.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting index within the Euclidean-transformed indices
       */
#ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
#else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
      {
        using Scalar = decltype(g(std::declval<std::size_t>()));
        Scalar d = g(euclidean_start + d2_i);
        auto dr = real_projection(d);

        if (local_index == d_i)
        {
          return inverse_real_projection(d, std::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
          const Scalar cf_cir {2 * numbers::pi_v<R> / (CircleLimits::max - CircleLimits::min)};
          const Scalar mid {R{CircleLimits::max + CircleLimits::min} * R{0.5}};

          Scalar x = g(euclidean_start + x_i);
          Scalar y = g(euclidean_start + y_i);

          switch(local_index)
          {
            case a_i:
            {
              auto xp = real_projection(g(euclidean_start + x_i));
              auto yp = real_projection(g(euclidean_start + y_i));
              // If distance is negative, flip x and y axes 180 degrees:
              Scalar x2 = inverse_real_projection(x, std::signbit(dr) ? -xp : xp);
              Scalar y2 = inverse_real_projection(y, std::signbit(dr) ? -yp : yp);
              return interface::ScalarTraits<Scalar>::atan2(y2, x2) / cf_cir + mid;
            }
            default: // case i_i
            {
              const Scalar cf_inc {numbers::pi_v<R> / (InclinationLimits::up - InclinationLimits::down)};
              const Scalar horiz {R{InclinationLimits::up + InclinationLimits::down} * R{0.5}};
              Scalar z {g(euclidean_start + z_i)};
              auto zp = real_projection(z);
              Scalar z2 {inverse_real_projection(z, std::signbit(dr) ? -zp : zp)};
              Scalar r {interface::ScalarTraits<Scalar>::sqrt(x*x + y*y + z2*z2)};
              return interface::ScalarTraits<Scalar>::asin2(z2, r) / cf_inc + horiz;
            }
          }
        }

      }

    private:

      template<typename Scalar>
      static constexpr auto
      inclination_wrap_impl(const Scalar& a) -> std::tuple<std::decay_t<std::decay_t<decltype(real_projection(a))>>, bool>
      {
        auto ap = real_projection(a);
        using R = std::decay_t<decltype(ap)>;
        if (ap >= InclinationLimits::down and ap <= InclinationLimits::up) // A shortcut, for the easy case.
        {
          return { ap, false };
        }
        else
        {
          constexpr R period = 2 * (InclinationLimits::up - InclinationLimits::down);
          R ar = std::fmod(ap - R{InclinationLimits::down}, period);
          R ar2 = ar < 0 ? ar + period : ar;
          bool b = ar2 > InclinationLimits::up - InclinationLimits::down; // Whether there is a mirror reflection about vertical axis.
          return { R{InclinationLimits::down} + (b ? period - ar2 : ar2), b };
        }
      }


      template<typename Scalar>
      static constexpr std::decay_t<Scalar>
      azimuth_wrap_impl(bool reflect_azimuth, Scalar&& a)
      {
        using R = std::decay_t<decltype(real_projection(std::declval<decltype(a)>()))>;
        constexpr R period {CircleLimits::max - CircleLimits::min};
        constexpr R half_period {(CircleLimits::max - CircleLimits::min) / R{2}};
        R ap = reflect_azimuth ? real_projection(a) - half_period : real_projection(a);

        if (ap >= CircleLimits::min and ap < CircleLimits::max) // Check if angle doesn't need wrapping.
        {
          return inverse_real_projection(std::forward<decltype(a)>(a), ap);;
        }
        else // Wrap the angle.
        {
          auto ar = std::fmod(ap - R{CircleLimits::min}, period);
          if (ar < 0) return inverse_real_projection(std::forward<decltype(a)>(a), R{CircleLimits::min} + ar + period);
          else return inverse_real_projection(std::forward<decltype(a)>(a), R{CircleLimits::min} + ar);
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of spherical coordinates.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index accessing the angle (in this case, it must be 0)
       * \param start The starting location of the angle within any larger set of index type descriptors
       */
#ifdef __cpp_concepts
      static constexpr floating_scalar_type auto
      wrap_get_element(const auto& g, std::size_t local_index, std::size_t start)
      requires requires (std::size_t i){ {g(i)} -> floating_scalar_type; }
#else
      template<typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      wrap_get_element(const G& g, std::size_t local_index, std::size_t start)
#endif
      {
        auto d = g(start + d_i);
        auto dp = real_projection(d);

        switch(local_index)
        {
          case d_i:
          {
            return inverse_real_projection(d, std::abs(dp));
          }
          case a_i:
          {
            const bool b = std::get<1>(inclination_wrap_impl(g(start + i_i)));
            return azimuth_wrap_impl(b != std::signbit(dp), g(start + a_i));
          }
          default: // case i_i
          {
            auto i = g(start + i_i);
            auto new_i = std::get<0>(inclination_wrap_impl(i));
            return inverse_real_projection(i, std::signbit(dp) ? -new_i : new_i);
          }
        }
      }


      /**
       * \brief Set an element and then perform any necessary modular wrapping.
       * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
       * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param x The scalar value to be set.
       * \param local_index A local index accessing the angle (in this case, it must be 0)
       * \param start The starting location of the angle within any larger set of index type descriptors
       */
#ifdef __cpp_concepts
      static constexpr void
      wrap_set_element(const auto& s, const auto& g,
        const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
      requires requires (std::size_t i){ s(x, i); {x} -> floating_scalar_type; }
#else
      template<typename S, typename G, std::enable_if_t<floating_scalar_type<typename std::invoke_result<G, std::size_t>::type> and
        std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
      static constexpr void
      wrap_set_element(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
                       std::size_t local_index, std::size_t start)
#endif
      {
        switch(local_index)
        {
          case d_i:
          {
            auto dp = real_projection(x);
            s(inverse_real_projection(x, std::abs(dp)), start + d_i);
            if (std::signbit(dp)) // If new distance would have been negative
            {
              auto azimuth_i = start + a_i;
              auto inclination = start + i_i;
              s(azimuth_wrap_impl(true, g(azimuth_i)), azimuth_i); // Reflect azimuth.
              s(-g(inclination), inclination); // Reflect inclination.
            }
            break;
          }
          case a_i:
          {
            s(azimuth_wrap_impl(false, x), start + a_i);
            break;
          }
          default: // case i_i
          {
            const auto [ip, b] = inclination_wrap_impl(x);
            s(inverse_real_projection(x, ip), start + i_i); // Reflect inclination.
            const auto azimuth_i = start + a_i;
            s(azimuth_wrap_impl(b, g(azimuth_i)), azimuth_i); // Maybe reflect azimuth.
            break;
          }
        }
      }

    };

  } // namespace detail


  namespace interface
  {
    /**
     * \internal
     * \brief traits for Spherical<Distance, Angle, Inclination>.
     */
    template<typename ALimits, typename ILimits>
    struct FixedIndexDescriptorTraits<Spherical<Distance, Angle<ALimits>, Inclination<ILimits>>>
      : detail::SphericalBase<ALimits, ILimits, 0, 1, 2>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Distance>::difference_type,
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Spherical<Distance, Inclination, Angle>.
     */
    template<typename ILimits, typename ALimits>
    struct FixedIndexDescriptorTraits<Spherical<Distance, Inclination<ILimits>, Angle<ALimits>>>
      : detail::SphericalBase<ALimits, ILimits, 0, 2, 1>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Distance>::difference_type,
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Spherical<Angle, Distance, Inclination>.
     */
    template<typename ALimits, typename ILimits>
    struct FixedIndexDescriptorTraits<Spherical<Angle<ALimits>, Distance, Inclination<ILimits>>>
      : detail::SphericalBase<ALimits, ILimits, 1, 0, 2>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Distance>::difference_type,
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Spherical<Inclination, Distance, Angle>.
     */
    template<typename ILimits, typename ALimits>
    struct FixedIndexDescriptorTraits<Spherical<Inclination<ILimits>, Distance, Angle<ALimits>>>
      : detail::SphericalBase<ALimits, ILimits, 1, 2, 0>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Distance>::difference_type,
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Spherical<Angle, Inclination, Distance>.
     */
    template<typename ALimits, typename ILimits>
    struct FixedIndexDescriptorTraits<Spherical<Angle<ALimits>, Inclination<ILimits>, Distance>>
      : detail::SphericalBase<ALimits, ILimits, 2, 0, 1>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Distance>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Spherical<Inclination, Angle, Distance>.
     */
    template<typename ILimits, typename ALimits>
    struct FixedIndexDescriptorTraits<Spherical<Inclination<ILimits>, Angle<ALimits>, Distance>>
      : detail::SphericalBase<ALimits, ILimits, 2, 1, 0>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Inclination<ILimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Angle<ALimits>>::difference_type,
        typename FixedIndexDescriptorTraits<Distance>::difference_type>;
    };


  } // namespace interface

}// namespace OpenKalman

#endif //OPENKALMAN_SPHERICAL_HPP
