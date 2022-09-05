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
  template<typename C1 = Distance, typename C2 = angle::Radians>
#ifdef __cpp_concepts
  requires (distance_descriptor<C1> and angle_descriptor<C2>) or (distance_descriptor<C2> and angle_descriptor<C1>)
#endif
  struct Polar
  {
#ifndef __cpp_concepts
    static_assert((distance_descriptor<C1> and angle_descriptor<C2>) or (distance_descriptor<C2> and angle_descriptor<C1>));
#endif
  };


  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename Limits,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
      static constexpr std::size_t size = 2;
      static constexpr std::size_t euclidean_size = 3;
      static constexpr std::size_t component_count = 1;
      static constexpr bool always_euclidean = false;


      /**
       * \brief Maps a polar coordinate to coordinates in Euclidean space.
       * \details This function takes a set of polar coordinates and converts them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
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
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        Scalar d = g(start + d_i);
        if (euclidean_local_index == d2_i)
        {
          return d;
        }
        else
        {
          using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
          const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

          Scalar theta = cf * g(start + a_i) - mid;
          switch(euclidean_local_index)
          {
            case x_i: return interface::ScalarTraits<Scalar>::cos(theta);
            default: return interface::ScalarTraits<Scalar>::sin(theta); // case y_i
          }
        }
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes x, y, and z Cartesian coordinates representing a location on a
       * unit half-cylinder, and converts them to polar coordinates.
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
          // A negative distance is reflected to the positive axis.
          return inverse_real_projection(d, std::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
          const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

          // If distance is negative, flip 180 degrees:
          Scalar x = std::signbit(dr) ? -g(euclidean_start + x_i) : g(euclidean_start + x_i);
          Scalar y = std::signbit(dr) ? -g(euclidean_start + y_i) : g(euclidean_start + y_i);
          return interface::ScalarTraits<Scalar>::atan2(y, x) / cf + mid;
        }
      }

    private:

#ifdef __cpp_concepts
      static constexpr auto polar_angle_wrap_impl(bool distance_is_negative, auto&& a) -> std::decay_t<decltype(a)>
#else
      template<typename Scalar>
      static constexpr std::decay_t<Scalar> polar_angle_wrap_impl(bool distance_is_negative, Scalar&& a)
#endif
      {
        using R = std::decay_t<decltype(real_projection(std::declval<decltype(a)>()))>;
        constexpr R period {Limits::max - Limits::min};
        R ap {distance_is_negative ? real_projection(a) + period * R{0.5} : real_projection(a)};

        if (ap >= Limits::min and ap < Limits::max) // Check if the angle doesn't need wrapping.
        {
          return inverse_real_projection(std::forward<decltype(a)>(a), ap);;
        }
        else // Wrap the angle.
        {
          auto ar = std::fmod(ap - R{Limits::min}, period);
          if (ar < 0) return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::min} + ar + period);
          else return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::min} + ar);
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of polar coordinates.
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
        switch(local_index)
        {
          case d_i: return inverse_real_projection(d, std::abs(real_projection(d)));
          default: return polar_angle_wrap_impl(std::signbit(real_projection(d)), g(start + a_i)); // case a_i
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
            auto xp = real_projection(x);
            s(inverse_real_projection(x, std::abs(xp)), start + d_i);
            s(polar_angle_wrap_impl(std::signbit(xp), g(start + a_i)), start + a_i); //< Possibly reflect angle
            break;
          }
          default: // case a_i
          {
            s(polar_angle_wrap_impl(false, x), start + a_i);
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
     * \brief traits for Polar<Distance, Angle>.
     */
    template<typename Limits>
    struct FixedIndexDescriptorTraits<Polar<Distance, Angle<Limits>>> : detail::PolarBase<Limits, 0, 1,  0, 1, 2>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Distance>::difference_type,
        typename FixedIndexDescriptorTraits<Angle<Limits>>::difference_type>;
    };


    /**
     * \internal
     * \brief traits for Polar<Angle, Distance>.
     */
    template<typename Limits>
    struct FixedIndexDescriptorTraits<Polar<Angle<Limits>, Distance>> : detail::PolarBase<Limits, 1, 0,  2, 0, 1>
    {
      using difference_type = concatenate_fixed_index_descriptor_t<
        typename FixedIndexDescriptorTraits<Angle<Limits>>::difference_type,
        typename FixedIndexDescriptorTraits<Distance>::difference_type>;
    };

  } // namespace interface


}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_HPP
