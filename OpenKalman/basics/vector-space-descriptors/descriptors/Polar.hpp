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


namespace OpenKalman::vector_space_descriptors
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
  requires (distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C2>) or (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C1>)
#endif
  struct Polar
  {
#ifndef __cpp_concepts
    static_assert((distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C2>) or (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C1>));
#endif

    /// Default constructor
    constexpr Polar() = default;


    /// Conversion constructor
#ifdef __cpp_concepts
    template<maybe_equivalent_to<Polar> D> requires (not std::same_as<std::decay_t<D>, Polar>)
#else
    template<typename D, std::enable_if_t<
      maybe_equivalent_to<D, Polar> and not std::is_same_v<std::decay_t<D>, Polar>, int> = 0>
#endif
    explicit constexpr Polar(D&& d)
    {
      if constexpr (dynamic_vector_space_descriptor<D>)
      {
        if (d != Polar{}) throw std::invalid_argument{"Dynamic argument of 'Polar' constructor is not a polar vector space descriptor."};
      }
    }

  };

} // namespace OpenKalman::vector_space_descriptors


namespace OpenKalman::detail
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
     * \param start The starting index within the \ref vector_space_descriptor object
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
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
        using R = std::decay_t<decltype(internal::constexpr_real(std::declval<Scalar>()))>;
        const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
        const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

        Scalar theta = cf * g(start + a_i) - mid;
        switch(euclidean_local_index)
        {
          using std::cos, std::sin;
          case x_i: return cos(theta);
          default: return sin(theta); // case y_i
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
    static constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
      Scalar d = g(euclidean_start + d2_i);
      auto dr = internal::constexpr_real(d);
      if (local_index == d_i)
      {
        // A negative distance is reflected to the positive axis.
        using std::abs;
        return internal::update_real_part(d, abs(dr));
      }
      else
      {
        using R = std::decay_t<decltype(internal::constexpr_real(std::declval<Scalar>()))>;
        const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
        const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

        // If distance is negative, flip 180 degrees:
        using std::signbit;
        Scalar x = signbit(dr) ? -g(euclidean_start + x_i) : g(euclidean_start + x_i);
        Scalar y = signbit(dr) ? -g(euclidean_start + y_i) : g(euclidean_start + y_i);

        if constexpr (complex_number<Scalar>) return internal::constexpr_atan2(y, x) / cf + mid;
        else { using std::atan2; return atan2(y, x) / cf + mid; }
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
      using R = std::decay_t<decltype(internal::constexpr_real(std::declval<decltype(a)>()))>;
      constexpr R period {Limits::max - Limits::min};
      R ap {distance_is_negative ? internal::constexpr_real(a) + period * R{0.5} : internal::constexpr_real(a)};

      if (ap >= Limits::min and ap < Limits::max) // Check if the angle doesn't need wrapping.
      {
        return internal::update_real_part(std::forward<decltype(a)>(a), ap);;
      }
      else // Wrap the angle.
      {
        using std::fmod;
        auto ar = fmod(ap - R{Limits::min}, period);
        if (ar < 0) return internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::min} + ar + period);
        else return internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::min} + ar);
      }
    }

  public:

    /**
     * \brief Perform modular wrapping of polar coordinates.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      using std::abs, std::signbit;
      auto d = g(start + d_i);
      switch(local_index)
      {
        case d_i: return internal::update_real_part(d, abs(internal::constexpr_real(d)));
        default: return polar_angle_wrap_impl(signbit(internal::constexpr_real(d)), g(start + a_i)); // case a_i
      }
    }


    /**
     * \brief Set an element and then perform any necessary modular wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param x The scalar value to be set.
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      switch(local_index)
      {
        case d_i:
        {
          auto xp = internal::constexpr_real(x);
          using std::abs, std::signbit;
          s(internal::update_real_part(x, abs(xp)), start + d_i);
          s(polar_angle_wrap_impl(signbit(xp), g(start + a_i)), start + a_i); //< Possibly reflect angle
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

} // namespace OpenKalman::detail


namespace OpenKalman::vector_space_descriptors
{
  /**
   * \internal
   * \brief traits for Polar<Distance, Angle>.
   */
  template<typename Limits>
  struct fixed_vector_space_descriptor_traits<Polar<Distance, Angle<Limits>>> : detail::PolarBase<Limits, 0, 1,  0, 1, 2>
  {
    using difference_type = concatenate_fixed_vector_space_descriptor_t<
      typename fixed_vector_space_descriptor_traits<Distance>::difference_type,
      typename fixed_vector_space_descriptor_traits<Angle<Limits>>::difference_type>;
  };


  /**
   * \internal
   * \brief traits for Polar<Angle, Distance>.
   */
  template<typename Limits>
  struct fixed_vector_space_descriptor_traits<Polar<Angle<Limits>, Distance>> : detail::PolarBase<Limits, 1, 0,  2, 0, 1>
  {
    using difference_type = concatenate_fixed_vector_space_descriptor_t<
      typename fixed_vector_space_descriptor_traits<Angle<Limits>>::difference_type,
      typename fixed_vector_space_descriptor_traits<Distance>::difference_type>;
  };


}// namespace OpenKalman::vector_space_descriptors

#endif //OPENKALMAN_POLAR_HPP
