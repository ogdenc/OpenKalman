/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
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

#include <type_traits>
#include <typeindex>
#include <cmath>
#include <array>
#include "../../../basics/compatibility/language-features.hpp"
#include "values/concepts/number.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/signbit.hpp"
#include "values/math/abs.hpp"
#include "values/math/cos.hpp"
#include "values/math/sin.hpp"
#include "values/math/atan2.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/functions/internal/get_hash_code.hpp"
#include "Distance.hpp"
#include "Angle.hpp"


namespace OpenKalman::coordinates
{
  /**
   * \brief An atomic \ref coordinates::descriptor reflecting polar coordinates.
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
  requires (std::same_as<C1, Distance> and angle::angle<C2>) or (std::same_as<C2, Distance> and angle::angle<C1>)
#endif
  struct Polar
  {
#ifndef __cpp_concepts
    static_assert((std::is_same_v<C1, Distance> and angle::angle<C2>) or (std::is_same_v<C2, Distance> and angle::angle<C1>));
#endif
  };

} // namespace OpenKalman::coordinates


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename Min, typename Max,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
    private:

      static constexpr auto min = values::fixed_number_of_v<Min>;
      static constexpr auto max = values::fixed_number_of_v<Max>;

    public:

      static constexpr bool is_specialized = true;


      static constexpr auto
      dimension(const T&) { return std::integral_constant<std::size_t, 2>{}; };


      static constexpr auto
      stat_dimension(const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto
      is_euclidean(const T&) { return std::false_type{}; }


      static constexpr std::size_t
      hash_code(const T&)
      {
        constexpr auto a = coordinates::internal::get_hash_code(coordinates::Angle<Min, Max>{});
        constexpr auto bits = std::numeric_limits<std::size_t>::digits;
        if constexpr (bits < 32) return a - 0x97C1_uz + d_i * 0x1000_uz;
        else if constexpr (bits < 64) return a - 0x97C195FE_uz + d_i * 0x10000000_uz;
        else return a - 0x97C195FEC488C0BC_uz + d_i * 0x1000000000000000_uz;
      }


      /**
       * \brief Maps a polar coordinate to coordinates in Euclidean space.
       * \details This function takes a set of polar coordinates and converts them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
       * \param start The starting index within the \ref coordinates::pattern object
       */
#ifdef __cpp_concepts
      static constexpr values::value auto
      to_euclidean_component(const T& t, const auto& g, const values::index auto& euclidean_local_index)
      requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
      template<typename Getter, typename L, std::enable_if_t<values::index<L> and
        values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
      static constexpr auto
      to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
      {
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        Scalar d = g(d_i);
        if (euclidean_local_index == d2_i)
        {
          return d;
        }
        else
        {
          using R = std::decay_t<decltype(values::real(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (max - min)};
          const Scalar mid {R{max + min} * R{0.5}};

          Scalar theta = cf * g(a_i) - mid;
          switch(euclidean_local_index)
          {
            case x_i: return values::cos(theta);
            default: return values::sin(theta); // case y_i
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
      static constexpr values::value auto
      from_euclidean_component(const T& t, const auto& g, const values::index auto& local_index)
      requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
      template<typename Getter, typename L, std::enable_if_t<values::index<L> and
        values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
      static constexpr auto
      from_euclidean_component(const T& t, const Getter& g, const L& local_index)
  #endif
        {
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        Scalar d = g(d2_i);
        auto dr = values::real(d);
        if (local_index == d_i)
        {
          // A negative distance is reflected to the positive axis.
          return values::internal::update_real_part(d, values::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(values::real(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (max - min)};
          const Scalar mid {R{max + min} * R{0.5}};

          // If distance is negative, flip 180 degrees:
          Scalar x = values::signbit(dr) ? -g(x_i) : g(x_i);
          Scalar y = values::signbit(dr) ? -g(y_i) : g(y_i);

          if constexpr (values::complex<Scalar>) return values::atan2(y, x) / cf + mid;
          else { return values::atan2(y, x) / cf + mid; }
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
        using R = std::decay_t<decltype(values::real(std::declval<decltype(a)>()))>;
        constexpr R period {max - min};
        R ap {distance_is_negative ? values::real(a) + period * R{0.5} : values::real(a)};

        if (ap >= min and ap < max) // Check if the angle doesn't need wrapping.
        {
          return values::internal::update_real_part(std::forward<decltype(a)>(a), ap);;
        }
        else // Wrap the angle.
        {
          using std::fmod;
          auto ar = fmod(ap - R{min}, period);
          if (ar < 0) return values::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar + period);
          else return values::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar);
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of polar coordinates.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index accessing the angle (in this case, it must be 0)
       * \param start The starting location of the angle within any larger set of \ref coordinates::pattern
       */
#ifdef __cpp_concepts
      static constexpr values::value auto
      get_wrapped_component(const T& t, const auto& g, const values::index auto& local_index)
      requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
      template<typename Getter, typename L, std::enable_if_t<values::index<L> and
        values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
      static constexpr auto
      get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
      {
        auto d = g(d_i);
        switch(local_index)
        {
          case d_i: return values::internal::update_real_part(d, values::abs(values::real(d)));
          default: return polar_angle_wrap_impl(values::signbit(values::real(d)), g(a_i)); // case a_i
        }
      }


      /**
       * \brief Set an element and then perform any necessary modular wrapping.
       * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
       * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param x The scalar value to be set.
       * \param local_index A local index accessing the angle (in this case, it must be 0)
       * \param start The starting location of the angle within any larger set of \ref coordinates::pattern
       */
#ifdef __cpp_concepts
      static constexpr void
      set_wrapped_component(const T& t, const auto& s, const auto& g, const values::value auto& x, const values::index auto& local_index)
      requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
      template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<values::value<X> and values::index<L> and
        std::is_invocable<const Setter&, const X&, std::size_t>::value and
        std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
      static constexpr void
      set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
      {
        switch(local_index)
        {
          case d_i:
          {
            auto xp = values::real(x);
            s(values::internal::update_real_part(x, values::abs(xp)), d_i);
            s(polar_angle_wrap_impl(values::signbit(xp), g(a_i)), a_i); //< Possibly reflect angle
            break;
          }
          default: // case a_i
          {
            s(polar_angle_wrap_impl(false, x), a_i);
            break;
          }
        }
      }

    };

  } // namespace detail


  /**
   * \internal
   * \brief traits for Polar<Distance, Angle>.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Polar<coordinates::Distance, coordinates::Angle<Min, Max>>>
    : detail::PolarBase<coordinates::Polar<coordinates::Distance, coordinates::Angle<Min, Max>>, Min, Max, 0, 1,  0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Polar<Angle, Distance>.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Polar<coordinates::Angle<Min, Max>, coordinates::Distance>>
    : detail::PolarBase<coordinates::Polar<coordinates::Angle<Min, Max>, coordinates::Distance>, Min, Max, 1, 0,  2, 0, 1>
  {};


}// namespace OpenKalman::interface

#endif //OPENKALMAN_POLAR_HPP
