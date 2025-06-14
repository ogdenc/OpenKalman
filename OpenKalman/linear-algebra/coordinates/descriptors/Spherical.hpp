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

#include <type_traits>
#include <typeinfo>
#include <cmath>
#include <array>
#include "basics/compatibility/language-features.hpp"
#include "values/concepts/number.hpp"
#include "values/math/real.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/signbit.hpp"
#include "values/math/sqrt.hpp"
#include "values/math/abs.hpp"
#include "values/math/sin.hpp"
#include "values/math/cos.hpp"
#include "values/math/asin.hpp"
#include "values/math/atan2.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/functions/internal/get_hash_code.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"


namespace OpenKalman::coordinates
{
  /**
   * \brief A \ref coordinates::descriptor reflecting spherical coordinates.
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
    (std::same_as<C1, Distance> and angle::angle<C2> and inclination::inclination<C3>) or
    (std::same_as<C1, Distance> and angle::angle<C3> and inclination::inclination<C2>) or
    (std::same_as<C2, Distance> and angle::angle<C1> and inclination::inclination<C3>) or
    (std::same_as<C2, Distance> and angle::angle<C3> and inclination::inclination<C1>) or
    (std::same_as<C3, Distance> and angle::angle<C1> and inclination::inclination<C2>) or
    (std::same_as<C3, Distance> and angle::angle<C2> and inclination::inclination<C1>)
#endif
  struct Spherical
  {
#ifndef __cpp_concepts
    static_assert(
      (std::is_same_v<C1, Distance> and angle::angle<C2> and inclination::inclination<C3>) or
      (std::is_same_v<C1, Distance> and angle::angle<C3> and inclination::inclination<C2>) or
      (std::is_same_v<C2, Distance> and angle::angle<C1> and inclination::inclination<C3>) or
      (std::is_same_v<C2, Distance> and angle::angle<C3> and inclination::inclination<C1>) or
      (std::is_same_v<C3, Distance> and angle::angle<C1> and inclination::inclination<C2>) or
      (std::is_same_v<C3, Distance> and angle::angle<C2> and inclination::inclination<C1>));
#endif
  };

} // namespace OpenKalman::coordinates


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename Min, typename Max, typename Down, typename Up, std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
    private:

      static constexpr auto min = values::fixed_number_of_v<Min>;
      static constexpr auto max = values::fixed_number_of_v<Max>;
      static constexpr auto down = values::fixed_number_of_v<Down>;
      static constexpr auto up = values::fixed_number_of_v<Up>;

    public:

      static constexpr bool is_specialized = true;


      static constexpr auto
      dimension(const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto
      stat_dimension(const T&) { return std::integral_constant<std::size_t, 4>{}; };


      static constexpr auto
      is_euclidean(const T&) { return std::false_type{}; }


      static constexpr std::size_t
      hash_code(const T&)
      {
        constexpr auto a = coordinates::internal::get_hash_code(coordinates::Angle<Min, Max>{});
        constexpr auto b = coordinates::internal::get_hash_code(coordinates::Inclination<Down, Up>{});
        constexpr std::size_t f = (a_i * 2 + i_i * 3 - 2);
        constexpr auto bits = std::numeric_limits<std::size_t>::digits;
        if constexpr (bits < 32) return (a - 0x83B0_uz) + (b - 0x83B0_uz) + f * 0x1000_uz;
        else if constexpr (bits < 64) return (a - 0x83B023AB_uz) + (b - 0x83B023AB_uz) + f * 0x10000000_uz;
        else return (a - 0x83B023AB3EEFB99A_uz) + (b - 0x83B023AB3EEFB99A_uz) + f * 0x1000000000000000_uz;
      }

    private:

      static constexpr std::size_t d2_i = 0, x_i = 1, y_i = 2, z_i = 3;

    public:

      /**
       * \brief Maps an element to coordinates in Euclidean space.
       * \details This function takes a set of spherical coordinates and converts them to d, x, y, and z
       * Cartesian coordinates representing a location on a unit 4D half-cylinder.
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
        if (euclidean_local_index == d2_i)
        {
          return g(d_i);
        }
        else
        {
          using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
          using R = std::decay_t<decltype(values::real(std::declval<Scalar>()))>;
          const Scalar cf_inc {numbers::pi_v<R> / (up - down)};
          const Scalar horiz {R{up + down} * R{0.5}};

          Scalar phi = cf_inc * (g(i_i) - horiz);
          if (euclidean_local_index == z_i)
          {
            return values::sin(phi);
          }
          else
          {
            const Scalar cf_cir {2 * numbers::pi_v<R> / (max - min)};
            const Scalar mid {R{max + min} * R{0.5}};
            Scalar theta = cf_cir * (g(a_i) - mid);

            if (euclidean_local_index == x_i) return values::cos(theta) * values::cos(phi);
            else return values::sin(theta) * values::cos(phi); // euclidean_local_index == y_i
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
        using Scalar = decltype(g(std::declval<std::size_t>()));
        Scalar d = g(d2_i);
        auto dr = values::real(d);

        if (local_index == d_i)
        {
          return values::internal::update_real_part(d, values::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(values::real(std::declval<Scalar>()))>;
          const Scalar cf_cir {2 * numbers::pi_v<R> / (max - min)};
          const Scalar mid {R{max + min} * R{0.5}};

          Scalar x = g(x_i);
          Scalar y = g(y_i);

          switch(local_index)
          {
            case a_i:
            {
              auto xp = values::real(g(x_i));
              auto yp = values::real(g(y_i));
              // If distance is negative, flip x and y axes 180 degrees:
              Scalar x2 = values::internal::update_real_part(x, values::signbit(dr) ? -xp : xp);
              Scalar y2 = values::internal::update_real_part(y, values::signbit(dr) ? -yp : yp);

              if constexpr (values::complex<Scalar>) return values::atan2(y2, x2) / cf_cir + mid;
              else { return values::atan2(y2, x2) / cf_cir + mid; }
            }
            default: // case i_i
            {
              const Scalar cf_inc {numbers::pi_v<R> / (up - down)};
              const Scalar horiz {R{up + down} * R{0.5}};
              Scalar z {g(z_i)};
              auto zp = values::real(z);
              Scalar z2 {values::internal::update_real_part(z, values::signbit(dr) ? -zp : zp)};
              Scalar r {values::sqrt(x*x + y*y + z2*z2)};
              if constexpr (values::complex<Scalar>)
              {
                auto theta = (r == Scalar{0}) ? r : values::asin(z2/r);
                return theta / cf_inc + horiz;
              }
              else
              {
                using R = std::decay_t<decltype(values::asin(z2/r))>;
                // This is so that a zero-radius or faulty spherical coordinate has horizontal inclination:
                auto theta = (r == 0 or r < z2 or z2 < -r) ? R{0} : values::asin(z2/r);
                return theta / cf_inc + horiz;
              }

            }
          }
        }

      }

    private:

      template<typename Scalar>
      static constexpr auto
      inclination_wrap_impl(const Scalar& a)
      {
        using Ret = std::tuple<std::decay_t<std::decay_t<decltype(values::real(a))>>, bool>;
        auto ap = values::real(a);
        using R = std::decay_t<decltype(ap)>;
        if (ap >= down and ap <= up) // A shortcut, for the easy case.
        {
          return Ret { ap, false };
        }
        else
        {
          constexpr R period = 2 * (up - down);
          using std::fmod;
          R ar = fmod(ap - R{down}, period);
          R ar2 = ar < 0 ? ar + period : ar;
          bool b = ar2 > up - down; // Whether there is a mirror reflection about vertical axis.
          return Ret { R{down} + (b ? period - ar2 : ar2), b };
        }
      }


      template<typename Scalar>
      static constexpr std::decay_t<Scalar>
      azimuth_wrap_impl(bool reflect_azimuth, Scalar&& a)
      {
        using R = std::decay_t<decltype(values::real(std::declval<decltype(a)>()))>;
        constexpr R period {max - min};
        constexpr R half_period {(max - min) / R{2}};
        R ap = reflect_azimuth ? values::real(a) - half_period : values::real(a);

        if (ap >= min and ap < max) // Check if angle doesn't need wrapping.
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
       * \brief Perform modular wrapping of spherical coordinates.
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
        auto dp = values::real(d);

        switch(local_index)
        {
          case d_i:
          {
            return values::internal::update_real_part(d, values::abs(dp));
          }
          case a_i:
          {
            const bool b = std::get<1>(inclination_wrap_impl(g(i_i)));
            return azimuth_wrap_impl(b != values::signbit(dp), g(a_i));
          }
          default: // case i_i
          {
            auto i = g(i_i);
            auto new_i = std::get<0>(inclination_wrap_impl(i));
            return values::internal::update_real_part(i, values::signbit(dp) ? -new_i : new_i);
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
            auto dp = values::real(x);
            s(values::internal::update_real_part(x, values::abs(dp)), d_i);
            if (values::signbit(dp)) // If new distance would have been negative
            {
              auto azimuth_i = a_i;
              auto inclination = i_i;
              s(azimuth_wrap_impl(true, g(azimuth_i)), azimuth_i); // Reflect azimuth.
              s(-g(inclination), inclination); // Reflect inclination.
            }
            break;
          }
          case a_i:
          {
            s(azimuth_wrap_impl(false, x), a_i);
            break;
          }
          default: // case i_i
          {
            const auto [ip, b] = inclination_wrap_impl(x);
            s(values::internal::update_real_part(x, ip), i_i); // Reflect inclination.
            const auto azimuth_i = a_i;
            s(azimuth_wrap_impl(b, g(azimuth_i)), azimuth_i); // Maybe reflect azimuth.
            break;
          }
        }
      }

    };

  } // namespace detail


  /**
   * \internal
   * \brief traits for Spherical<Distance, Angle, Inclination>.
   */
  template<typename Min, typename Max, typename Down, typename Up>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Distance, coordinates::Angle<Min, Max>, coordinates::Inclination<Down, Up>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Distance, coordinates::Angle<Min, Max>, coordinates::Inclination<Down, Up>>, Min, Max, Down, Up, 0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Distance, Inclination, Angle>.
   */
  template<typename Down, typename Up, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Distance, coordinates::Inclination<Down, Up>, coordinates::Angle<Min, Max>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Distance, coordinates::Inclination<Down, Up>, coordinates::Angle<Min, Max>>, Min, Max, Down, Up, 0, 2, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Distance, Inclination>.
   */
  template<typename Min, typename Max, typename Down, typename Up>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Distance, coordinates::Inclination<Down, Up>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Distance, coordinates::Inclination<Down, Up>>, Min, Max, Down, Up, 1, 0, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Distance, Angle>.
   */
  template<typename Down, typename Up, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Inclination<Down, Up>, coordinates::Distance, coordinates::Angle<Min, Max>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Inclination<Down, Up>, coordinates::Distance, coordinates::Angle<Min, Max>>, Min, Max, Down, Up, 1, 2, 0>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Inclination, Distance>.
   */
  template<typename Min, typename Max, typename Down, typename Up>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Inclination<Down, Up>, coordinates::Distance>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Inclination<Down, Up>, coordinates::Distance>, Min, Max, Down, Up, 2, 0, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Angle, Distance>.
   */
  template<typename Down, typename Up, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Inclination<Down, Up>, coordinates::Angle<Min, Max>, coordinates::Distance>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Inclination<Down, Up>, coordinates::Angle<Min, Max>, coordinates::Distance>, Min, Max, Down, Up, 2, 1, 0>
  {};


} // namespace OpenKalman::interface

#endif //OPENKALMAN_SPHERICAL_HPP
