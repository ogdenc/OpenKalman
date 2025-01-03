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
#include <stdexcept>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include <cmath>
#include <array>
#include "basics/language-features.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/functions/real.hpp"
#include "linear-algebra/values/functions/internal/update_real_part.hpp"
#include "linear-algebra/values/functions/signbit.hpp"
#include "linear-algebra/values/functions/sqrt.hpp"
#include "linear-algebra/values/functions/abs.hpp"
#include "linear-algebra/values/functions/sin.hpp"
#include "linear-algebra/values/functions/cos.hpp"
#include "linear-algebra/values/functions/asin.hpp"
#include "linear-algebra/values/functions/atan2.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"


namespace OpenKalman::descriptor
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
    (distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C2> and inclination_vector_space_descriptor<C3>) or
    (distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C3> and inclination_vector_space_descriptor<C2>) or
    (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C1> and inclination_vector_space_descriptor<C3>) or
    (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C3> and inclination_vector_space_descriptor<C1>) or
    (distance_vector_space_descriptor<C3> and angle_vector_space_descriptor<C1> and inclination_vector_space_descriptor<C2>) or
    (distance_vector_space_descriptor<C3> and angle_vector_space_descriptor<C2> and inclination_vector_space_descriptor<C1>)
#endif
  struct Spherical
  {
#ifndef __cpp_concepts
    static_assert(
      (distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C2> and inclination_vector_space_descriptor<C3>) or
      (distance_vector_space_descriptor<C1> and angle_vector_space_descriptor<C3> and inclination_vector_space_descriptor<C2>) or
      (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C1> and inclination_vector_space_descriptor<C3>) or
      (distance_vector_space_descriptor<C2> and angle_vector_space_descriptor<C3> and inclination_vector_space_descriptor<C1>) or
      (distance_vector_space_descriptor<C3> and angle_vector_space_descriptor<C1> and inclination_vector_space_descriptor<C2>) or
      (distance_vector_space_descriptor<C3> and angle_vector_space_descriptor<C2> and inclination_vector_space_descriptor<C1>));
#endif

    /// Default constructor
    constexpr Spherical() = default;


    /// Conversion constructor
#ifdef __cpp_concepts
    template<maybe_equivalent_to<Spherical> D> requires (not std::same_as<std::decay_t<D>, Spherical>)
#else
    template<typename D, std::enable_if_t<
      maybe_equivalent_to<D, Spherical> and not std::is_same_v<std::decay_t<D>, Spherical>, int> = 0>
#endif
    explicit constexpr Spherical(D&& d)
    {
      if constexpr (dynamic_vector_space_descriptor<D>)
      {
        if (d != Spherical{}) throw std::invalid_argument{"Dynamic argument of 'Spherical' constructor is not a spherical vector space descriptor."};
      }
    }

  };

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename CircleLimits, typename InclinationLimits, std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
      static constexpr auto
      size(const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto
      euclidean_size(const T&) { return std::integral_constant<std::size_t, 4>{}; };


      static constexpr auto
      collection(const T& t) { return std::array {t}; }


      static constexpr auto
      is_euclidean(const T&) { return std::false_type{}; }

    private:

      template<std::size_t i>
      using Part = std::conditional_t<a_i == i,
        std::decay_t<decltype(descriptor::internal::canonical_equivalent(descriptor::Angle<CircleLimits>{}))>,
        std::conditional_t<i_i == i, descriptor::Inclination<InclinationLimits>, descriptor::Distance>>;

    public:

      static constexpr auto
      canonical_equivalent(const T& t)
      {
        return descriptor::Spherical<Part<0>, Part<1>, Part<2>>{};
      };

    private:

      static constexpr std::size_t d2_i = 0, x_i = 1, y_i = 2, z_i = 3;

    public:

      /**
       * \brief Maps an element to coordinates in Euclidean space.
       * \details This function takes a set of spherical coordinates and converts them to d, x, y, and z
       * Cartesian coordinates representing a location on a unit 4D half-cylinder.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
       * \param start The starting index within the \ref vector_space_descriptor object
       */
#ifdef __cpp_concepts
      static constexpr value::value auto
      to_euclidean_component(const T&, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
      requires requires { {g(start)} -> value::value; }
#else
      template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
        value::value<typename std::invoke_result<const Getter&, const S&>::type> and value::index<L> and value::index<S>, int> = 0>
      static constexpr auto
      to_euclidean_component(const T&, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
      {
        if (euclidean_local_index == d2_i)
        {
          return g(start + d_i);
        }
        else
        {
          using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
          using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
          const Scalar cf_inc {numbers::pi_v<R> / (InclinationLimits::up - InclinationLimits::down)};
          const Scalar horiz {R{InclinationLimits::up + InclinationLimits::down} * R{0.5}};

          Scalar phi = cf_inc * (g(start + i_i) - horiz);
          if (euclidean_local_index == z_i)
          {
            return value::sin(phi);
          }
          else
          {
            const Scalar cf_cir {2 * numbers::pi_v<R> / (CircleLimits::max - CircleLimits::min)};
            const Scalar mid {R{CircleLimits::max + CircleLimits::min} * R{0.5}};
            Scalar theta = cf_cir * (g(start + a_i) - mid);

            if (euclidean_local_index == x_i) return value::cos(theta) * value::cos(phi);
            else return value::sin(theta) * value::cos(phi); // euclidean_local_index == y_i
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
      static constexpr value::value auto
      from_euclidean_component(const T&, const auto& g, std::size_t local_index, std::size_t euclidean_start)
      requires requires (std::size_t i){ {g(i)} -> value::value; }
  #else
      template<typename G, std::enable_if_t<value::value<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
      static constexpr auto
      from_euclidean_component(const T&, const G& g, std::size_t local_index, std::size_t euclidean_start)
  #endif
      {
        using Scalar = decltype(g(std::declval<std::size_t>()));
        Scalar d = g(euclidean_start + d2_i);
        auto dr = value::real(d);

        if (local_index == d_i)
        {
          return value::internal::update_real_part(d, value::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
          const Scalar cf_cir {2 * numbers::pi_v<R> / (CircleLimits::max - CircleLimits::min)};
          const Scalar mid {R{CircleLimits::max + CircleLimits::min} * R{0.5}};

          Scalar x = g(euclidean_start + x_i);
          Scalar y = g(euclidean_start + y_i);

          switch(local_index)
          {
            case a_i:
            {
              auto xp = value::real(g(euclidean_start + x_i));
              auto yp = value::real(g(euclidean_start + y_i));
              // If distance is negative, flip x and y axes 180 degrees:
              Scalar x2 = value::internal::update_real_part(x, value::signbit(dr) ? -xp : xp);
              Scalar y2 = value::internal::update_real_part(y, value::signbit(dr) ? -yp : yp);

              if constexpr (value::complex<Scalar>) return value::atan2(y2, x2) / cf_cir + mid;
              else { return value::atan2(y2, x2) / cf_cir + mid; }
            }
            default: // case i_i
            {
              const Scalar cf_inc {numbers::pi_v<R> / (InclinationLimits::up - InclinationLimits::down)};
              const Scalar horiz {R{InclinationLimits::up + InclinationLimits::down} * R{0.5}};
              Scalar z {g(euclidean_start + z_i)};
              auto zp = value::real(z);
              Scalar z2 {value::internal::update_real_part(z, value::signbit(dr) ? -zp : zp)};
              Scalar r {value::sqrt(x*x + y*y + z2*z2)};
              if constexpr (value::complex<Scalar>)
              {
                auto theta = (r == Scalar{0}) ? r : value::asin(z2/r);
                return theta / cf_inc + horiz;
              }
              else
              {
                using R = std::decay_t<decltype(value::asin(z2/r))>;
                // This is so that a zero-radius or faulty spherical coordinate has horizontal inclination:
                auto theta = (r == 0 or r < z2 or z2 < -r) ? R{0} : value::asin(z2/r);
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
        using Ret = std::tuple<std::decay_t<std::decay_t<decltype(value::real(a))>>, bool>;
        auto ap = value::real(a);
        using R = std::decay_t<decltype(ap)>;
        if (ap >= InclinationLimits::down and ap <= InclinationLimits::up) // A shortcut, for the easy case.
        {
          return Ret { ap, false };
        }
        else
        {
          constexpr R period = 2 * (InclinationLimits::up - InclinationLimits::down);
          using std::fmod;
          R ar = fmod(ap - R{InclinationLimits::down}, period);
          R ar2 = ar < 0 ? ar + period : ar;
          bool b = ar2 > InclinationLimits::up - InclinationLimits::down; // Whether there is a mirror reflection about vertical axis.
          return Ret { R{InclinationLimits::down} + (b ? period - ar2 : ar2), b };
        }
      }


      template<typename Scalar>
      static constexpr std::decay_t<Scalar>
      azimuth_wrap_impl(bool reflect_azimuth, Scalar&& a)
      {
        using R = std::decay_t<decltype(value::real(std::declval<decltype(a)>()))>;
        constexpr R period {CircleLimits::max - CircleLimits::min};
        constexpr R half_period {(CircleLimits::max - CircleLimits::min) / R{2}};
        R ap = reflect_azimuth ? value::real(a) - half_period : value::real(a);

        if (ap >= CircleLimits::min and ap < CircleLimits::max) // Check if angle doesn't need wrapping.
        {
          return value::internal::update_real_part(std::forward<decltype(a)>(a), ap);;
        }
        else // Wrap the angle.
        {
          using std::fmod;
          auto ar = fmod(ap - R{CircleLimits::min}, period);
          if (ar < 0) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{CircleLimits::min} + ar + period);
          else return value::internal::update_real_part(std::forward<decltype(a)>(a), R{CircleLimits::min} + ar);
        }
      }

    public:

      /**
       * \brief Perform modular wrapping of spherical coordinates.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index accessing the angle (in this case, it must be 0)
       * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
       */
#ifdef __cpp_concepts
      static constexpr value::value auto
      get_wrapped_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& start)
      requires requires { {g(start)} -> value::value; }
#else
      template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
        value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
      static constexpr auto
      get_wrapped_component(const T&, const Getter& g, const L& local_index, const S& start)
#endif
      {
        auto d = g(start + d_i);
        auto dp = value::real(d);

        switch(local_index)
        {
          case d_i:
          {
            return value::internal::update_real_part(d, value::abs(dp));
          }
          case a_i:
          {
            const bool b = std::get<1>(inclination_wrap_impl(g(start + i_i)));
            return azimuth_wrap_impl(b != value::signbit(dp), g(start + a_i));
          }
          default: // case i_i
          {
            auto i = g(start + i_i);
            auto new_i = std::get<0>(inclination_wrap_impl(i));
            return value::internal::update_real_part(i, value::signbit(dp) ? -new_i : new_i);
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
       * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
       */
#ifdef __cpp_concepts
      static constexpr void
      set_wrapped_component(const T&, const auto& s, const auto& g, const value::value auto& x,
        const value::index auto& local_index, const value::index auto& start)
      requires requires { s(x, start); s(g(start), start); }
#else
      template<typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
        value::value<X> and value::index<L> and value::index<S> and
        std::is_invocable<const Setter&, const X&, const S&>::value and
        std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, const S&>::type, const S&>::value, int> = 0>
      static constexpr void
      set_wrapped_component(const T&, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start)
#endif
      {
        switch(local_index)
        {
          case d_i:
          {
            auto dp = value::real(x);
            s(value::internal::update_real_part(x, value::abs(dp)), start + d_i);
            if (value::signbit(dp)) // If new distance would have been negative
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
            s(value::internal::update_real_part(x, ip), start + i_i); // Reflect inclination.
            const auto azimuth_i = start + a_i;
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
  template<typename ALimits, typename ILimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Distance, descriptor::Angle<ALimits>, descriptor::Inclination<ILimits>>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Distance, descriptor::Angle<ALimits>, descriptor::Inclination<ILimits>>, ALimits, ILimits, 0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Distance, Inclination, Angle>.
   */
  template<typename ILimits, typename ALimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Distance, descriptor::Inclination<ILimits>, descriptor::Angle<ALimits>>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Distance, descriptor::Inclination<ILimits>, descriptor::Angle<ALimits>>, ALimits, ILimits, 0, 2, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Distance, Inclination>.
   */
  template<typename ALimits, typename ILimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Angle<ALimits>, descriptor::Distance, descriptor::Inclination<ILimits>>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Angle<ALimits>, descriptor::Distance, descriptor::Inclination<ILimits>>, ALimits, ILimits, 1, 0, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Distance, Angle>.
   */
  template<typename ILimits, typename ALimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Inclination<ILimits>, descriptor::Distance, descriptor::Angle<ALimits>>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Inclination<ILimits>, descriptor::Distance, descriptor::Angle<ALimits>>, ALimits, ILimits, 1, 2, 0>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Inclination, Distance>.
   */
  template<typename ALimits, typename ILimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Angle<ALimits>, descriptor::Inclination<ILimits>, descriptor::Distance>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Angle<ALimits>, descriptor::Inclination<ILimits>, descriptor::Distance>, ALimits, ILimits, 2, 0, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Angle, Distance>.
   */
  template<typename ILimits, typename ALimits>
  struct vector_space_traits<descriptor::Spherical<descriptor::Inclination<ILimits>, descriptor::Angle<ALimits>, descriptor::Distance>>
    : detail::SphericalBase<descriptor::Spherical<descriptor::Inclination<ILimits>, descriptor::Angle<ALimits>, descriptor::Distance>, ALimits, ILimits, 2, 1, 0>
  {};


} // namespace OpenKalman::interface

#endif //OPENKALMAN_SPHERICAL_HPP
