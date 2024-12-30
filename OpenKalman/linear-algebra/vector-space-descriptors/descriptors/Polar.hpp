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
#include <stdexcept>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include <cmath>
#include "basics/language-features.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/complex.hpp"
#include "linear-algebra/values/functions/internal/update_real_part.hpp"
#include "linear-algebra/values/functions/signbit.hpp"
#include "linear-algebra/values/functions/abs.hpp"
#include "linear-algebra/values/functions/cos.hpp"
#include "linear-algebra/values/functions/sin.hpp"
#include "linear-algebra/values/functions/atan2.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"
#include "Distance.hpp"
#include "Angle.hpp"


namespace OpenKalman::descriptor
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

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename Limits,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
      static constexpr auto
      size(const T&) { return std::integral_constant<std::size_t, 2>{}; };


      static constexpr auto
      euclidean_size(const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto
      component_count(const T&) { return std::integral_constant<std::size_t, 1>{}; };


      static constexpr auto
      is_euclidean(const T&) { return std::false_type{}; }


      static constexpr auto
      canonical_equivalent(const T& t)
      {
        using A = std::decay_t<decltype(descriptor::internal::canonical_equivalent(descriptor::Angle<Limits>{}))>;
        if constexpr (d_i == 0)
          return descriptor::Polar<descriptor::Distance, A>{};
        else
          return descriptor::Polar<A, descriptor::Distance>{};
      };


      /**
       * \brief Maps a polar coordinate to coordinates in Euclidean space.
       * \details This function takes a set of polar coordinates and converts them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder.
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
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        Scalar d = g(start + d_i);
        if (euclidean_local_index == d2_i)
        {
          return d;
        }
        else
        {
          using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
          const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

          Scalar theta = cf * g(start + a_i) - mid;
          switch(euclidean_local_index)
          {
            case x_i: return value::cos(theta);
            default: return value::sin(theta); // case y_i
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
      static constexpr value::value auto
      from_euclidean_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
      requires requires { {g(euclidean_start)} -> value::value; }
#else
      template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
        value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
      static constexpr auto
      from_euclidean_component(const T&, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
      {
        using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
        Scalar d = g(euclidean_start + d2_i);
        auto dr = value::real(d);
        if (local_index == d_i)
        {
          // A negative distance is reflected to the positive axis.
          return value::internal::update_real_part(d, value::abs(dr));
        }
        else
        {
          using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
          const Scalar cf {2 * numbers::pi_v<R> / (Limits::max - Limits::min)};
          const Scalar mid {R{Limits::max + Limits::min} * R{0.5}};

          // If distance is negative, flip 180 degrees:
          Scalar x = value::signbit(dr) ? -g(euclidean_start + x_i) : g(euclidean_start + x_i);
          Scalar y = value::signbit(dr) ? -g(euclidean_start + y_i) : g(euclidean_start + y_i);

          if constexpr (value::complex<Scalar>) return value::atan2(y, x) / cf + mid;
          else { return value::atan2(y, x) / cf + mid; }
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
        using R = std::decay_t<decltype(value::real(std::declval<decltype(a)>()))>;
        constexpr R period {Limits::max - Limits::min};
        R ap {distance_is_negative ? value::real(a) + period * R{0.5} : value::real(a)};

        if (ap >= Limits::min and ap < Limits::max) // Check if the angle doesn't need wrapping.
        {
          return value::internal::update_real_part(std::forward<decltype(a)>(a), ap);;
        }
        else // Wrap the angle.
        {
          using std::fmod;
          auto ar = fmod(ap - R{Limits::min}, period);
          if (ar < 0) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::min} + ar + period);
          else return value::internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::min} + ar);
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
        switch(local_index)
        {
          case d_i: return value::internal::update_real_part(d, value::abs(value::real(d)));
          default: return polar_angle_wrap_impl(value::signbit(value::real(d)), g(start + a_i)); // case a_i
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
            auto xp = value::real(x);
            s(value::internal::update_real_part(x, value::abs(xp)), start + d_i);
            s(polar_angle_wrap_impl(value::signbit(xp), g(start + a_i)), start + a_i); //< Possibly reflect angle
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


  /**
   * \internal
   * \brief traits for Polar<Distance, Angle>.
   */
  template<typename Limits>
  struct vector_space_traits<descriptor::Polar<descriptor::Distance, descriptor::Angle<Limits>>>
    : detail::PolarBase<descriptor::Polar<descriptor::Distance, descriptor::Angle<Limits>>, Limits, 0, 1,  0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Polar<Angle, Distance>.
   */
  template<typename Limits>
  struct vector_space_traits<descriptor::Polar<descriptor::Angle<Limits>, descriptor::Distance>>
    : detail::PolarBase<descriptor::Polar<descriptor::Angle<Limits>, descriptor::Distance>, Limits, 1, 0,  2, 0, 1>
  {};


}// namespace OpenKalman::interface

#endif //OPENKALMAN_POLAR_HPP