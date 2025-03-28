/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#include <type_traits>
#include <cmath>
#include "basics/language-features.hpp"
#include "values/concepts/fixed.hpp"
#include "values/concepts/value.hpp"
#include "values/classes/fixed-constants.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/sin.hpp"
#include "values/math/cos.hpp"
#include "values/math/atan2.hpp"
#include "values/functions/cast_to.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [Min,Max) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam Min A \ref value::fixed "fixed value" representing the minimum value beyond which wrapping occurs. This must be no greater than 0.
   * \tparam Max A \ref value::fixed "fixed value" representing the maximum value beyond which wrapping occurs. This must be greater than 0.
   */
#ifdef __cpp_concepts
  template<value::fixed Min = value::fixed_minus_pi<long double>, value::fixed Max = value::fixed_pi<long double>>
  requires (value::fixed_number_of_v<Min> <= 0) and (value::fixed_number_of_v<Max> > 0) and
    std::common_with<long double, value::number_type_of_t<Min>> and
    std::common_with<long double, value::number_type_of_t<Max>> and
    std::common_with<value::number_type_of_t<Min>, value::number_type_of_t<Max>>
#else
template<typename Min = value::fixed_minus_pi<long double>, typename Max = value::fixed_pi<long double>>
#endif
  struct Angle
  {
#ifndef __cpp_concepts
    static_assert(value::fixed<Min>);
    static_assert(value::fixed<Max>);
    static_assert(value::fixed_number_of_v<Min> <= 0);
    static_assert(value::fixed_number_of_v<Max> > 0);
#endif
  };


  /// Namespace for definitions relating to specialized instances of \ref Angle.
  namespace angle
  {
    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<>;


    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<value::Fixed<long double, 0>, value::fixed_2pi<long double>>;


    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<value::Fixed<long double, 0>, value::Fixed<long double, 360>>;


    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<value::Fixed<long double, -180>, value::Fixed<long double, 180>>;


    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<value::Fixed<long double, 0>, value::Fixed<long double, 1>>;


    namespace detail
    {
      template<typename T>
      struct is_angle : std::false_type {};

      template<typename Min, typename Max>
      struct is_angle<Angle<Min, Max>> : std::true_type {};
    }


    /**
     * \brief T is a \ref coordinate::pattern object representing an angle.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept angle =
#else
    static constexpr bool angle =
#endif
      detail::is_angle<T>::value;

  } // namespace angle


} // OpenKalman::coordinate


namespace OpenKalman::interface
{

  /**
   * \internal
   * \brief traits for Angle.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinate::Angle<Min, Max>>
  {
  private:

    using T = coordinate::Angle<Min, Max>;
    static constexpr auto min = value::fixed_number_of_v<Min>;
    static constexpr auto max = value::fixed_number_of_v<Max>;


  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    euclidean_size(const T&) { return std::integral_constant<std::size_t, 2>{}; };


    static constexpr auto
    is_euclidean(const T&) { return std::false_type{}; }


    static constexpr std::size_t
    hash_code(const T&)
    {

      constexpr auto min_float = static_cast<float>(min);
      constexpr auto max_float = static_cast<float>(max);
      constexpr float a = (min_float * 3.f + max_float * 2.f) / (max_float - min_float);
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0x62BB_uz + static_cast<std::size_t>(a * a * 0x1.p2f);
      else if constexpr (bits < 64) return 0x62BB0D37_uz + static_cast<std::size_t>(a * a * 0x1.p4f);
      else return 0x62BB0D37A58D6F96_uz + static_cast<std::size_t>(a * a * 0x1.p8f);
    }


    /*
     * \details The angle corresponds to x and y coordinates on a unit circle.
     * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean space,
     * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
     * so Limits<Scalar>::max wraps back to the point (-1, 0).
     * \param euclidean_local_index A local index accessing either the x (if 0) or y (if 1) coordinate in Euclidean space
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      using Scalar = std::decay_t<decltype(g(0_uz))>;
      using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
      const Scalar cf {2 * numbers::pi_v<R> / (max - min)};
      const Scalar mid { R{max + min} * R{0.5}};

      Scalar theta = cf * (g(0_uz) - mid); // Convert to radians

      if (euclidean_local_index == 0) return value::cos(theta);
      else return value::sin(theta);
    }


    /*
     * \details The angle corresponds to x and y coordinates on a unit circle.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      using Scalar = std::decay_t<decltype(g(std::declval<std::size_t>()))>;
      using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
      const Scalar cf {2 * numbers::pi_v<R> / (max - min)};
      const Scalar mid { R{max + min} * R{0.5}};

      Scalar x = g(0_uz);
      Scalar y = g(1_uz);

      if constexpr (value::complex<Scalar>) return value::atan2(y, x) / cf + mid;
      else { return value::atan2(y, x) / cf + mid; }
    }


  private:

#ifdef __cpp_concepts
    static constexpr auto wrap_impl(auto&& a) -> std::decay_t<decltype(a)>
#else
    template<typename Scalar>
    static constexpr std::decay_t<Scalar> wrap_impl(Scalar&& a)
#endif
    {
      auto ap = value::real(a);
      if (not (ap < min) and ap < max)
      {
        return std::forward<decltype(a)>(a);
      }
      else
      {
        using R = std::decay_t<decltype(ap)>;
        constexpr R period {max - min};
        using std::fmod;
        R ar {fmod(ap - R{min}, period)};
        if (ar < 0) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar + period);
        else return value::internal::update_real_part(std::forward<decltype(a)>(a), R{min} + ar);
      }
    }

  public:

    /*
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      return wrap_impl(g(0_uz));
    }


    /**
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const value::value auto& x, const value::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<value::value<X> and value::index<L> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
    {
      s(wrap_impl(x), 0_uz);
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_ANGLE_HPP
