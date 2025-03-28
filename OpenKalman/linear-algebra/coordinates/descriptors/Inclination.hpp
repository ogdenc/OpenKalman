/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Inclination class and related limits.
 */

#ifndef OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
#define OPENKALMAN_COEFFICIENTS_INCLINATION_HPP

#include <type_traits>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include <cmath>
#include "basics/language-features.hpp"
#include "values/concepts/floating.hpp"
#include "values/classes/fixed-constants.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/abs.hpp"
#include "values/math/sin.hpp"
#include "values/math/cos.hpp"
#include "values/math/atan2.hpp"
#include "values/functions/cast_to.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"


namespace OpenKalman::coordinate
{
  /**
   * \brief A positive or negative real number &phi; representing an inclination or declination from the horizon.
   * \details &phi;<sub>down</sub>&le;&phi;&le;&phi;<sub>up</sub>, where &phi;<sub>down</sub> is a real number
   * representing down, and &phi;<sub>up</sub> is a real number representing up. Normally, the horizon will be zero and
   * &phi;<sub>down</sub>=&minus;&phi;<sub>up</sub>, but in general, the horizon is at
   * &frac12;(&phi;<sub>down</sub>+&minus;&phi;<sub>up</sub>).
   * The inclinations inclination::Radians and inclination::Degrees are predefined.
   * \tparam Down a \ref value::fixed "fixed value" representing the down direction. This must be no greater than zero.
   * \tparam Up a \ref value::fixed "fixed value" representing the up direction. This must be no less than zero and must exceed Down.
   */
#ifdef __cpp_concepts
  template<value::fixed Down = value::fixed_minus_half_pi<long double>, value::fixed Up = value::fixed_half_pi<long double>>
  requires (value::fixed_number_of_v<Up> - value::fixed_number_of_v<Down> > 0) and
    (value::fixed_number_of_v<Down> <= 0) and (value::fixed_number_of_v<Up> >= 0)
#else
  template<typename Down = value::fixed_minus_half_pi<long double>, typename Up = value::fixed_half_pi<long double>>
#endif
  struct Inclination
  {
#ifndef __cpp_concepts
    static_assert(value::fixed<Down>);
    static_assert(value::fixed<Up>);
    static_assert(value::fixed_number_of_v<Up> - value::fixed_number_of_v<Down> > 0);
    static_assert(value::fixed_number_of_v<Down> <= 0);
    static_assert(value::fixed_number_of_v<Up> >= 0);
#endif
  };


  /// Namespace for definitions relating to specialized instances of \ref Inclination.
  namespace inclination
  {
    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<>;


    /// An inclination measured in degrees [-90,90].
    using Degrees = Inclination<value::Fixed<long double, -90>, value::Fixed<long double, 90>>;


    namespace detail
    {
      template<typename T>
      struct is_inclination : std::false_type {};

      template<typename Down, typename Up>
      struct is_inclination<Inclination<Down, Up>> : std::true_type {};
    }


    /**
     * \brief T is a \ref coordinate::pattern object representing an inclination.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept inclination =
#else
    static constexpr bool inclination =
#endif
      detail::is_inclination<T>::value;

  } // namespace inclination

} // namespace OpenKalman::coordinate


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Inclination.
   */
  template<typename Down, typename Up>
  struct coordinate_descriptor_traits<coordinate::Inclination<Down, Up>>
  {
  private:

    using T = coordinate::Inclination<Down, Up>;
    static constexpr auto down = value::fixed_number_of_v<Down>;
    static constexpr auto up = value::fixed_number_of_v<Up>;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    euclidean_size(const T&) { return std::integral_constant<std::size_t, 2>{}; };


    static constexpr auto
    is_euclidean(const T&) { return std::false_type{}; }


    static constexpr auto
    hash_code(const T&)
    {
      constexpr auto down_float = static_cast<float>(down);
      constexpr auto up_float = static_cast<float>(up);
      constexpr float a = (down_float * 3.f + up_float * 2.f) / (up_float - down_float);
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0x8CE6_uz + static_cast<std::size_t>(a * a * 0x1.p2f);
      else if constexpr (bits < 64) return 0x8CE6267E_uz + static_cast<std::size_t>(a * a * 0x1.p4f);
      else return 0x8CE6267E341642F7_uz + static_cast<std::size_t>(a * a * 0x1.p8f);
    }


    /**
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
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
      const Scalar cf {numbers::pi_v<R> / (up - down)};
      const Scalar horiz {R{up + down} * R{0.5}};

      Scalar theta = cf * (g(0_uz) - horiz); // Convert to radians

      if (euclidean_local_index == 0) return value::cos(theta);
      else return value::sin(theta);
    }


    /**
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
     * \param local_index A local index accessing the angle (in this case, it must be 0)
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
      using Scalar = std::decay_t<decltype(g(0_uz))>;
      using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
      const Scalar cf {numbers::pi_v<R> / (up - down)};
      const Scalar horiz {R{up + down} * R{0.5}};

      Scalar x = g(0_uz);
      // In Euclidean space, (the real part of) x must be non-negative since the inclination is in range [-½pi,½pi].
      Scalar pos_x = value::internal::update_real_part(x, value::abs(value::real(x)));
      Scalar y = g(1_uz);

      if constexpr (value::complex<Scalar>) return value::atan2(y, pos_x) / cf + horiz;
      else { return value::atan2(y, pos_x) / cf + horiz; }
    }

  private:

    template<typename A>
    static constexpr std::decay_t<A> wrap_impl(A&& a)
    {
      auto ap = value::real(a);
      if (ap >= down and ap <= up)
      {
        return std::forward<decltype(a)>(a);
      }
      else
      {
        using R = std::decay_t<decltype(ap)>;
        constexpr R range {up - down};
        constexpr R period {range * 2};
        using std::fmod;
        auto ar = fmod(ap - R{down}, period);

        if (ar < 0) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{down} + ar + period);
        else if (ar > range) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{down} - ar + period);
        else return value::internal::update_real_part(std::forward<decltype(a)>(a), R{down} + ar);
      }
    }

  public:

    /**
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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
    };


    /**
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
