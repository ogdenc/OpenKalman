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
#include <stdexcept>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include <cmath>
#include "basics/language-features.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/floating.hpp"
#include "linear-algebra/values/functions/internal/update_real_part.hpp"
#include "linear-algebra/values/functions/abs.hpp"
#include "linear-algebra/values/functions/sin.hpp"
#include "linear-algebra/values/functions/cos.hpp"
#include "linear-algebra/values/functions/atan2.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/maybe_equivalent_to.hpp"


namespace OpenKalman::descriptor
{
  template<typename Limits>
#ifdef __cpp_concepts
  requires value::value<decltype(Limits::down)> and value::value<decltype(Limits::up)> and
    (Limits::down < Limits::up) and (Limits::down <= 0) and (Limits::up >= 0)
#endif
  struct Inclination;


  /// Namespace for definitions relating to coefficients representing an inclination.
  namespace inclination
  {
    /**
     * \brief The numerical limits of an inclination.
     * \details The lower limit represents the down direction and the upper limit represents the up direction.
     * The range [lower_limit, upper_limit] must include 0.
     * \tparam lower_limit The lower limit of the inclination, representing the down direction
     * \tparam upper_limit The upper limit of the inclination, representing the up direction
     */
    template<auto lower_limit, auto upper_limit>
    struct Limits
    {
      static constexpr auto down = lower_limit;
      static constexpr auto up = upper_limit;
    };


#if __cpp_nontype_template_args >= 201911L
    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<Limits<-numbers::pi_v<long double> / 2, numbers::pi_v<long double> / 2>>;
#else
    /// The limits of an inclination measured in radians: [-½&pi;,½&pi;].
    struct RadiansLimits
    {
      static constexpr long double down = -numbers::pi_v<long double> / 2;
      static constexpr long double up = numbers::pi_v<long double> / 2;
    };

    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<RadiansLimits>;
#endif


    /// An inclination measured in degrees [-90,90].
    using Degrees = Inclination<Limits<-90, 90>>;

  } // namespace inclination


  /**
   * \brief A positive or negative real number &phi; representing an inclination or declination from the horizon.
   * \details &phi;<sub>down</sub>&le;&phi;&le;&phi;<sub>up</sub>, where &phi;<sub>down</sub> is a real number
   * representing down, and &phi;<sub>up</sub> is a real number representing up. Normally, the horizon will be zero and
   * &phi;<sub>down</sub>=&minus;&phi;<sub>up</sub>, but in general, the horizon is at
   * &frac12;(&phi;<sub>down</sub>+&minus;&phi;<sub>up</sub>).
   * The inclinations inclination::Radians and inclination::Degrees are predefined.
   * \tparam Limits A class template defining the real values <code>down</code> and <code>up</code>, where
   * <code>down</code>=&phi;<sub>down</sub> and <code>up</code>=&phi;<sub>up</sub>.
   * Scalar is a \ref value::number.
   */
#if __cpp_nontype_template_args >= 201911L
  template<typename Limits = inclination::Limits<-numbers::pi_v<long double> / 2, numbers::pi_v<long double> / 2>>
#else
  template<typename Limits = inclination::RadiansLimits>
#endif
#ifdef __cpp_concepts
  requires value::value<decltype(Limits::down)> and value::value<decltype(Limits::up)> and
    (Limits::down < Limits::up) and (Limits::down <= 0) and (Limits::up >= 0)
#endif
  struct Inclination
  {
#ifndef __cpp_concepts
    static_assert(std::is_integral_v<decltype(Limits::down)> or value::floating<decltype(Limits::down)>);
    static_assert(std::is_integral_v<decltype(Limits::up)> or value::floating<decltype(Limits::up)>);
    static_assert(Limits::down < Limits::up);
    static_assert(Limits::down <= 0);
    static_assert(Limits::up >= 0);
#endif

    /// Default constructor
    constexpr Inclination() = default;


    /// Conversion constructor
#ifdef __cpp_concepts
    template<maybe_equivalent_to<Inclination> D> requires (not std::same_as<std::decay_t<D>, Inclination>)
#else
    template<typename D, std::enable_if_t<
      maybe_equivalent_to<D, Inclination> and not std::is_same_v<std::decay_t<D>, Inclination>, int> = 0>
#endif
    explicit constexpr Inclination(D&& d)
    {
      if constexpr (dynamic_vector_space_descriptor<D>)
      {
        if (d != Inclination{}) throw std::invalid_argument{"Dynamic argument of 'Inclination' constructor is not an inclination vector space descriptor."};
      }
    }

  };


  namespace detail
  {
    template<typename T>
    struct is_inclination_vector_space_descriptor : std::false_type {};

    template<typename Limits>
    struct is_inclination_vector_space_descriptor<Inclination<Limits>> : std::true_type {};
  }


  /**
   * \brief T is a \ref vector_space_descriptor object representing an inclination.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept inclination_vector_space_descriptor =
#else
  static constexpr bool inclination_vector_space_descriptor =
#endif
    detail::is_inclination_vector_space_descriptor<T>::value;

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Inclination.
   */
  template<typename Limits>
  struct vector_space_traits<descriptor::Inclination<Limits>>
  {
  private:

    using T = descriptor::Inclination<Limits>;

  public:

    static constexpr auto
    size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    euclidean_size(const T&) { return std::integral_constant<std::size_t, 2>{}; };


    static constexpr auto
    component_count(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    is_euclidean(const T&) { return std::false_type{}; }


    /**
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
     * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean space,
     * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
     * so Limits<Scalar>::max wraps back to the point (-1, 0).
     * \param euclidean_local_index A local index accessing either the x (if 0) or y (if 1) coordinate in Euclidean space
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
      using Scalar = std::decay_t<decltype(g(start))>;
      using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
      const Scalar cf {numbers::pi_v<R> / (Limits::up - Limits::down)};
      const Scalar horiz {R{Limits::up + Limits::down} * R{0.5}};

      Scalar theta = cf * (g(start) - horiz); // Convert to radians

      if (euclidean_local_index == 0) return value::cos(theta);
      else return value::sin(theta);
    }


    /**
     * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
     * \param local_index A local index accessing the angle (in this case, it must be 0)
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
      using Scalar = std::decay_t<decltype(g(euclidean_start))>;
      using R = std::decay_t<decltype(value::real(std::declval<Scalar>()))>;
      const Scalar cf {numbers::pi_v<R> / (Limits::up - Limits::down)};
      const Scalar horiz {R{Limits::up + Limits::down} * R{0.5}};

      Scalar x = g(euclidean_start);
      // In Euclidean space, (the real part of) x must be non-negative since the inclination is in range [-½pi,½pi].
      Scalar pos_x = value::internal::update_real_part(x, value::abs(value::real(x)));
      Scalar y = g(euclidean_start + 1);

      if constexpr (value::complex<Scalar>) return value::atan2(y, pos_x) / cf + horiz;
      else { return value::atan2(y, pos_x) / cf + horiz; }
    }

  private:

    template<typename A>
    static constexpr std::decay_t<A> wrap_impl(A&& a)
    {
      auto ap = value::real(a);
      if (ap >= Limits::down and ap <= Limits::up)
      {
        return std::forward<decltype(a)>(a);
      }
      else
      {
        using R = std::decay_t<decltype(ap)>;
        constexpr R range {Limits::up - Limits::down};
        constexpr R period {range * 2};
        using std::fmod;
        auto ar = fmod(ap - R{Limits::down}, period);

        if (ar < 0) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::down} + ar + period);
        else if (ar > range) return value::internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::down} - ar + period);
        else return value::internal::update_real_part(std::forward<decltype(a)>(a), R{Limits::down} + ar);
      }
    }

  public:

    /**
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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
      return wrap_impl(g(start));
    };


    /**
     * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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
      s(wrap_impl(x), start);
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
