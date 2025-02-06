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
#include <array>
#include "basics/language-features.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/floating.hpp"
#include "linear-algebra/values/classes/fixed-constants.hpp"
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
  /// Namespace for definitions relating to coefficients representing an inclination.
  namespace inclination
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


    /// An inclination measured in radians [-½&pi;,½&pi;].
    using Radians = Inclination<>;


    /// An inclination measured in degrees [-90,90].
    using Degrees = Inclination<value::Fixed<long double, std::intmax_t{-90}>, value::Fixed<long double, std::intmax_t{90}>>;


    namespace detail
    {
      template<typename T>
      struct is_inclination : std::false_type {};

      template<typename Down, typename Up>
      struct is_inclination<Inclination<Down, Up>> : std::true_type {};
    }


    /**
     * \brief T is a \ref vector_space_descriptor object representing an inclination.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept inclination =
#else
    static constexpr bool inclination =
#endif
      detail::is_inclination<T>::value;

  } // namespace inclination


  using inclination::Inclination;

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Inclination.
   */
  template<typename Down, typename Up>
  struct vector_space_traits<descriptor::Inclination<Down, Up>>
  {
  private:

    using T = descriptor::Inclination<Down, Up>;
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

  private:

#ifdef __cpp_concepts
    template<typename Arg, typename CommonLimitType>
#else
    template<typename Arg, typename CommonLimitType, typename = void>
#endif
    struct CanonicalCast
    {
      using type = value::Fixed<CommonLimitType, static_cast<std::intmax_t>(value::fixed_number_of_v<Arg>)>;
    };

#ifdef __cpp_concepts
    template<value::floating Arg, typename CommonLimitType>
    struct CanonicalCast<Arg, CommonLimitType>
#else
    template<typename Arg, typename CommonLimitType>
    struct CanonicalCast<Arg, CommonLimitType, std::enable_if_t<value::floating<Arg>>>
#endif
    {
      static constexpr auto x = value::fixed_number_of_v<Arg>;
      using X = value::number_type_of_t<Arg>;
      static constexpr auto pi = numbers::pi_v<value::number_type_of_t<Arg>>;
      using type =
        std::conditional_t<x == static_cast<std::intmax_t>(x), value::Fixed<CommonLimitType, static_cast<std::intmax_t>(x)>,
        std::conditional_t<value::internal::near(x, pi), value::fixed_pi<CommonLimitType>,
        std::conditional_t<value::internal::near(x, -pi), value::fixed_minus_pi<CommonLimitType>,
        std::conditional_t<value::internal::near(x, X{2} * pi), value::fixed_2pi<CommonLimitType>,
        std::conditional_t<value::internal::near(x, X{0.5} * pi), value::fixed_half_pi<CommonLimitType>,
        std::conditional_t<value::internal::near(x, X{-0.5} * pi), value::fixed_minus_half_pi<CommonLimitType>,
#if __cpp_nontype_template_args >= 201911L
        value::Fixed<CommonLimitType, static_cast<CommonLimitType>(x)>
#else
        Arg
#endif
        >>>>>>;
    };

  public:

    static constexpr auto
    component_collection(const T&)
    {
      using CommonLimitType = std::common_type_t<long double, value::number_type_of_t<Down>, value::number_type_of_t<Up>>;
      using CDown = typename CanonicalCast<Down, CommonLimitType>::type;
      using CUp = typename CanonicalCast<Up, CommonLimitType>::type;
      return std::array {descriptor::Inclination<CDown, CUp>{}};
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
