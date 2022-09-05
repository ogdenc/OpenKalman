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

#include <array>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif


namespace OpenKalman
{
  template<typename Limits>
#ifdef __cpp_concepts
  requires floating_scalar_type<decltype(Limits::down)> and floating_scalar_type<decltype(Limits::up)> and
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
    using Radians = Inclination<Limits<-numbers::pi_v<long double> / 2, -numbers::pi_v<long double> / 2>>;
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
   * Scalar is a \ref floating_scalar_type.
   */
#if __cpp_nontype_template_args >= 201911L
  template<typename Limits = inclination::Limits<-numbers::pi_v<long double> / 2, -numbers::pi_v<long double> / 2>>
#else
  template<typename Limits = inclination::RadiansLimits>
#endif
#ifdef __cpp_concepts
    requires floating_scalar_type<decltype(Limits::down)> and floating_scalar_type<decltype(Limits::up)> and
      (Limits::down < Limits::up) and (Limits::down <= 0) and (Limits::up >= 0)
#endif
  struct Inclination
  {
#ifndef __cpp_concepts
    static_assert(floating_scalar_type<decltype(Limits::down)>);
    static_assert(floating_scalar_type<decltype(Limits::up)>);
    static_assert(Limits::down < Limits::up);
    static_assert(Limits::down <= 0);
    static_assert(Limits::up >= 0);
#endif
  };


  namespace internal
  {
    template<typename T>
    struct is_inclination_descriptor : std::false_type {};

    template<typename Limits>
    struct is_inclination_descriptor<Inclination<Limits>> : std::true_type {};
  }


  /**
   * \brief T is an index descriptor of an inclination.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept inclination_descriptor =
#else
  static constexpr bool inclination_descriptor =
#endif
    internal::is_inclination_descriptor<T>::value;


  namespace interface
  {
    /**
     * \internal
     * \brief traits for Inclination.
     */
    template<typename Limits>
    struct FixedIndexDescriptorTraits<Inclination<Limits>>
    {
      static constexpr std::size_t size = 1;
      static constexpr std::size_t euclidean_size = 2;
      static constexpr std::size_t component_count = 1;
      using difference_type = Dimensions<1>;
      static constexpr bool always_euclidean = false;


      /**
       * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
       * By convention, the minimum angle limit Limits<Scalar::min corresponds to the point (-1,0) in Euclidean space,
       * and the angle is scaled so that the difference between Limits<Scalar>::min and Limits<<Scalar>::max is 2&pi;,
       * so Limits<Scalar>::max wraps back to the point (-1, 0).
       * \param euclidean_local_index A local index accessing either the x (if 0) or y (if 1) coordinate in Euclidean space
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
        using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
        const Scalar cf {numbers::pi_v<R> / (Limits::up - Limits::down)};
        const Scalar horiz {R{Limits::up + Limits::down} * R{0.5}};

        Scalar theta = cf * (g(start) - horiz); // Convert to radians
        if (euclidean_local_index == 0)
          return interface::ScalarTraits<Scalar>::cos(theta);
        else
          return interface::ScalarTraits<Scalar>::sin(theta);
      }


      /**
       * \details The inclination angle corresponds to x and y coordinates in quadrants I or IV of a unit circle.
       * \param local_index A local index accessing the angle (in this case, it must be 0)
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
        using R = std::decay_t<decltype(real_projection(std::declval<Scalar>()))>;
        const Scalar cf {numbers::pi_v<R> / (Limits::up - Limits::down)};
        const Scalar horiz {R{Limits::up + Limits::down} * R{0.5}};

        Scalar x = g(euclidean_start);
        // In Euclidean space, (the real part of) x must be non-negative since the inclination is in range [-½pi,½pi].
        Scalar pos_x = inverse_real_projection(x, std::abs(real_projection(x)));
        Scalar y = g(euclidean_start + 1);
        return interface::ScalarTraits<Scalar>::atan2(y, pos_x) / cf + horiz;
      }

    private:

      template<typename A>
      static constexpr std::decay_t<A> wrap_impl(A&& a)
      {
        auto ap = real_projection(a);
        if (ap >= Limits::down and ap <= Limits::up)
        {
          return std::forward<decltype(a)>(a);
        }
        else
        {
          using R = std::decay_t<decltype(ap)>;
          constexpr R range {Limits::up - Limits::down};
          constexpr R period {range * 2};
          auto ar = std::fmod(ap - R{Limits::down}, period);

          if (ar < 0) return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::down} + ar + period);
          else if (ar > range) return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::down} - ar + period);
          else return inverse_real_projection(std::forward<decltype(a)>(a), R{Limits::down} + ar);
        }
      }

    public:

      /**
       * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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
        return wrap_impl(g(start));
      };


      /**
       * \param local_index A local index accessing the inclination angle (in this case, it must be 0)
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
        s(wrap_impl(x), start);
      }

    };


  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_HPP
