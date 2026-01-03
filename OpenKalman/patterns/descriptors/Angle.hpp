/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2026 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "values/functions/internal/update_real_part.hpp"
#include "collections/collections.hpp"
#include "patterns/interfaces/pattern_descriptor_traits.hpp"
#include "Any.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief An angle or any other simple modular value.
   * \details An angle wraps to a given interval [Min,Max) when it increases or decreases outside that range.
   * There are several predefined angles, including angle::Radians, angle::Degrees, angle::PositiveRadians,
   * angle::PositiveDegrees, and angle::Circle.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam Min A \ref values::fixed "fixed value" representing the minimum value beyond which wrapping occurs. This must be no greater than 0.
   * \tparam Max A \ref values::fixed "fixed value" representing the maximum value beyond which wrapping occurs. This must be greater than 0.
   */
#ifdef __cpp_concepts
  template<values::fixed Min = values::fixed_minus_pi<long double>, values::fixed Max = values::fixed_pi<long double>>
  requires (values::fixed_value_of_v<Min> <= 0) and (values::fixed_value_of_v<Max> > 0) and
    (not values::complex<Min>) and (not values::complex<Max>) and
    std::convertible_to<values::value_type_of_t<Min>, float> and
    std::convertible_to<values::value_type_of_t<Max>, float> and
    std::common_with<values::value_type_of_t<Min>, values::value_type_of_t<Max>>
#else
template<typename Min = values::fixed_minus_pi<long double>, typename Max = values::fixed_pi<long double>>
#endif
  struct Angle
  {
#ifndef __cpp_concepts
    static_assert(values::fixed<Min>);
    static_assert(values::fixed<Max>);
    static_assert(values::fixed_value_of_v<Min> <= 0);
    static_assert(values::fixed_value_of_v<Max> > 0);
    static_assert(not values::complex<Min>);
    static_assert(not values::complex<Max>);
    static_assert(stdex::convertible_to<values::value_type_of_t<Min>, float>);
    static_assert(stdex::convertible_to<values::value_type_of_t<Max>, float>);
    static_assert(stdex::common_with<values::value_type_of_t<Min>, values::value_type_of_t<Max>>);
#endif
  };


  /// Namespace for definitions relating to specialized instances of \ref Angle.
  namespace angle
  {
    /// An angle measured in radians [-&pi;,&pi;).
    using Radians = Angle<>;


    /// An angle measured in positive radians [0,2&pi;).
    using PositiveRadians = Angle<values::fixed_value<long double, 0>, values::fixed_2pi<long double>>;


    /// An angle measured in degrees [0,360).
    using PositiveDegrees = Angle<values::fixed_value<long double, 0>, values::fixed_value<long double, 360>>;


    /// An angle measured in positive or negative degrees [-180,180).
    using Degrees = Angle<values::fixed_value<long double, -180>, values::fixed_value<long double, 180>>;


    /// An wrapping circle such as the wrapping interval [0,1).
    using Circle = Angle<values::fixed_value<long double, 0>, values::fixed_value<long double, 1>>;


    namespace detail
    {
      template<typename T>
      struct is_angle : std::false_type {};

      template<typename Min, typename Max>
      struct is_angle<Angle<Min, Max>> : std::true_type {};
    }


    /**
     * \brief T is a \ref patterns::pattern object representing an angle.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept angle =
#else
    static constexpr bool angle =
#endif
      detail::is_angle<T>::value;

  }

}


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Angle.
   */
  template<typename Min, typename Max>
  struct pattern_descriptor_traits<patterns::Angle<Min, Max>>
  {
  private:

    using T = patterns::Angle<Min, Max>;
    static constexpr auto min = values::fixed_value_of_v<Min>;
    static constexpr auto max = values::fixed_value_of_v<Max>;


    template<typename...Args>
    static constexpr auto make_range(Args&&...args)
    {
      if constexpr ((... or values::fixed<Args>))
        return std::tuple {std::forward<Args>(args)...};
      else
        return std::array<std::common_type_t<Args...>, sizeof...(Args)> {std::forward<Args>(args)...};
    }

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto dimension = [](const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto stat_dimension = [](const T&) { return std::integral_constant<std::size_t, 2>{}; };


    static constexpr auto is_euclidean = [](const T&) { return std::false_type{}; };


    static constexpr auto hash_code = [](const T&)
    {
      constexpr auto min_float = static_cast<float>(min);
      constexpr auto max_float = static_cast<float>(max);
      constexpr float a = (max_float * 3.f + min_float * 2.f + 1.f) / (max_float - min_float + 1.f);
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32)
        return std::integral_constant<std::size_t, 0x62BB_uz + static_cast<std::size_t>(a * a * 0x1.p2f)>{};
      else if constexpr (bits < 64)
        return std::integral_constant<std::size_t, 0x62BB0D37_uz + static_cast<std::size_t>(a * a * 0x1.p4f)>{};
      else
        return std::integral_constant<std::size_t, 0x62BB0D37A58D6F96_uz + static_cast<std::size_t>(a * a * 0x1.p8f)>{};
    };


    /*
     * \brief Maps the angle to corresponding x and y coordinates on a unit circle.
     */
    static constexpr auto
    to_stat_space = [](const T&, auto&& data_view)
    {
      decltype(auto) a = collections::get<0>(std::forward<decltype(data_view)>(data_view));
      using R = values::real_type_of_t<values::real_type_of_t<decltype(a)>>;
      if constexpr (min == -stdex::numbers::pi_v<R> and max == stdex::numbers::pi_v<R>) //< Avoid scaling, if possible.
      {
        return make_range(values::cos(a), values::sin(a));
      }
      else
      {
        constexpr auto period = values::cast_to<R>(values::operation(std::minus{}, Max{}, Min{}));
        constexpr auto scale = values::operation(std::divides{}, values::fixed_2pi<R>{}, period);
        auto phi = values::operation(std::multiplies{}, std::forward<decltype(a)>(a), scale);
        return make_range(values::cos(phi), values::sin(phi));
      }
    };

  private:

    struct wrap_phi
    {
      template<typename R>
      constexpr R operator()(const R& phi_real) const
      {
        constexpr R period = max - min;
        if (phi_real < R{min}) return phi_real + period;
        if (phi_real >= R{max}) return phi_real - period;
        return phi_real;
      }
    };

  public:

    /*
     * \brief Maps x and y coordinates on Euclidean space back to an angle.
     * \details This performs bounds checking to ensure that the angle is within the primary range.
     */
    static constexpr auto
    from_stat_space = [](const T&, auto&& data_view)
    {
      decltype(auto) x = collections::get<0>(std::forward<decltype(data_view)>(data_view));
      decltype(auto) y = collections::get<1>(std::forward<decltype(data_view)>(data_view));
      using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;
      if constexpr (min == -stdex::numbers::pi_v<R> and max == stdex::numbers::pi_v<R>) //< Avoid scaling and wrapping, if possible.
      {
        return std::array {values::atan2(std::forward<decltype(y)>(y), std::forward<decltype(x)>(x))};
      }
      else
      {
        constexpr auto period = values::cast_to<R>(values::operation(std::minus{}, Max{}, Min{}));
        constexpr auto scale = values::operation(std::divides{}, period, values::fixed_2pi<R>{});
        auto phi = values::operation(std::multiplies{}, values::atan2(std::forward<decltype(y)>(y), std::forward<decltype(x)>(x)), scale);
        return std::array {values::internal::update_real_part(std::move(phi),
          values::operation(wrap_phi{}, values::real(std::move(phi))))};
      }
    };

  private:

    struct wrap_phi_mod
    {
      template<typename R>
      constexpr R operator()(const R& phi_real) const
      {
        constexpr R period = max - min;
        if (phi_real >= R{min} and phi_real < max) return phi_real;
        R phi_real_mod {values::fmod(phi_real - R{min}, period)};
        if (phi_real_mod < 0) return R{min} + phi_real_mod + period;
        return R{min} + phi_real_mod;
      }
    };

  public:

    /*
     * \brief Wrap the angle to its primary range.
     */
    static constexpr auto
    wrap = [](const T&, auto&& data_view)
    {
      decltype(auto) phi = collections::get<0>(std::forward<decltype(data_view)>(data_view));
      return std::array {values::internal::update_real_part(std::forward<decltype(phi)>(phi),
        values::operation(wrap_phi_mod{}, values::real(values::real(phi))))};
    };

  };

}


namespace std
{
  template<typename Min1, typename Max1, typename Min2, typename Max2>
  struct common_type<OpenKalman::patterns::Angle<Min1, Max1>, OpenKalman::patterns::Angle<Min2, Max2>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2>,
      OpenKalman::patterns::Angle<Min1, Max1>,
      OpenKalman::patterns::Any<>> {};


  template<typename Min1, typename Max1, typename Scalar>
  struct common_type<OpenKalman::patterns::Angle<Min1, Max1>, OpenKalman::patterns::Any<Scalar>>
    : common_type<OpenKalman::patterns::Any<Scalar>, OpenKalman::patterns::Angle<Min1, Max1>> {};


  template<typename Min1, typename Max1, typename T>
  struct common_type<OpenKalman::patterns::Angle<Min1, Max1>, T>
    : std::conditional_t<
      OpenKalman::patterns::descriptor<T>,
      OpenKalman::stdex::type_identity<OpenKalman::patterns::Any<>>,
      std::monostate> {};
}

#endif
