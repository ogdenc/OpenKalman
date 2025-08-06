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

#ifndef OPENKALMAN_INCLINATION_HPP
#define OPENKALMAN_INCLINATION_HPP

#include <type_traits>
#include <cmath>
#include <array>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/functions/comparison-operators.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief A non-negative real number &phi; representing an inclination (angle from the positive z axis).
   * \details 0&le;&phi;&le;&phi;<sub>down</sub>, where &phi;<sub>down</sub> is a positive real number
   * representing down. The horizon will be ½&phi;<sub>down</sub>.
   * inclination::Radians (&phi;<sub>down</sub>=&pi;) and inclination::Degrees (&phi;<sub>down</sub>=180;) are predefined.
   * \tparam Down a \ref values::fixed "fixed value" representing the down direction. This must be positive.
   */
#ifdef __cpp_concepts
  template<values::fixed Down = values::fixed_pi<long double>> requires
    (values::fixed_number_of_v<Down> > 0) and (not values::complex<Down>) and
    std::convertible_to<values::number_type_of_t<Down>, float>
#else
  template<typename Down = values::fixed_pi<long double>>
#endif
  struct Inclination
  {
#ifndef __cpp_concepts
    static_assert(values::fixed<Down>);
    static_assert(not values::complex<Down>);
    static_assert(values::fixed_number_of_v<Down> > 0);
    static_assert(stdcompat::convertible_to<values::number_type_of_t<Down>, float>);
#endif
  };


  /// Namespace for definitions relating to specialized instances of \ref Inclination.
  namespace inclination
  {
    /// An inclination measured in radians [0,&pi;].
    using Radians = Inclination<>;


    /// An inclination measured in degrees [0,180].
    using Degrees = Inclination<values::Fixed<long double, 180>>;


    namespace detail
    {
      template<typename T>
      struct is_inclination : std::false_type {};

      template<typename Down>
      struct is_inclination<Inclination<Down>> : std::true_type {};
    }


    /**
     * \brief T is a \ref coordinates::pattern object representing an inclination.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept inclination =
#else
    static constexpr bool inclination =
#endif
      detail::is_inclination<T>::value;

  }

}


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Inclination.
   */
  template<typename Down>
  struct coordinate_descriptor_traits<coordinates::Inclination<Down>>
  {
  private:

    using T = coordinates::Inclination<Down>;
    static constexpr auto down = values::fixed_number_of_v<Down>;


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


    static constexpr auto hash_code = [](const T&) -> std::size_t
    {
      auto a = static_cast<float>(down);
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0x8CE6_uz + static_cast<std::size_t>(a * a * 0x1.p2f);
      else if constexpr (bits < 64) return 0x8CE6267E_uz + static_cast<std::size_t>(a * a * 0x1.p4f);
      else return 0x8CE6267E341642F7_uz + static_cast<std::size_t>(a * a * 0x1.p8f);
    };


    /**
     * \brief Maps an inclination to x and y coordinates on quadrants I or II of a unit circle.
     * \details The inclination angle always corresponds to (z,w) coordinates in quadrants I or II
     * of a unit circle of directional-statistics space.
     */
    static constexpr auto
    to_stat_space = [](const T&, auto&& data_view)
    {
      decltype(auto) i = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, 0>{});
      using R = values::real_type_of_t<values::real_type_of_t<decltype(i)>>;
      auto theta = [](auto&& i)
      {
        if constexpr (down == stdcompat::numbers::pi_v<R>) //< Avoid scaling, if possible.
        {
          return std::forward<decltype(i)>(i);
        }
        else
        {
          constexpr auto scale = values::operation(std::divides{}, values::fixed_pi<R>{}, values::cast_to<R>(Down{}));
          return values::operation(std::multiplies{}, std::forward<decltype(i)>(i), scale);
        }
      }(std::forward<decltype(i)>(i));
      auto w = values::sin(theta);
      auto pos_w = values::internal::update_real_part(std::move(w), values::abs(values::real(w)));
      return make_range(values::cos(std::move(theta)), std::move(pos_w));
    };


    /**
     * \brief Maps x and y coordinates on quadrants I or II of a unit circle back to an inclination angle.
     * \details The inclination angle corresponds to (z,w) coordinates directional-statistics space.
     * This does not perform bounds checking to ensure that the angle is in quadrants I or II of the z-w plane.
     */
    static constexpr auto
    from_stat_space = [](const T&, auto&& data_view)
    {
      decltype(auto) z = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, 0>{});
      decltype(auto) w = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, 1>{});
      using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;
      auto pos_w = values::internal::update_real_part(std::forward<decltype(w)>(w), values::abs(values::real(w)));
      if constexpr (down == stdcompat::numbers::pi_v<R>) //< avoid scaling, if possible
      {
        return std::array {values::atan2(std::move(pos_w), std::forward<decltype(z)>(z))};
      }
      else
      {
        constexpr auto scale = values::operation(std::divides{}, values::cast_to<R>(Down{}), values::fixed_pi<R>{});
        return std::array {values::operation(std::multiplies{}, values::atan2(std::move(pos_w), std::forward<decltype(z)>(z)), scale)};
      }
    };

  private:

    struct wrap_theta
    {
      template<typename R>
      constexpr R operator()(const R& theta_real) const
      {
        if (theta_real >= R{0} and theta_real <= R{down}) return theta_real;
        constexpr R down2 = R{down * 2};
        auto am = values::fmod(values::abs(theta_real), down2);
        if (am > R{down}) return down2 - am;
        else return am;
      }
    };

  public:

    /**
     * \brief Wrap the inclination to its primary range.
     */
    static constexpr auto
    wrap = [](const T&, auto&& data_view)
    {
      decltype(auto) i = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, 0>{});
      return std::array {values::internal::update_real_part(std::forward<decltype(i)>(i),
        values::operation(wrap_theta{}, values::real(values::real(i))))};
    };

  };


}


namespace std
{
  template<typename Down1, typename Down2>
  struct common_type<OpenKalman::coordinates::Inclination<Down1>, OpenKalman::coordinates::Inclination<Down2>>
    : std::conditional<
      OpenKalman::values::fixed_number_of_v<Down1> == OpenKalman::values::fixed_number_of_v<Down2>,
      OpenKalman::coordinates::Inclination<Down1>,
      OpenKalman::coordinates::Any<>> {};


  template<typename Down, typename Scalar>
  struct common_type<OpenKalman::coordinates::Inclination<Down>, OpenKalman::coordinates::Any<Scalar>>
    : common_type<OpenKalman::coordinates::Any<Scalar>, OpenKalman::coordinates::Inclination<Down>> {};


  template<typename Down, typename T>
  struct common_type<OpenKalman::coordinates::Inclination<Down>, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdcompat::type_identity<OpenKalman::coordinates::Any<>>,
      std::monostate> {};
}

#endif
