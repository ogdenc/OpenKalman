/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <cmath>
#include <array>
#include "collections/collections.hpp"
#include "coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "coordinates/functions/internal/get_descriptor_hash_code.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Any.hpp"


namespace OpenKalman::coordinates
{
  /**
   * \brief An atomic \ref coordinates::descriptor reflecting polar coordinates.
   * \details C1 and C2 are coefficients, and must be some combination of Distance and Angle, such as
   * <code>Polar&lt;Distance, angle::Radians&gt; or Polar&lt;angle::Degrees, Distance&gt;</code>.
   * Polar coordinates span two adjacent coefficients in a matrix.
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \tparam C1, C2 Distance and Angle, in either order. By default, they are Distance and angle::Radians, respectively.
   */
  template<typename C1 = Distance, typename C2 = angle::Radians>
#ifdef __cpp_concepts
  requires (std::same_as<C1, Distance> and angle::angle<C2>) or (std::same_as<C2, Distance> and angle::angle<C1>)
#endif
  struct Polar
  {
#ifndef __cpp_concepts
    static_assert((std::is_same_v<C1, Distance> and angle::angle<C2>) or (std::is_same_v<C2, Distance> and angle::angle<C1>));
#endif
  };

}


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename Min, typename Max,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
    private:

      static constexpr auto min = values::fixed_value_of_v<Min>;
      static constexpr auto max = values::fixed_value_of_v<Max>;

    public:

      static constexpr bool is_specialized = true;


      static constexpr auto dimension = [](const T&) { return std::integral_constant<std::size_t, 2>{}; };


      static constexpr auto stat_dimension = [](const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto is_euclidean = [](const T&) { return std::false_type{}; };


      static constexpr auto hash_code = [](const T&)
      {
        constexpr auto a = coordinates::internal::get_descriptor_hash_code(coordinates::Angle<Min, Max>{});
        constexpr auto bits = std::numeric_limits<std::size_t>::digits;
        if constexpr (bits < 32)
          return std::integral_constant<std::size_t, a - 0x97C1_uz + d_i * 0x1000_uz>{};
        else if constexpr (bits < 64)
          return std::integral_constant<std::size_t, a - 0x97C195FE_uz + d_i * 0x10000000_uz>{};
        else
          return std::integral_constant<std::size_t,  a - 0x97C195FEC488C0BC_uz + d_i * 0x1000000000000000_uz>{};
      };

    private:

      struct flip_axis
      {
        template<typename R>
        constexpr auto operator()(bool flip_, R a_) const { return flip_ ? -std::move(a_) : std::move(a_); }
      };


      template<typename...Args>
      static constexpr auto make_range(Args&&...args)
      {
        if constexpr ((... or values::fixed<Args>))
        {
          return std::tuple {std::forward<Args>(args)...};
        }
        else
        {
          using C = std::common_type_t<Args...>;
          return std::array {static_cast<C>(std::forward<Args>(args))...};
        }
      }

    public:

      /**
       * \brief Maps a polar coordinate to coordinates in Euclidean space.
       * \details This function takes a set of polar coordinates and converts them to x, y, and z
       * Cartesian coordinates representing a location on a unit half-cylinder (z is the long axis).
       */
      static constexpr auto
      to_stat_space = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get<d_i>(std::forward<decltype(data_view)>(data_view));
        decltype(auto) a = collections::get<a_i>(std::forward<decltype(data_view)>(data_view));
        using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;
        auto phi = [](auto&& a)
        {
          if constexpr (min == -stdex::numbers::pi_v<R> and max == stdex::numbers::pi_v<R>) //< Avoid scaling, if possible.
          {
            return std::forward<decltype(a)>(a);
          }
          else
          {
            constexpr auto period = values::cast_to<R>(values::operation(std::minus{}, Max{}, Min{}));
            constexpr auto scale = values::operation(std::divides{}, values::fixed_2pi<R>{}, period);
            return values::operation(std::multiplies{}, std::forward<decltype(a)>(a), scale);
          }
        }(std::forward<decltype(a)>(a));

        auto x = values::cos(phi);
        auto y = values::sin(phi);
        auto d_real = values::real(d);
        auto flip = values::signbit(d_real);
        auto d_flip = values::internal::update_real_part(std::forward<decltype(d)>(d), values::abs(d_real));
        auto x_flip = values::internal::update_real_part(std::move(x), values::operation(flip_axis{}, flip, values::real(x)));
        auto y_flip = values::internal::update_real_part(std::move(y), values::operation(flip_axis{}, flip, values::real(y)));
        if constexpr (d2_i == 0)
          return make_range(std::move(d_flip), std::move(x_flip), std::move(y_flip));
        else
          return make_range(std::move(x_flip), std::move(y_flip), std::move(d_flip));
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


      template<typename D, typename Phi>
      static constexpr auto make_ordered_range(D d, Phi phi)
      {
        if constexpr (d_i == 0)
          return make_range(std::move(d), std::move(phi));
        else
          return make_range(std::move(phi), std::move(d));
      }

    public:

      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes x, y, and z Cartesian coordinates representing a location on a
       * unit half-cylinder, and converts them to polar coordinates.
       * \note Although this performs bounds checking on the angle, it does not check to ensure the distance is positive.
       */
      static constexpr auto
      from_stat_space = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get<d2_i>(std::forward<decltype(data_view)>(data_view));
        decltype(auto) x = collections::get<x_i>(std::forward<decltype(data_view)>(data_view));
        decltype(auto) y = collections::get<y_i>(std::forward<decltype(data_view)>(data_view));
        using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;
        if constexpr (min == -stdex::numbers::pi_v<R> and max == stdex::numbers::pi_v<R>) //< Avoid scaling, if possible.
        {
          return make_ordered_range(std::forward<decltype(d)>(d), values::atan2(std::forward<decltype(y)>(y), std::forward<decltype(x)>(x)));
        }
        else
        {
          constexpr auto period = values::cast_to<R>(values::operation(std::minus{}, Max{}, Min{}));
          constexpr auto scale = values::operation(std::divides{}, period, values::fixed_2pi<R>{});
          auto phi = values::operation(std::multiplies{}, values::atan2(std::forward<decltype(y)>(y), std::forward<decltype(x)>(x)), scale);
          return make_ordered_range(std::forward<decltype(d)>(d),
            values::internal::update_real_part(std::move(phi), values::operation(wrap_phi{}, values::real(phi))));
        }
      };

    private:

      struct wrap_phi_mod
      {
        template<typename R>
        constexpr R operator()(const R& d_r, const R& phi_r) const
        {
          constexpr R period = max - min;
          R phi_flip_real = values::signbit(d_r) ? phi_r + period * R{0.5} : phi_r;
          if (phi_flip_real >= R{min} and phi_flip_real < max) return phi_flip_real;
          R phi_real_mod = values::fmod(phi_flip_real - R{min}, period);
          if (phi_real_mod < 0) return R{min} + phi_real_mod + period;
          return R{min} + phi_real_mod;
        }
      };

    public:

      /**
       * \brief Perform modular wrapping of polar coordinates.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       */
      static constexpr auto
      wrap = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get<d_i>(std::forward<decltype(data_view)>(data_view));
        decltype(auto) phi = collections::get<a_i>(std::forward<decltype(data_view)>(data_view));

        auto d_real = values::real(values::real(d));
        auto phi_real = values::real(values::real(phi));

        return make_ordered_range(
          values::internal::update_real_part(std::forward<decltype(d)>(d), values::abs(d_real)),
          values::internal::update_real_part(std::forward<decltype(phi)>(phi), values::operation(wrap_phi_mod{}, std::move(d_real), std::move(phi_real))));
      };

    };

  }


  /**
   * \internal
   * \brief traits for Polar<Distance, Angle>.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Polar<coordinates::Distance, coordinates::Angle<Min, Max>>>
    : detail::PolarBase<coordinates::Polar<coordinates::Distance, coordinates::Angle<Min, Max>>, Min, Max, 0, 1,  0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Polar<Angle, Distance>.
   */
  template<typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Polar<coordinates::Angle<Min, Max>, coordinates::Distance>>
    : detail::PolarBase<coordinates::Polar<coordinates::Angle<Min, Max>, coordinates::Distance>, Min, Max, 1, 0,  2, 0, 1>
  {};

}


namespace std
{
  template<typename Min1, typename Max1, typename Min2, typename Max2>
  struct common_type<
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Distance, OpenKalman::coordinates::Angle<Min1, Max1>>,
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Distance, OpenKalman::coordinates::Angle<Min2, Max2>>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2>,
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Distance, OpenKalman::coordinates::Angle<Min1, Max1>>,
      OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2>
  struct common_type<
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Angle<Min1, Max1>, OpenKalman::coordinates::Distance>,
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Angle<Min2, Max2>, OpenKalman::coordinates::Distance>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2>,
      OpenKalman::coordinates::Polar<OpenKalman::coordinates::Angle<Min1, Max1>, OpenKalman::coordinates::Distance>,
      OpenKalman::coordinates::Any<>> {};


  template<typename C1, typename C2, typename Scalar>
  struct common_type<OpenKalman::coordinates::Polar<C1, C2>, OpenKalman::coordinates::Any<Scalar>>
    : common_type<OpenKalman::coordinates::Any<Scalar>, OpenKalman::coordinates::Polar<C1, C2>> {};


  template<typename C1, typename C2, typename T>
  struct common_type<OpenKalman::coordinates::Polar<C1, C2>, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdex::type_identity<OpenKalman::coordinates::Any<>>,
      std::monostate> {};
}


#endif
