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
#include <typeinfo>
#include <cmath>
#include <array>
#include "collections/collections.hpp"
#include "coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "coordinates/functions/internal/get_descriptor_hash_code.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"
#include "Any.hpp"


namespace OpenKalman::coordinates
{
  /**
   * \brief A \ref coordinates::descriptor reflecting spherical coordinates according to the ISO 80000-2:2019 convention.
   * \details C1, C2, and C3 must be some combination of Distance (radius r), Inclination (polar angle θ), and
   * Angle (azimuth angle φ) in any order. The coordinate system is right-handed.
   * Spherical coordinates span three adjacent coefficients in a matrix.<br/>
   * See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
   * 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
   * \par Examples
   * <code>Spherical&lt;Distance, inclination::Degrees, angle::Radians&gt;,<br/>
   * Spherical&lt;angle::PositiveDegrees, Distance, inclination::Radians&gt;</code>
   * \tparam C1, C2, C3 Distance, Inclination, and Angle, in any order.
   * By default, they are Distance, inclination::Radians, and angle::Radians, respectively.
   */
  template<typename C1 = Distance, typename C2 = inclination::Radians, typename C3 = angle::Radians>
#ifdef __cpp_concepts
  requires
    (std::same_as<C1, Distance> and inclination::inclination<C2> and angle::angle<C3>) or
    (std::same_as<C1, Distance> and inclination::inclination<C3> and angle::angle<C2>) or
    (std::same_as<C2, Distance> and inclination::inclination<C1> and angle::angle<C3>) or
    (std::same_as<C2, Distance> and inclination::inclination<C3> and angle::angle<C1>) or
    (std::same_as<C3, Distance> and inclination::inclination<C1> and angle::angle<C2>) or
    (std::same_as<C3, Distance> and inclination::inclination<C2> and angle::angle<C1>)
#endif
  struct Spherical
  {
#ifndef __cpp_concepts
    static_assert(
      (std::is_same_v<C1, Distance> and inclination::inclination<C2> and angle::angle<C3>) or
      (std::is_same_v<C1, Distance> and inclination::inclination<C3> and angle::angle<C2>) or
      (std::is_same_v<C2, Distance> and inclination::inclination<C1> and angle::angle<C3>) or
      (std::is_same_v<C2, Distance> and inclination::inclination<C3> and angle::angle<C1>) or
      (std::is_same_v<C3, Distance> and inclination::inclination<C1> and angle::angle<C2>) or
      (std::is_same_v<C3, Distance> and inclination::inclination<C2> and angle::angle<C1>));
#endif
  };

}


namespace OpenKalman::interface
{
  namespace detail
  {
    // Implementation of polar coordinates.
    template<typename T, typename Min, typename Max, typename Down, std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
    private:

      static constexpr auto min = values::fixed_value_of_v<Min>;
      static constexpr auto max = values::fixed_value_of_v<Max>;
      static constexpr auto down = values::fixed_value_of_v<Down>;
      static constexpr std::size_t d2_i = 0, x_i = 1, y_i = 2, z_i = 3;

    public:

      static constexpr bool is_specialized = true;


      static constexpr auto dimension = [](const T&) { return std::integral_constant<std::size_t, 3>{}; };


      static constexpr auto stat_dimension = [](const T&) { return std::integral_constant<std::size_t, 4>{}; };


      static constexpr auto is_euclidean = [](const T&) { return std::false_type{}; };


      static constexpr auto hash_code = [](const T&) -> std::size_t
      {
        constexpr auto a = coordinates::internal::get_descriptor_hash_code(coordinates::Angle<Min, Max>{});
        constexpr auto b = coordinates::internal::get_descriptor_hash_code(coordinates::Inclination<Down>{});
        constexpr std::size_t f = (a_i * 2 + i_i * 3 - 2);
        constexpr auto bits = std::numeric_limits<std::size_t>::digits;
        if constexpr (bits < 32)
          return std::integral_constant<std::size_t, (a - 0x83B0_uz) + (b - 0x83B0_uz) + f * 0x1000_uz>{};
        else if constexpr (bits < 64)
          return std::integral_constant<std::size_t, (a - 0x83B023AB_uz) + (b - 0x83B023AB_uz) + f * 0x10000000_uz>{};
        else
          return std::integral_constant<std::size_t, (a - 0x83B023AB3EEFB99A_uz) + (b - 0x83B023AB3EEFB99A_uz) + f * 0x1000000000000000_uz>{};
      };

    private:

      struct flip
      {
        template<typename R>
        constexpr std::decay_t<R> operator()(bool f, R&& r) const
        {
          if (f)
            return values::internal::update_real_part(std::forward<R>(r), -values::real(values::real(r)));
          else
            return std::forward<R>(r);
        }
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
       * \brief Maps an element to coordinates in Euclidean space.
       * \details This function takes a set of spherical coordinates and converts them to d, x, y, and z
       * Cartesian coordinates representing a location on a unit 4D half-cylinder.
       */
      static constexpr auto
      to_stat_space = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, d_i>{});
        decltype(auto) i = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, i_i>{});
        decltype(auto) a = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, a_i>{});

        using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;

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

        auto phi = [](auto&& a)
        {
          if constexpr (min == -stdcompat::numbers::pi_v<R> and max == stdcompat::numbers::pi_v<R>) //< Avoid scaling, if possible.
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

        auto x = values::operation(std::multiplies{}, values::sin(theta), values::cos(phi));
        auto y = values::operation(std::multiplies{}, values::sin(theta), values::sin(phi));
        auto z = values::cos(theta);

        auto f = values::signbit(values::real(d));

        return make_range(
          values::operation(flip{}, f, std::forward<decltype(d)>(d)),
          values::operation(flip{}, f, std::move(x)),
          values::operation(flip{}, f, std::move(y)),
          values::operation(flip{}, f, std::move(z)));
      };

    private:

      struct calc_theta
      {
        template<typename R>
        constexpr auto operator()(R h, const R& z) const
        {
          return (R{down} / stdcompat::numbers::pi_v<R>) * (h == R{0} ? std::move(h) : values::acos(z/h));
        }
      };


      struct calc_phi
      {
        template<typename R>
        constexpr auto operator()(R hypot_xy, const R& x, const R& signed_scale) const
        {
          return signed_scale * (hypot_xy == R{0} ? std::move(hypot_xy) : values::acos(x / hypot_xy));
        }
      };


      struct wrap_phi
      {
        template<typename R>
        constexpr auto operator()(const R& phi_real) const
        {
          constexpr R period = max - min;
          if (phi_real < R{min}) return phi_real + period;
          if (phi_real >= R{max}) return phi_real - period;
          return phi_real;
        }
      };


      template<typename D, typename I, typename A>
      static constexpr auto make_ordered_range(D d, I i, A a)
      {
        if constexpr (d_i == 0 and i_i == 1 and a_i == 2)
          return make_range(std::move(d), std::move(i), std::move(a));
        else if constexpr (d_i == 0 and a_i == 1 and i_i == 2)
          return make_range(std::move(d), std::move(a), std::move(i));
        else if constexpr (i_i == 0 and d_i == 1 and a_i == 2)
          return make_range(std::move(i), std::move(d), std::move(a));
        else if constexpr (a_i == 0 and d_i == 1 and i_i == 2)
          return make_range(std::move(a), std::move(d), std::move(i));
        else if constexpr (i_i == 0 and a_i == 1 and d_i == 2)
          return make_range(std::move(i), std::move(a), std::move(d));
        else // if constexpr (a_i == 0 and i_i == 1 and d_i == 2)
          return make_range(std::move(a), std::move(i), std::move(d));
      }

    public:

      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \details This function takes d, x, y, and z Cartesian coordinates representing a location on a
       * 4D unit half-cylinder, and converts them to spherical coordinates.
       */
      static constexpr auto
      from_stat_space = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, d2_i>{});
        decltype(auto) x = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, x_i>{});
        decltype(auto) y = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, y_i>{});
        decltype(auto) z = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, z_i>{});

        using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;

        constexpr auto period = values::cast_to<R>(values::operation(std::minus{}, Max{}, Min{}));
        constexpr auto scale = values::operation(std::divides{}, period, values::fixed_2pi<R>{});

        auto signed_scale = values::copysign(scale, values::real(y));
        if constexpr (values::fixed<decltype(d)> and values::fixed<decltype(x)> and values::fixed<decltype(y)> and values::fixed<decltype(z)>)
          static_assert(values::fixed<decltype(signed_scale)>);
        auto hypot_xy = values::hypot(x, std::forward<decltype(y)>(y));
        auto x2y2 = values::operation(std::multiplies{}, hypot_xy, hypot_xy);
        auto z2 = values::operation(std::multiplies{}, z, z);
        auto h = values::sqrt(values::operation(std::plus{}, x2y2, z2));

        auto theta = values::operation(calc_theta{}, std::move(h), z);
        auto phi = values::operation(calc_phi{}, std::move(hypot_xy), x, signed_scale);

        return make_ordered_range(
          std::forward<decltype(d)>(d),
          std::move(theta),
          values::internal::update_real_part(std::move(phi), values::operation(wrap_phi{}, values::real(phi))));
      };

    private:

      struct wrap_theta
      {
        template<typename R>
        constexpr auto operator()(bool flip_d, R i_a) const
        {
          constexpr R half_period {down};
          if (i_a > half_period) return flip_d ? i_a - half_period : R{down * 2} - i_a;
          return flip_d ? half_period - i_a : std::move(i_a);
        }
      };


      struct wrap_phi_mod
      {
        template<typename R>
        constexpr auto operator()(bool flip_a, R a_r) const
        {
          constexpr R period = max - min;
          R a_real = flip_a ? a_r + period * R{0.5} : std::move(a_r);
          auto ar = values::fmod(a_real - R{min}, period);
          return (ar < 0 ? ar + period : std::move(ar)) + R{min};
        }
      };

    public:

      /**
       * \brief Perform modular wrapping of spherical coordinates.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       */
      static constexpr auto
      wrap = [](const T&, auto&& data_view)
      {
        decltype(auto) d = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, d_i>{});
        decltype(auto) i = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, i_i>{});
        decltype(auto) a = collections::get(std::forward<decltype(data_view)>(data_view), std::integral_constant<std::size_t, a_i>{});

        using R = values::real_type_of_t<values::real_type_of_t<collections::common_collection_type_t<decltype(data_view)>>>;
        auto d_real = values::real(values::real(d));
        auto flip_d = values::signbit(d_real);
        auto d_new = values::internal::update_real_part(std::forward<decltype(d)>(d), values::abs(d_real));

        constexpr auto i_period = values::operation(std::multiplies{}, values::cast_to<R>(Down{}), values::fixed_value<R, 2>{});
        auto i_real = values::real(values::real(i));
        auto im = values::fmod(i_real, i_period);
        auto iabs = values::abs(im);
        auto high_i = values::operation(std::greater{}, iabs, values::cast_to<R>(Down{}));
        auto flip_i = values::operation(std::not_equal_to{}, values::signbit(im), high_i);
        auto i_new = values::internal::update_real_part(std::forward<decltype(i)>(i), values::operation(wrap_theta{}, flip_d, std::move(iabs)));

        auto flip_a = values::operation(std::not_equal_to{}, flip_i, flip_d);
        auto aw = values::operation(wrap_phi_mod{}, flip_a, values::real(values::real(a)));
        auto a_new = values::internal::update_real_part(std::forward<decltype(a)>(a), aw);

        return make_ordered_range(std::move(d_new), std::move(i_new), std::move(a_new));
      };

    };

  }


  /**
   * \internal
   * \brief traits for Spherical<Distance, Inclination, Angle>.
   */
  template<typename Down, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Distance, coordinates::Inclination<Down>, coordinates::Angle<Min, Max>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Distance, coordinates::Inclination<Down>, coordinates::Angle<Min, Max>>, Min, Max, Down, 0, 2, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Distance, Angle, Inclination>.
   */
  template<typename Min, typename Max, typename Down>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Distance, coordinates::Angle<Min, Max>, coordinates::Inclination<Down>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Distance, coordinates::Angle<Min, Max>, coordinates::Inclination<Down>>, Min, Max, Down, 0, 1, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Distance, Inclination>.
   */
  template<typename Min, typename Max, typename Down>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Distance, coordinates::Inclination<Down>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Distance, coordinates::Inclination<Down>>, Min, Max, Down, 1, 0, 2>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Distance, Angle>.
   */
  template<typename Down, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Inclination<Down>, coordinates::Distance, coordinates::Angle<Min, Max>>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Inclination<Down>, coordinates::Distance, coordinates::Angle<Min, Max>>, Min, Max, Down, 1, 2, 0>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Angle, Inclination, Distance>.
   */
  template<typename Min, typename Max, typename Down>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Inclination<Down>, coordinates::Distance>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Angle<Min, Max>, coordinates::Inclination<Down>, coordinates::Distance>, Min, Max, Down, 2, 0, 1>
  {};


  /**
   * \internal
   * \brief traits for Spherical<Inclination, Angle, Distance>.
   */
  template<typename Down, typename Min, typename Max>
  struct coordinate_descriptor_traits<coordinates::Spherical<coordinates::Inclination<Down>, coordinates::Angle<Min, Max>, coordinates::Distance>>
    : detail::SphericalBase<coordinates::Spherical<coordinates::Inclination<Down>, coordinates::Angle<Min, Max>, coordinates::Distance>, Min, Max, Down, 2, 1, 0>
  {};


}


namespace std
{
  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Inclination<Down1>,
        OpenKalman::coordinates::Angle<Min1, Max1>>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Inclination<Down2>,
        OpenKalman::coordinates::Angle<Min2, Max2>>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Distance,
          OpenKalman::coordinates::Inclination<Down1>,
          OpenKalman::coordinates::Angle<Min1, Max1>>,
        OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Angle<Min1, Max1>,
        OpenKalman::coordinates::Inclination<Down1>>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Angle<Min2, Max2>,
        OpenKalman::coordinates::Inclination<Down2>>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Distance,
          OpenKalman::coordinates::Angle<Min1, Max1>,
          OpenKalman::coordinates::Inclination<Down1>>,
        OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Inclination<Down1>,
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Angle<Min1, Max1>>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Inclination<Down2>,
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Angle<Min2, Max2>>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Inclination<Down1>,
          OpenKalman::coordinates::Distance,
          OpenKalman::coordinates::Angle<Min1, Max1>>,
        OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Inclination<Down1>,
        OpenKalman::coordinates::Angle<Min1, Max1>,
        OpenKalman::coordinates::Distance>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Inclination<Down2>,
        OpenKalman::coordinates::Angle<Min2, Max2>,
        OpenKalman::coordinates::Distance>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Inclination<Down1>,
          OpenKalman::coordinates::Angle<Min1, Max1>,
          OpenKalman::coordinates::Distance>,
        OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Angle<Min1, Max1>,
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Inclination<Down1>>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Angle<Min2, Max2>,
        OpenKalman::coordinates::Distance,
        OpenKalman::coordinates::Inclination<Down2>>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Angle<Min1, Max1>,
          OpenKalman::coordinates::Distance,
          OpenKalman::coordinates::Inclination<Down1>>,
        OpenKalman::coordinates::Any<>> {};


  template<typename Min1, typename Max1, typename Min2, typename Max2, typename Down1, typename Down2>
  struct common_type<
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Angle<Min1, Max1>,
        OpenKalman::coordinates::Inclination<Down1>,
        OpenKalman::coordinates::Distance>,
      OpenKalman::coordinates::Spherical<
        OpenKalman::coordinates::Angle<Min2, Max2>,
        OpenKalman::coordinates::Inclination<Down2>,
        OpenKalman::coordinates::Distance>>
    : std::conditional<
      OpenKalman::values::fixed_value_of_v<Min1> == OpenKalman::values::fixed_value_of_v<Min2> and
          OpenKalman::values::fixed_value_of_v<Max1> == OpenKalman::values::fixed_value_of_v<Max2> and
          OpenKalman::values::fixed_value_of_v<Down1> == OpenKalman::values::fixed_value_of_v<Down2>,
        OpenKalman::coordinates::Spherical<
          OpenKalman::coordinates::Angle<Min1, Max1>,
          OpenKalman::coordinates::Inclination<Down1>,
          OpenKalman::coordinates::Distance>,
        OpenKalman::coordinates::Any<>> {};


  template<typename C1, typename C2, typename C3, typename Scalar>
  struct common_type<OpenKalman::coordinates::Spherical<C1, C2, C3>, OpenKalman::coordinates::Any<Scalar>>
    : common_type<OpenKalman::coordinates::Any<Scalar>, OpenKalman::coordinates::Spherical<C1, C2, C3>> {};


  template<typename C1, typename C2, typename C3, typename T>
  struct common_type<OpenKalman::coordinates::Spherical<C1, C2, C3>, T>
    : std::conditional_t<
      OpenKalman::coordinates::descriptor<T>,
      OpenKalman::stdcompat::type_identity<OpenKalman::coordinates::Any<>>,
      std::monostate> {};
}

#endif
