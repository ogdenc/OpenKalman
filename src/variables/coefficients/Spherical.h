/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SPHERICAL_H
#define OPENKALMAN_SPHERICAL_H


namespace OpenKalman
{
  template<typename Coefficient1 = Distance, typename Coefficient2 = Angle, typename Coefficient3 = InclinationAngle>
  struct Spherical;

  namespace detail
  {
    template<typename Coefficient, typename CircleTraits, typename InclinationTraits, typename Scalar>
    struct SphericalImpl;

    template<typename CircleTraits, typename InclinationTraits, typename Scalar>
    struct SphericalImpl<OpenKalman::Distance, CircleTraits, InclinationTraits, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr {return std::abs(get_coeff(i + d_i)); }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array = {[](const GetCoeff& get_coeff) constexpr { return std::abs(get_coeff(i + d_i)); }};
    };

    template<typename Scalar, typename InclinationTraits>
    static constexpr std::tuple<Scalar, bool>
    wrap_inclination(const Scalar a)
    {
      constexpr Scalar period = InclinationTraits::template wrap_max<Scalar> - InclinationTraits::template wrap_min<Scalar>;
      constexpr Scalar wrap_mod = 0.5 * period;
      constexpr Scalar upward_angle = 0.25 * period;
      if (a >= -upward_angle and a < upward_angle) // A shortcut, for the easy case.
      {
        return { a, false };
      }
      else
      {
        Scalar ar = std::fmod(a + upward_angle, period);
        if (ar < 0) ar += period;
        if (ar > wrap_mod)
        {
          // Do a mirror reflection about vertical axis.
          return { 0.75 * period - ar, true };
        }
        else
        {
          return { ar - upward_angle, false };
        }
      }
    }

    template<typename CircleTraits, typename InclinationTraits, typename Scalar>
    struct SphericalImpl<OpenKalman::Circle<CircleTraits>, CircleTraits, InclinationTraits, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      static constexpr Scalar cf_cir = 2 * M_PI / (CircleTraits::template wrap_max<Scalar> - CircleTraits::template wrap_min<Scalar>);

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
        const auto x = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + x_i) : get_coeff(i + x_i);
        const auto y = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + y_i) : get_coeff(i + y_i);
        return std::atan2(y, x) / cf_cir;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array = {[](const GetCoeff& get_coeff) constexpr {
        constexpr Scalar wrap_max = CircleTraits::template wrap_max<Scalar>;
        constexpr Scalar wrap_min = CircleTraits::template wrap_min<Scalar>;
        constexpr Scalar period = wrap_max - wrap_min;
        Scalar a = get_coeff(i + a_i);
        if (std::get<1>(wrap_inclination<Scalar, InclinationTraits>(get_coeff(i + i_i))) != std::signbit(get_coeff(i + d_i)))
        {
          a += period * 0.5; // Reflect azimuth across the origin.
        }
        if (a >= wrap_min and a < wrap_max) // Check if angle doesn't need wrapping.
        {
          return a;
        }
        else // Wrap the angle.
        {
          Scalar ar = std::fmod(a - wrap_min, period);
          if (ar < 0)
          {
            ar += period;
          }
          return ar + wrap_min;
        }
      }};
    };

    template<typename CircleTraits, typename InclinationTraits, typename Scalar>
    struct SphericalImpl<OpenKalman::Inclination<InclinationTraits>, CircleTraits, InclinationTraits, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      static constexpr Scalar cf_inc = 2 * M_PI / (InclinationTraits::template wrap_max<Scalar> - InclinationTraits::template wrap_min<Scalar>);

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i, std::size_t z_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
        const auto r = std::hypot(get_coeff(i + x_i), get_coeff(i + y_i), get_coeff(i + z_i));
        const auto ret = std::asin(get_coeff(i + z_i) / r) / cf_inc;
        if (std::isnan(ret)) return 0.; // Avoid NAN when all coefficients are zero.
        else return std::signbit(get_coeff(i + d_i)) ? -ret : ret;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i, std::size_t i_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array =  {[](const GetCoeff& get_coeff) constexpr {
        const Scalar ar = std::get<0>(wrap_inclination<Scalar, InclinationTraits>(get_coeff(i + i_i)));
        return std::signbit(get_coeff(i + d_i)) ? -ar : ar;
      }};
    };

    // Implementation of polar coordinates.
    template<typename Derived,
      typename Coeff1, typename Coeff2, typename Coeff3,
      typename CircleTraits, typename InclinationTraits,
      std::size_t d_i, std::size_t a_i, std::size_t i_i>
    struct SphericalBase
    {
      static constexpr std::size_t size = 3;
      static constexpr std::size_t dimension = 4;
      static constexpr bool axes_only = false;

      template<typename ... Cnew>
      using Prepend = Coefficients<Cnew..., Derived>;

      template<typename ... Cnew>
      using Append = Coefficients<Derived, Cnew ...>;

      template<typename Scalar>
      using GetCoeff = std::function<Scalar(const std::size_t)>;

      template<typename Scalar>
      static constexpr Scalar cf_cir = 2 * M_PI / (CircleTraits::template wrap_max<Scalar> - CircleTraits::template wrap_min<Scalar>);

      template<typename Scalar>
      static constexpr Scalar cf_inc = 2 * M_PI / (InclinationTraits::template wrap_max<Scalar> - InclinationTraits::template wrap_min<Scalar>);

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
        to_Euclidean_array = {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return get_coeff(i + d_i); },
        [](const GetCoeff<Scalar>& get_coeff) constexpr {
          return std::cos(get_coeff(i + a_i) * cf_cir<Scalar>) * std::cos(get_coeff(i + i_i) * cf_inc<Scalar>);
        },
        [](const GetCoeff<Scalar>& get_coeff) constexpr {
          return std::sin(get_coeff(i + a_i) * cf_cir<Scalar>) * std::cos(get_coeff(i + i_i) * cf_inc<Scalar>);
        },
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::sin(get_coeff(i + i_i) * cf_inc<Scalar>); }
      };

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        from_Euclidean_array = internal::join(
        internal::join(
          detail::SphericalImpl<Coeff1, CircleTraits, InclinationTraits, Scalar>::template from_Euclidean_array<i, 0, 1, 2, 3>,
          detail::SphericalImpl<Coeff2, CircleTraits, InclinationTraits, Scalar>::template from_Euclidean_array<i, 0, 1, 2, 3>),
          detail::SphericalImpl<Coeff3, CircleTraits, InclinationTraits, Scalar>::template from_Euclidean_array<i, 0, 1, 2, 3>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        wrap_array = internal::join(
        internal::join(
          detail::SphericalImpl<Coeff1, CircleTraits, InclinationTraits, Scalar>::template wrap_array<i, d_i, a_i, i_i>,
          detail::SphericalImpl<Coeff2, CircleTraits, InclinationTraits, Scalar>::template wrap_array<i, d_i, a_i, i_i>),
          detail::SphericalImpl<Coeff3, CircleTraits, InclinationTraits, Scalar>::template wrap_array<i, d_i, a_i, i_i>
      );
    };
  }

  /// Spherical coordinates (Radius, Angle, Inclination).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Distance, OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>>
    : detail::SphericalBase<Spherical<OpenKalman::Distance, OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>>,
    OpenKalman::Distance, OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>, Traits1, Traits2, 0, 1, 2> {};

  /// Spherical coordinates (Radius, Inclination, Angle).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Distance, OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>>
    : detail::SphericalBase<Spherical<OpenKalman::Distance, OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>>,
    OpenKalman::Distance, OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>, Traits2, Traits1, 0, 2, 1> {};

  /// Spherical coordinates (Angle, Radius, Inclination).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Circle<Traits1>, OpenKalman::Distance, OpenKalman::Inclination<Traits2>>
    : detail::SphericalBase<Spherical<OpenKalman::Circle<Traits1>, OpenKalman::Distance, OpenKalman::Inclination<Traits2>>,
      OpenKalman::Circle<Traits1>, OpenKalman::Distance, OpenKalman::Inclination<Traits2>, Traits1, Traits2, 1, 0, 2> {};

  /// Spherical coordinates (Inclination, Radius, Angle).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Inclination<Traits1>, OpenKalman::Distance, OpenKalman::Circle<Traits2>>
    : detail::SphericalBase<Spherical<OpenKalman::Inclination<Traits1>, OpenKalman::Distance, OpenKalman::Circle<Traits2>>,
      OpenKalman::Inclination<Traits1>, OpenKalman::Distance, OpenKalman::Circle<Traits2>, Traits2, Traits1, 1, 2, 0> {};

  /// Spherical coordinates (Angle, Inclination, Radius).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>, OpenKalman::Distance>
    : detail::SphericalBase<Spherical<OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>, OpenKalman::Distance>,
      OpenKalman::Circle<Traits1>, OpenKalman::Inclination<Traits2>, OpenKalman::Distance, Traits1, Traits2, 2, 0, 1> {};

  /// Spherical coordinates (Inclination, Angle, Radius).
  template<typename Traits1, typename Traits2>
  struct Spherical<OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>, OpenKalman::Distance>
    : detail::SphericalBase<Spherical<OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>, OpenKalman::Distance>,
      OpenKalman::Inclination<Traits1>, OpenKalman::Circle<Traits2>, OpenKalman::Distance, Traits2, Traits1, 2, 1, 0> {};

  /// Spherical is aa coefficient.
  template<typename Coeff1, typename Coeff2, typename Coeff3>
  struct is_coefficient<Spherical<Coeff1, Coeff2, Coeff3>> : std::true_type {};

  template<typename Coeff1a, typename Coeff2a, typename Coeff3a, typename Coeff1b, typename Coeff2b, typename Coeff3b>
  struct is_equivalent<Spherical<Coeff1a, Coeff2a, Coeff3a>, Spherical<Coeff1b, Coeff2b, Coeff3b>>
    : std::integral_constant<bool, is_equivalent_v<Coeff1a, Coeff1b> and
      is_equivalent_v<Coeff2a, Coeff2b> and is_equivalent_v<Coeff3a, Coeff3b>> {};

  namespace internal
  {
    template<typename Coeff1, typename Coeff2, typename Coeff3, typename...Coeffs>
    struct ConcatenateImpl<Spherical<Coeff1, Coeff2, Coeff3>, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Spherical<Coeff1, Coeff2, Coeff3>>;
    };
  }


}// namespace OpenKalman

#endif //OPENKALMAN_SPHERICAL_H
