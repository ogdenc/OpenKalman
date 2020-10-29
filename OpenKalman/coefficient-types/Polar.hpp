/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_POLAR_H
#define OPENKALMAN_POLAR_H


namespace OpenKalman
{
  template<typename Coefficient1 = Distance, typename Coefficient2 = Angle>
  struct Polar;


  namespace detail
  {
    template<typename AngleTraits, typename Coefficient, typename Scalar>
    struct PolarImpl;

    template<typename AngleTraits, typename Scalar>
    static inline Scalar polar_angle_wrap_impl(const bool distance_is_negative, const Scalar s)
    {
      constexpr Scalar wrap_max = AngleTraits::template wrap_max<Scalar>;
      constexpr Scalar wrap_min = AngleTraits::template wrap_min<Scalar>;
      constexpr Scalar period = wrap_max - wrap_min;

      Scalar a = distance_is_negative ? s + period * 0.5 : s;

      if (a >= wrap_min and a < wrap_max) // Check if the angle doesn't need wrapping.
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
    }

    template<typename AngleTraits, typename Scalar>
    struct PolarImpl<AngleTraits, Distance, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        to_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
          return get_coeff(i + d_i);
        }};

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr {
          return std::abs(get_coeff(i + d_i));
        }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get =
        {
          [](const GetCoeff& get_coeff) { return std::abs(get_coeff(i + d_i)); }
        };

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff& get_coeff) {
            set_coeff(std::abs(s), i + d_i);
            const auto a = get_coeff(i + a_i);
            set_coeff(polar_angle_wrap_impl<AngleTraits>(std::signbit(s), a), i + a_i); // May need to reflect angle.
          }
        };

    };

    template<typename AngleTraits, typename Scalar>
    struct PolarImpl<AngleTraits, Circle<AngleTraits>, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

      static constexpr Scalar cf =
        2 * M_PI / (AngleTraits::template wrap_max<Scalar> - AngleTraits::template wrap_min<Scalar>);

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 2>
        to_Euclidean_array =
        {
          [](const GetCoeff& get_coeff) { return std::cos(get_coeff(i + a_i) * cf); },
          [](const GetCoeff& get_coeff) { return std::sin(get_coeff(i + a_i) * cf); }
        };

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff)
      {
        const auto x = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + x_i) : get_coeff(i + x_i);
        const auto y = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + y_i) : get_coeff(i + y_i);
        return std::atan2(y, x) / cf;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array_get = {[](const GetCoeff& get_coeff)
      {
        return polar_angle_wrap_impl<AngleTraits>(std::signbit(get_coeff(i + d_i)), get_coeff(i + a_i));
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff&, const GetCoeff&), 1>
        wrap_array_set =
        {
          [](const Scalar s, const SetCoeff& set_coeff, const GetCoeff&) {
            set_coeff(polar_angle_wrap_impl<AngleTraits>(false, s), i + a_i); // Assumes that the corresponding distance is positive.
          }
        };

  };

    // Implementation of polar coordinates.
    template<typename AngleTraits, typename Derived, typename Coeff1, typename Coeff2,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
      static constexpr std::size_t size = 2;
      static constexpr std::size_t dimension = 3;
      static constexpr bool axes_only = false;

      /// See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
      /// 18th Int'l Conf. on Information Fusion 1550, 1553 (2015).
      using difference_type = Concatenate<typename Coeff1::difference_type, typename Coeff2::difference_type>;

      template<typename ... Cnew>
      using Prepend = Coefficients<Cnew..., Derived>;

      template<typename ... Cnew>
      using Append = Coefficients<Derived, Cnew ...>;

      template<typename Scalar>
      using GetCoeff = std::function<Scalar(const std::size_t)>;

      template<typename Scalar>
      using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
        to_Euclidean_array = internal::join(
        detail::PolarImpl<AngleTraits, Coeff1, Scalar>::template to_Euclidean_array<i, d_i, a_i>,
        detail::PolarImpl<AngleTraits, Coeff2, Scalar>::template to_Euclidean_array<i, d_i, a_i>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        from_Euclidean_array = internal::join(
        detail::PolarImpl<AngleTraits, Coeff1, Scalar>::template from_Euclidean_array<i, d2_i, x_i, y_i>,
        detail::PolarImpl<AngleTraits, Coeff2, Scalar>::template from_Euclidean_array<i, d2_i, x_i, y_i>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        wrap_array_get = internal::join(
        detail::PolarImpl<AngleTraits, Coeff1, Scalar>::template wrap_array_get<i, d_i, a_i>,
        detail::PolarImpl<AngleTraits, Coeff2, Scalar>::template wrap_array_get<i, d_i, a_i>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
        wrap_array_set = internal::join(
        detail::PolarImpl<AngleTraits, Coeff1, Scalar>::template wrap_array_set<i, d_i, a_i>,
        detail::PolarImpl<AngleTraits, Coeff2, Scalar>::template wrap_array_set<i, d_i, a_i>
      );

    };
  }

  /// Polar coordinates (Radius, Angle).
  template<typename Traits>
  struct Polar<Distance, Circle<Traits>>
    : detail::PolarBase<Traits, Polar<Distance, Circle<Traits>>, Distance, Circle<Traits>, 0, 1,  0, 1, 2> {};

  /// Polar coordinates (Angle, Radius).
  template<typename Traits>
  struct Polar<Circle<Traits>, Distance>
    : detail::PolarBase<Traits, Polar<Circle<Traits>, Distance>, Circle<Traits>, Distance, 1, 0,  2, 0, 1> {};

}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_H
