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
    template<typename Coefficient, typename Scalar>
    struct PolarImpl;

    template<typename Scalar>
    struct PolarImpl<OpenKalman::Distance, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        to_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr { return get_coeff(i + d_i); }};

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr { return std::abs(get_coeff(i + d_i)); }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array = {[](const GetCoeff& get_coeff) constexpr { return std::abs(get_coeff(i + d_i)); }};
    };

    template<typename Traits, typename Scalar>
    struct PolarImpl<OpenKalman::Circle<Traits>, Scalar>
    {
      using GetCoeff = std::function<Scalar(const std::size_t)>;
      static constexpr Scalar cf = 2 * M_PI / (Traits::template wrap_max<Scalar> - Traits::template wrap_min<Scalar>);

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 2>
        to_Euclidean_array =
        {
          [](const GetCoeff& get_coeff) constexpr { return std::cos(get_coeff(i + a_i) * cf); },
          [](const GetCoeff& get_coeff) constexpr { return std::sin(get_coeff(i + a_i) * cf); }
        };

      template<std::size_t i, std::size_t d_i, std::size_t x_i, std::size_t y_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        from_Euclidean_array = {[](const GetCoeff& get_coeff) constexpr
      {
        const auto x = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + x_i) : get_coeff(i + x_i);
        const auto y = std::signbit(get_coeff(i + d_i)) ? -get_coeff(i + y_i) : get_coeff(i + y_i);
        return std::atan2(y, x) / cf;
      }};

      template<std::size_t i, std::size_t d_i, std::size_t a_i>
      static constexpr std::array<Scalar (*const)(const GetCoeff&), 1>
        wrap_array = {[](const GetCoeff& get_coeff) constexpr
      {
        constexpr Scalar wrap_max = Traits::template wrap_max<Scalar>;
        constexpr Scalar wrap_min = Traits::template wrap_min<Scalar>;
        constexpr Scalar period = wrap_max - wrap_min;
        Scalar a = get_coeff(i + a_i);

        if (std::signbit(get_coeff(i + d_i))) // If radius is negative,
        {
          a += period * 0.5; // Reflect across the origin.
        }

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
      }};
    };

    // Implementation of polar coordinates.
    template<typename Derived, typename Coeff1, typename Coeff2,
      std::size_t d_i, std::size_t a_i, std::size_t d2_i, std::size_t x_i, std::size_t y_i>
    struct PolarBase
    {
      static constexpr std::size_t size = 2;
      static constexpr std::size_t dimension = 3;
      static constexpr bool axes_only = false;

      template<typename ... Cnew>
      using Prepend = Coefficients<Cnew..., Derived>;

      template<typename ... Cnew>
      using Append = Coefficients<Derived, Cnew ...>;

      template<typename Scalar>
      using GetCoeff = std::function<Scalar(const std::size_t)>;

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
        to_Euclidean_array = internal::join(
        detail::PolarImpl<Coeff1, Scalar>::template to_Euclidean_array<i, d_i, a_i>,
        detail::PolarImpl<Coeff2, Scalar>::template to_Euclidean_array<i, d_i, a_i>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        from_Euclidean_array = internal::join(
        detail::PolarImpl<Coeff1, Scalar>::template from_Euclidean_array<i, d2_i, x_i, y_i>,
        detail::PolarImpl<Coeff2, Scalar>::template from_Euclidean_array<i, d2_i, x_i, y_i>
      );

      template<typename Scalar, std::size_t i>
      static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
        wrap_array = internal::join(
        detail::PolarImpl<Coeff1, Scalar>::template wrap_array<i, d_i, a_i>,
        detail::PolarImpl<Coeff2, Scalar>::template wrap_array<i, d_i, a_i>
      );
    };
  }

  /// Polar coordinates (Radius, Angle).
  template<typename Traits>
  struct Polar<OpenKalman::Distance, OpenKalman::Circle<Traits>>
    : detail::PolarBase<Polar<OpenKalman::Distance, OpenKalman::Circle<Traits>>, OpenKalman::Distance, OpenKalman::Circle<Traits>, 0, 1,  0, 1, 2> {};

  /// Polar coordinates (Angle, Radius).
  template<typename Traits>
  struct Polar<OpenKalman::Circle<Traits>, OpenKalman::Distance>
    : detail::PolarBase<Polar<OpenKalman::Circle<Traits>, OpenKalman::Distance>, OpenKalman::Circle<Traits>, OpenKalman::Distance, 1, 0,  2, 0, 1> {};

  /// Polar is a coefficient.
  template<typename Coeff1, typename Coeff2>
  struct is_coefficient<Polar<Coeff1, Coeff2>> : std::true_type {};

  template<typename Coeff1a, typename Coeff2a, typename Coeff1b, typename Coeff2b>
  struct is_equivalent<Polar<Coeff1a, Coeff2a>, Polar<Coeff1b, Coeff2b>>
    : std::integral_constant<bool, is_equivalent_v<Coeff1a, Coeff1b> and is_equivalent_v<Coeff2a, Coeff2b>> {};

  namespace internal
  {
    template<typename Coeff1, typename Coeff2, typename...Coeffs>
    struct ConcatenateImpl<Polar<Coeff1, Coeff2>, Coeffs...>
    {
      using type = typename ConcatenateImpl<Coeffs...>::type::template Prepend<Polar<Coeff1, Coeff2>>;
    };
  }


}// namespace OpenKalman

#endif //OPENKALMAN_POLAR_H
