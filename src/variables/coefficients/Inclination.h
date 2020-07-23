/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COEFFICIENTS_INCLINATION_H
#define OPENKALMAN_COEFFICIENTS_INCLINATION_H

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  template<typename Traits = AngleTraits>
  struct Inclination
  {
    static constexpr std::size_t size = 1;
    static constexpr std::size_t dimension = 2;
    static constexpr bool axes_only = false;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    static constexpr Scalar cf = 2 * M_PI / (Traits::template wrap_max<Scalar> - Traits::template wrap_min<Scalar>);

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::cos(get_coeff(i)) * cf<Scalar>; },
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::sin(get_coeff(i)) * cf<Scalar>; }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr
          {
            return std::atan2(get_coeff(i + 1), std::abs(get_coeff(i))) / cf<Scalar>;
          }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr
        {
          constexpr Scalar inclination_max = M_PI / 2;
          constexpr Scalar wrap_mod = inclination_max * 2;
          Scalar a = std::fmod(get_coeff(i) + inclination_max, wrap_mod * 2);
          if (a < 0)
          {
            a += wrap_mod * 2;
          }
          if (a > wrap_mod)
          {
            a = wrap_mod * 2 - a;
          }
          return a - inclination_max;
        }
      };

  };

  /// Inclination is a coefficient.
  template<typename Traits>
  struct is_coefficient<Inclination<Traits>> : std::true_type {};

  template<typename Traits>
  struct is_equivalent<Inclination<Traits>, Inclination<Traits>> : std::true_type {};

  using InclinationAngle = Inclination<>;


} // namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_H
