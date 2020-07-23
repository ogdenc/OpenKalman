/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_CIRCLE_H
#define OPENKALMAN_CIRCLE_H

#include <cmath>
#include <functional>

namespace OpenKalman
{
  template<typename Traits>
  struct Circle
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
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::cos(get_coeff(i) * cf<Scalar>); },
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::sin(get_coeff(i) * cf<Scalar>); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr
        {
          constexpr Scalar wrap_max = Traits::template wrap_max<Scalar>;
          constexpr Scalar wrap_min = Traits::template wrap_min<Scalar>;
          constexpr Scalar period = wrap_max - wrap_min;
          auto a = std::atan2(get_coeff(i + 1), get_coeff(i)) / cf<Scalar>;
          if constexpr (wrap_min != - M_PI / cf<Scalar>)
          {
            if (a < wrap_min) return a + period; // Generally, this is for positive angle systems where wrap_min is 0.
          }
          return a;
        }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr
        {
          constexpr Scalar wrap_max = Traits::template wrap_max<Scalar>;
          constexpr Scalar wrap_min = Traits::template wrap_min<Scalar>;
          constexpr Scalar period = wrap_max - wrap_min;
          const Scalar a = get_coeff(i);
          if (a >= wrap_min and a < wrap_max)
          {
            return a;
          }
          else
          {
            Scalar ar = std::fmod(a - wrap_min, period);
            if (ar < 0) ar += period;
            return ar + wrap_min;
          }
        }
      };

  };

  /// Circle is a coefficient.
  template<typename Traits>
  struct is_coefficient<Circle<Traits>> : std::true_type {};


  template<typename Traits>
  struct is_equivalent<Circle<Traits>, Circle<Traits>> : std::true_type {};


  struct UnitWrapTraits
  {
    template<typename Scalar>
    static constexpr Scalar wrap_max = 1.;

    template<typename Scalar>
    static constexpr Scalar wrap_min = 0.;
  };

  using UnitWrap = Circle<UnitWrapTraits>;

} // namespace OpenKalman


#endif //OPENKALMAN_CIRCLE_H
