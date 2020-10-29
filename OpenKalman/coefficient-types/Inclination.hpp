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
  struct AngleInclinationTraits
  {
    template<typename Scalar>
    static constexpr Scalar max = M_PI/2;

    template<typename Scalar>
    static constexpr Scalar min = -M_PI/2;
  };


  template<typename Traits = AngleInclinationTraits>
  struct Inclination
  {
    static_assert(Traits::template max<double> > Traits::template min<double>);
    static constexpr std::size_t size = 1;
    static constexpr std::size_t dimension = 2;
    static constexpr bool axes_only = false;

    /// A difference between two Inclination values does not wrap, and is treated as Axis.
    /// See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
    /// 18th Int'l Conf. on Information Fusion 1550, 1555 (2015).
    using difference_type = Coefficients<Axis>;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    template<typename Scalar>
    static constexpr Scalar cf = M_PI / (Traits::template max<Scalar> - Traits::template min<Scalar>);

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return std::cos(get_coeff(i) * cf<Scalar>); },
        [](const GetCoeff<Scalar>& get_coeff) { return std::sin(get_coeff(i) * cf<Scalar>); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff)
          {
            /// #TODO Needs a unit test.
            constexpr Scalar max = Traits::template max<Scalar>;
            constexpr Scalar min = Traits::template min<Scalar>;
            if constexpr (max != -min)
            {
              return std::atan2(get_coeff(i + 1), std::abs(get_coeff(i))) / cf<Scalar>;
            }
            else
            {
              constexpr Scalar range = max - min;
              constexpr Scalar period = range * 2;
              auto a = std::atan2(get_coeff(i + 1), get_coeff(i)) / cf<Scalar>;
              if (a < min) return min - a;
              if (a > range) a = min + period - a;
              return a;
            }
          }
      };

  protected:
    template<typename Scalar>
    static Scalar wrap_impl(const Scalar s)
    {
      /// #TODO Needs a more comprehensive unit test for different min and max values.
      constexpr Scalar max = Traits::template max<Scalar>;
      constexpr Scalar min = Traits::template min<Scalar>;
      if (s >= min and s <= max)
      {
        return s;
      }
      else
      {
        constexpr Scalar range = max - min;
        constexpr Scalar period = range * 2;
        Scalar a = std::fmod(s - min, period);
        if (a < 0) a += period;
        if (a > range) a = period - a;
        return a + min;
      }
    }

  public:
    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return wrap_impl(get_coeff(i)); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(wrap_impl(s), i); }
      };

  };

  using InclinationAngle = Inclination<>;

} // namespace OpenKalman


#endif //OPENKALMAN_COEFFICIENTS_INCLINATION_H
