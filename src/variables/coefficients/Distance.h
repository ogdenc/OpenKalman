/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_DISTANCE_H
#define OPENKALMAN_DISTANCE_H

#include <cmath>
#include <array>
#include <functional>
#include "variables/support/Traits.h"

namespace OpenKalman
{
  struct Distance
  {
    static constexpr std::size_t size = 1;
    static constexpr std::size_t dimension = 1;
    static constexpr bool axes_only = false;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    ///A vector in which all coefficients are axes already represents a point in Euclidean space. No action taken.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return get_coeff(i); }
      };

    ///A vector in which all coefficients are axes already represents a point in Euclidean space. No action taken.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::abs(get_coeff(i)); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) constexpr { return std::abs(get_coeff(i)); }
      };
  };

  /// Radius is a coefficient.
  template<>
  struct is_coefficient<Distance> : std::true_type {};

  template<>
  struct is_equivalent<Distance, Distance> : std::true_type {};


} // namespace OpenKalman


#endif //OPENKALMAN_DISTANCE_H
