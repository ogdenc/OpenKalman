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

namespace OpenKalman
{
  struct Distance
  {
    static constexpr std::size_t size = 1;
    static constexpr std::size_t dimension = 1;
    static constexpr bool axes_only = false;

    /// A difference between two distances can be positive or negative, and is treated as Axis.
    /// See David Frederic Crouse, Cubature/Unscented/Sigma Point Kalman Filtering with Angular Measurement Models,
    /// 18th Int'l Conf. on Information Fusion 1553, 1555 (2015).
    using difference_type = Coefficients<Axis>;

    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;

    template<typename Scalar>
    using SetCoeff = std::function<void(const Scalar, const std::size_t)>;

    ///A vector in which all coefficients are axes already represents a point in Euclidean space. No action taken.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), dimension>
      to_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };

    ///A vector in which all coefficients are axes already represents a point in Euclidean space. No action taken.
    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      from_Euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return std::abs(get_coeff(i)); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return std::abs(get_coeff(i)); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(std::abs(s), i); }
      };

  };

  /// Radius is a coefficient.
  template<>
  struct is_coefficient<Distance> : std::true_type {};

  template<>
  struct is_equivalent<Distance, Distance> : std::true_type {};


} // namespace OpenKalman


#endif //OPENKALMAN_DISTANCE_H
