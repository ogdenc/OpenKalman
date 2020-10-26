/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_AXIS_H
#define OPENKALMAN_AXIS_H

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  struct Axis
  {
    static constexpr std::size_t size = 1;
    static constexpr std::size_t dimension = 1;
    static constexpr bool axes_only = true;

    /// A difference between two Axis values is also on an Axis, so there is no wrapping.
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
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), size>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(i); }
      };

    template<typename Scalar, std::size_t i>
    static constexpr std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), size>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(s, i); }
      };

  };

  /// Axis is a coefficient.
  template<>
  struct is_coefficients<Axis> : std::true_type {};

  template<>
  struct is_equivalent<Axis, Axis> : std::true_type {};


} // namespace OpenKalman


#endif //OPENKALMAN_AXIS_H
