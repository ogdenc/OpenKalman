/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_ANGLE_H
#define OPENKALMAN_ANGLE_H

namespace OpenKalman
{
  struct AngleTraits
  {
    template<typename Scalar>
    static constexpr Scalar wrap_max = std::numbers::pi_v<Scalar>;

    template<typename Scalar>
    static constexpr Scalar wrap_min = -std::numbers::pi_v<Scalar>;
  };

  using Angle = Circle<AngleTraits>;


  struct AnglePositiveRadiansTraits
  {
    template<typename Scalar>
    static constexpr Scalar wrap_max = std::numbers::pi_v<Scalar> * 2;

    template<typename Scalar>
    static constexpr Scalar wrap_min = 0.;
  };

  using AnglePositiveRadians = Circle<AnglePositiveRadiansTraits>;


  struct AngleDegreesTraits
  {
    template<typename Scalar>
    static constexpr Scalar wrap_max = 360.;

    template<typename Scalar>
    static constexpr Scalar wrap_min = 0.;
  };

  using AngleDegrees = Circle<AngleDegreesTraits>;

} // namespace OpenKalman


#endif //OPENKALMAN_ANGLE_H
