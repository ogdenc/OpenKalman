/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_IDENTITYTRANSFORM_H
#define OPENKALMAN_IDENTITYTRANSFORM_H

namespace OpenKalman
{

  /// An identity transformation from one statistical distribution to another.
  struct IdentityTransform
  {
    /// Apply the identity transform on an input distribution. Any noise distributions are treated as additive.
    template<typename InputDist, typename ... NoiseDist>
    auto operator()(InputDist&& in, const NoiseDist& ...n) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDist>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");

      auto cross_covariance = TypedMatrix {covariance(in)};
      return std::tuple {strict((in + ... + n)), std::move(cross_covariance)};
    }

  };

}

#endif //OPENKALMAN_IDENTITYTRANSFORM_H
