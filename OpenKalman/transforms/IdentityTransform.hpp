/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_IDENTITYTRANSFORM_HPP
#define OPENKALMAN_IDENTITYTRANSFORM_HPP

namespace OpenKalman
{

  /// An identity transform from one statistical distribution to another.
  struct IdentityTransform : TransformBase<IdentityTransform>
  {
    /**
     * Apply the identity transform on an input distribution. Any noise distributions are treated as additive.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDists Noise distribution.
     **/
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDists>...>, int> = 0>
    auto operator()(const InputDist& x, const NoiseDists&...ns) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");
      return strict((x + ... + ns));
    }

    /**
     * Perform identity transform, also returning the cross-covariance.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDists Noise distributions.
     **/
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDists>...>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const NoiseDists&...ns) const
    {
      static_assert(std::conjunction_v<is_equivalent<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients>...>,
        "Input and Noise distributions must be the same size and an equivalent type.");
      auto cross = TypedMatrix {covariance(x)};
      return std::tuple {strict((x + ... + ns)), std::move(cross)};
    }

  };

}

#endif //OPENKALMAN_IDENTITYTRANSFORM_HPP
