/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
     * \tparam InputDist Input distribution.
     * \tparam NoiseDists Noise distribution.
     **/
#ifdef __cpp_concepts
    template<distribution InputDist, distribution ... NoiseDists> requires
      (equivalent_to<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients> and ...)
#else
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<(distribution<InputDist> and ... and distribution<NoiseDists>) and
        (equivalent_to<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients> and ...), int> = 0>
#endif
    auto operator()(const InputDist& x, const NoiseDists&...ns) const
    {
      return make_self_contained((x + ... + ns));
    }

    /**
     * Perform identity transform, also returning the cross-covariance.
     * \tparam InputDist Input distribution.
     * \tparam NoiseDists Noise distributions.
     **/
#ifdef __cpp_concepts
    template<distribution InputDist, distribution ... NoiseDists> requires
      (equivalent_to<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients> and ...)
#else
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<(distribution<InputDist> and ... and distribution<NoiseDists>) and
        (equivalent_to<typename DistributionTraits<InputDist>::Coefficients,
        typename DistributionTraits<NoiseDists>::Coefficients> and ...), int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const NoiseDists&...ns) const
    {
      auto cross = Matrix {covariance_of(x)};
      return std::tuple {make_self_contained((x + ... + ns)), std::move(cross)};
    }

  };

}

#endif //OPENKALMAN_IDENTITYTRANSFORM_HPP
