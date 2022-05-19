/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of RecursiveLeastSquaresTransform.
 */

#ifndef OPENKALMAN_RLSTRANSFORM_HPP
#define OPENKALMAN_RLSTRANSFORM_HPP

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  /**
   * \brief Propagates a recursive least squares error distribution of parameters, with a forgetting factor λ.
   * Useful for parameter estimation, where the parameter is expected to possibly drift over time
   * \tparam Scalar The scalar type.
   */
  template<typename Scalar = double>
  struct RecursiveLeastSquaresTransform : oin::TransformBase<RecursiveLeastSquaresTransform<Scalar>>
  {
  private:

    using Base = oin::TransformBase<RecursiveLeastSquaresTransform<Scalar>>;

  public:

    explicit RecursiveLeastSquaresTransform(const Scalar lambda = 0.9995) : inv_lambda {1/lambda} {}


    using Base::operator();


    /**
     * \brief Apply the RLS transform on an input distribution.
     * \details Any noise distributions are treated as additive.
     * \tparam InputDist The prior distribution.
     * \tparam NoiseDists Zero or more noise distribution.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<distribution InputDist, distribution ... NoiseDists> requires
      (equivalent_to<typename DistributionTraits<InputDist>::TypedIndex,
        typename DistributionTraits<NoiseDists>::TypedIndex>and ...)
#else
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<(distribution<InputDist> and ... and distribution<NoiseDists>) and
        (equivalent_to<typename DistributionTraits<InputDist>::TypedIndex,
        typename DistributionTraits<NoiseDists>::TypedIndex>and ...), int> = 0>
#endif
    auto operator()(const InputDist& x, const NoiseDists& ...ns) const
    {
      const auto scaled_x = GaussianDistribution {mean_of(x), covariance_of(x) * inv_lambda};
      return make_self_contained((scaled_x + ... + ns));
    }


    using Base::transform_with_cross_covariance;


    /**
     * \brief Perform RLS transform, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam NoiseDists Zero or more noise distribution.
     * \return A tuple comprising the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<distribution InputDist, distribution ... NoiseDists> requires
      (equivalent_to<typename DistributionTraits<InputDist>::TypedIndex,
        typename DistributionTraits<NoiseDists>::TypedIndex>and ...)
#else
    template<typename InputDist, typename ... NoiseDists,
      std::enable_if_t<(distribution<InputDist> and ... and distribution<NoiseDists>) and
        (equivalent_to<typename DistributionTraits<InputDist>::TypedIndex,
        typename DistributionTraits<NoiseDists>::TypedIndex>and ...), int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const NoiseDists& ...ns) const
    {
      auto scaled_cov = make_self_contained(covariance_of(x) * inv_lambda);
      const auto scaled_x = GaussianDistribution {mean_of(x), scaled_cov};
      auto y = make_self_contained((scaled_x + ... + ns));
      auto cross = Matrix {scaled_cov};
      return std::tuple {std::move(y), std::move(cross)};
    }

  private:

    const Scalar inv_lambda;

  };

}


#endif //OPENKALMAN_RLSTRANSFORM_HPP
