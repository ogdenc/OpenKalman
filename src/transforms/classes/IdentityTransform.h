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

#include "transforms/internal/TransformBase.h"

namespace OpenKalman
{

  /// An identity transformation from one statistical distribution to another.
  template<
    /// Input distribution.
    typename InputDistribution>
  struct IdentityTransform;


  namespace internal
  {
    struct IdentityTransformFunction
    {
      template<typename Dist, typename ... Noise>
      inline auto operator()(const Dist& x, const Noise& ... n) const
      {
        return std::tuple {strict_matrix(x.moment1() + ... + n.moment1()), std::tuple {Noise::Identity()...}};
      }
    };
  }


  template<typename Dist>
  struct IdentityTransform
    : Transform<Dist, typename DistributionTraits<Dist>::Coefficients, internal::IdentityTransformFunction>
  {
  using Base = Transform<Dist, typename DistributionTraits<Dist>::Coefficients, internal::IdentityTransformFunction>;

  protected:
    static constexpr internal::IdentityTransformFunction function;

  public:
    Transform() : Base(function) {}

    /// Augmented case.
    template<typename ... NonlinearNoise, typename ... LinearNoise>
    constexpr auto operator()(const std::tuple<Dist, NonlinearNoise...>& in, const LinearNoise& ... linear_noise) const
    {
      auto sum = [](const auto&...elem) constexpr -> decltype(auto) { return strict_matrix(elem + ...); };
      return (std::apply(sum, in) + ... + linear_noise);
    }

    /// Non-augmented case.
    template<typename ... LinearNoise>
    constexpr auto operator()(const Dist& in, const LinearNoise& ... linear_noise) const
    {
      return (in + ... + linear_noise);
    }

  };

}

#endif //OPENKALMAN_IDENTITYTRANSFORM_H
