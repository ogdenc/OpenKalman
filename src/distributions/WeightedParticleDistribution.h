/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_WEIGHTEDPARTICLEDISTRIBUTION_H
#define OPENKALMAN_WEIGHTEDPARTICLEDISTRIBUTION_H

#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include "distributions/ParticleDistribution.h"

namespace OpenKalman {

    template<typename WeightScalar = double, typename... OtherProperties>
    struct WeightedParticleDistribution: ParticleDistribution<WeightScalar, OtherProperties...>
    {
    protected:
        using Parent = ParticleDistribution<WeightScalar, OtherProperties...>
        using Parent::Properties;

    public:
        const auto normalizeWeights()
        {
            auto sum = std::accumulate(
                    begin(),
                    end(),
                    WeightScalar {0},
                    [](const WeightScalar& a, const Properties& p) -> WeightScalar {
                        return a + p.template get<0>();
                    }
            );

            std::for_each(begin(), end(), [](const Properties& p) { p.template get<0>() /= sum; });
            return *this;
        }

    };

}

#endif //OPENKALMAN_WEIGHTEDPARTICLEDISTRIBUTION_H
