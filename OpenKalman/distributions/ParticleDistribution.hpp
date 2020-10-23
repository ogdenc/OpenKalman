/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_PARTICLEDISTRIBUTION_H
#define OPENKALMAN_PARTICLEDISTRIBUTION_H

#include <vector>
#include <tuple>

namespace OpenKalman {

    /**
     * @brief Distribution of particles
     *
     * Particles are stored in a vector, and each particle is a tuple of properties.
     *
     * @tparam Properties The properties of each particle
     */
    template<typename... Properties>
    struct ParticleDistribution: std::vector<std::tuple<Properties...>>
    {
    protected:
        using Properties = std::tuple<Properties...>;
    };

}

#endif //OPENKALMAN_PARTICLEDISTRIBUTION_H
