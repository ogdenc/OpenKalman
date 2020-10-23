/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_PARTICLEFILTER_H
#define OPENKALMAN_PARTICLEFILTER_H

#include <algorithm>
#include <random>
#include <type_traits>
#include <Eigen/Dense>

#include "distributions/ParticleDistribution.hpp"
#include "transformations/Transformation.h"


namespace OpenKalman {

    template<typename _StateTransitionModel,
            typename _MeasurementModel>
    struct ParticleFilter
    {
        using StateTransitionModel = _StateTransitionModel;
        using MeasurementModel = _MeasurementModel;

        using State = typename StateTransitionModel::Input_t;
        using Measurement = typename MeasurementModel::Output_t;
        static_assert(std::is_convertible<typename MeasurementModel::Input_t, State>);


    protected:
        const StateTransitionModel& state_transition_model;
        const MeasurementModel& measurement_model;

    public:
        ParticleFilter(
                const StateTransitionModel& state_transition_model,
                const MeasurementModel& measurement_model
        ):
                state_transition_model{state_transition_model},
                measurement_model{measurement_model}
        {}


        template<typename _Distribution>
        _Distribution&
        predict(_Distribution& x) const
        {
            std::generate(x.begin(), x.end(), state_transition_model);
            return x;
        }


        template<typename _Distribution, typename _Noise>
        _Distribution &
        predict(_Distribution& x,
                const _Noise& process_noise) const
        {
            x = predict(x);

            std::random_device rd;
            std::mt19937 gen = std::mt19937(rd());

            _Noise::covariance_t L = process_noise.get_sqrt_covariance();

            const auto add_noise = [&](const State& v) -> const State
            {
                // Create uncorrelated, normalized random variable (mean 0, variance 1)
                State Xnorm;
                for (int i=0; i<Samples_t::dim; i++) {
                    Xnorm(i) = std::normal_distribution<>(0,1)(gen);
                }
                return v + L * Xnorm;
            };

            std::generate(x.begin(), x.end(), add_noise);
            return x;
        }


        /**
         * @brief Update the state, using prior state possibly augmented with measurement noise, propagating variable as Gaussian
         * @tparam Args type of measurement noise (Gaussian or square root form, same dimension as measurement variable)
         * @param x the current state variable (Gaussian), possibly augmented with measurement noise
         * @param z The measurement vector
         * @param args the optional additive process noise
         * @return updated state variable
         */
        template<typename ... Args>
        auto update(
                const typename measurement_transform_t::Input_variable_t& x,
                const measurement_t& z,
                const Args& ... args) const
        {
            const auto& result = measurement_transform.template transform<true>(x, args...);
            const auto& y = std::get<0>(result);
            const auto& P_xy = std::get<1>(result);
            return x.unaugment().kalmanUpdate(y, P_xy, z);
        }


        /**
         * @brief Update the state, using prior state possibly augmented with measurement noise, propagating variable as Gaussian
         * @param x the current state variable (Gaussian), possibly augmented with measurement noise
         * @param z The measurement vector
         * @param aug the measurement noise for augmentation
         * @return updated state variable
         */
        auto update(
                typename measurement_transform_t::Input_variable_t::unaugmented_t& x,
                const measurement_t& z,
                const typename measurement_transform_t::Input_variable_t::augmentation_t& aug) const
        {
            const auto& result = measurement_transform.template transform<true>(typename measurement_transform_t::Input_variable_t(x, aug));
            const auto& y = std::get<0>(result);
            const auto& P_xy = std::get<1>(result);
            return x.kalmanUpdate(y, P_xy, z);
        }

    };

}


#endif //OPENKALMAN_PARTICLEFILTER_H
