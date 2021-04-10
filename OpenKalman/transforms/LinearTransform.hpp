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
 * \brief Definition of LinearTransform.
 */

#ifndef OPENKALMAN_LINEARTRANSFORM_HPP
#define OPENKALMAN_LINEARTRANSFORM_HPP


namespace OpenKalman
{
  /**
   * \brief A linear transformation from one statistical distribution to another.
   */
  class LinearTransform : public internal::LinearTransformBase<LinearTransform>
  {
    using Base = internal::LinearTransformBase<LinearTransform>;
    friend Base;


    /**
     * \internal
     * \brief The underlying transform model for LinearTransform.
     * \tparam Trans The transformation function.
     */
    template<typename Trans>
    struct TransformModel
    {

    private:

      const Trans& transformation;

    public:

      /**
       * \brief Constructor
       * \param t A transformation function.
       */
      TransformModel(const Trans& t) : transformation(t) {}


      /**
       * \tparam InputMean The input mean.
       * \tparam NoiseMean Zero or more noise means.
       * \return A tuple comprising the output mean and the Jacobians corresponding to the input mean and noise terms.
       */
      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...), internal::get_Taylor_term<1>(transformation)(x, n...)};
      }

    };


  };

}


#endif //OPENKALMAN_LINEARTRANSFORM_HPP
