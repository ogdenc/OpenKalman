/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORM_HPP
#define OPENKALMAN_LINEARTRANSFORM_HPP


namespace OpenKalman
{
  /**
   * @brief A linear transformation from one statistical distribution to another.
   */
  struct LinearTransform : internal::LinearTransformBase<LinearTransform>
  {
  protected:

    using Base = internal::LinearTransformBase<LinearTransform>;
    friend Base;

    /// The underlying transform function model for LinearTransform.
    template<typename LinTransformation>
    struct TransformFunction
    {
    protected:
      const LinTransformation& transformation;

    public:
      TransformFunction(const LinTransformation& t) : transformation(t) {}

      template<typename InputMean, typename ... NoiseMean>
      auto operator()(const InputMean& x, const NoiseMean& ... n) const
      {
        return std::tuple {transformation(x, n...),
        is_linearized_function<LinTransformation, 1>::get_lambda(transformation)(x, n...)};
      }

      static constexpr bool correction = false;
    };


  };


}


#endif //OPENKALMAN_LINEARTRANSFORM_HPP
