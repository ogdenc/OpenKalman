/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORM_H
#define OPENKALMAN_LINEARTRANSFORM_H


namespace OpenKalman
{
  /**
   * @brief A linear transformation from one statistical distribution to another.
   */
  struct LinearTransform : internal::LinearTransformBase
  {
  protected:

    using Base = internal::LinearTransformBase;

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

  public:
    /**
     * Linearly transform one statistical distribution to another.
     * @tparam LinTransformation A linear transformation (e.g., class LinearTransformation).
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<
      typename LinTransformation,
      typename InputDist,
      typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_linearized_function<LinTransformation, 1>,
        is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto operator()(
      const LinTransformation& g,
      const InputDist& x,
      const NoiseDist& ...ns) const
    {
      return Base::transform<false>(TransformFunction {g}, x, ns...);
    }

    /**
     * Perform one or more consecutive linear transforms.
     * @tparam InputDist Input distribution.
     * @tparam T The first tuple containing (1) a LinearTransformation and (2) zero or more noise terms for that transformation.
     * @tparam Ts A list of tuples containing (1) a LinearTransformation and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename T, typename...Ts,
      std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto operator()(const InputDist& x, const T& t, const Ts&...ts) const
    {
      auto g = std::get<0>(t);
      auto ns = internal::tuple_slice<1, std::tuple_size_v<T>>(t);
      auto out = std::apply([&](const auto&...args) {
        static_assert(is_linearized_function_v<decltype(g), 1>);
        return Base::transform<false>(TransformFunction {g}, x, args...);
      }, ns);
      if constexpr (sizeof...(Ts) > 0)
      {
        return this->operator()(std::move(out), ts...);
      }
      else
      {
        return out;
      }
    }

    /**
     * Linearly transform one statistical distribution to another, also returning the cross-covariance.
     * @tparam LinTransformation A linear transformation (e.g., class LinearTransformation).
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<
      typename LinTransformation,
      typename InputDist,
      typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_linearized_function<LinTransformation, 1>,
        is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto transform_with_cross_covariance(
      const LinTransformation& g,
      const InputDist& x,
      const NoiseDist& ...ns) const
    {
      return Base::transform<true>(TransformFunction {g}, x, ns...);
    }

    /**
     * Perform one or more consecutive linear transforms, also returning the cross-covariance.
     * @tparam InputDist Input distribution.
     * @tparam T The first tuple containing (1) a LinearTransformation and (2) zero or more noise terms for that transformation.
     * @tparam Ts A list of tuples containing (1) a LinearTransformation and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename T, typename...Ts,
      std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const T& t, const Ts&...ts) const
    {
      auto g = std::get<0>(t);
      auto ns = internal::tuple_slice<1, std::tuple_size_v<T>>(t);
      if constexpr (sizeof...(Ts) > 0)
      {
        auto out = std::apply([&](const auto&...args) {
          static_assert(is_linearized_function_v<decltype(g), 1>);
          return Base::transform<false>(TransformFunction {g}, x, args...);
        }, ns);
        return transform_with_cross_covariance(std::move(out), ts...);
      }
      else
      {
        return std::apply([&](const auto&...args) {
          static_assert(is_linearized_function_v<decltype(g), 1>);
          return Base::transform<true>(TransformFunction {g}, x, args...);
        }, ns);
      }
    }

  };


}


#endif //OPENKALMAN_LINEARTRANSFORM_H
