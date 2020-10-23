/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMBASE_HPP
#define OPENKALMAN_TRANSFORMBASE_HPP


namespace OpenKalman
{
  /// The base for all transforms.
  template<typename Derived>
  struct TransformBase
  {
   /**
    * Perform one or more consecutive linearized transforms.
    * @tparam InputDist Input distribution.
    * @tparam T The first tuple containing (1) a LinearTransformation (depending on transform) and (2) zero or more noise terms for that transformation.
    * @tparam Ts A list of tuples containing (1) a LinearTransformation (depending on transform) and (2) zero or more noise terms for that transformation.
    **/
   template<typename InputDist, typename...T_args, typename...Ts, std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto operator()(const InputDist& x, const std::tuple<T_args...>& t, const Ts&...ts) const
    {
      auto y = std::apply([&](const auto&...args) { return this->operator()(x, args...); }, t);
      if constexpr (sizeof...(Ts) > 0)
      {
        return static_cast<const Derived&>(*this)(y, ts...);
      }
      else
      {
        return y;
      }
    }


    /**
     * Perform one or more consecutive linearized transforms, also returning the cross-covariance.
     * @tparam InputDist Input distribution.
     * @tparam T The first tuple containing (1) a LinearTransformation (depending on transform) and (2) zero or more noise terms for that transformation.
     * @tparam Ts A list of tuples containing (1) a LinearTransformation (depending on transform) and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename...T_args, typename...Ts, std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const std::tuple<T_args...>& t, const Ts&...ts) const
    {
      if constexpr (sizeof...(Ts) > 0)
      {
        auto y = std::apply([&](const auto&...args) { return static_cast<const Derived&>(*this)(x, args...); }, t);
        if constexpr (sizeof...(Ts) > 1)
        {
          return static_cast<const Derived&>(*this).transform_with_cross_covariance(y, ts...);
        }
        else
        {
          auto [z, cross] = static_cast<const Derived&>(*this).transform_with_cross_covariance(y, ts...);
          return std::tuple {std::move(z), std::move(cross), std::move(y)};
        }
      }
      else
      {
        return std::apply([&](const auto&...args) {
          return static_cast<const Derived&>(*this).transform_with_cross_covariance(x, args...); }, t);
      }
    }

  };

}

#endif //OPENKALMAN_TRANSFORMBASE_HPP
