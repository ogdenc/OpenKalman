/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of internal::TransformBase.
 */

#ifndef OPENKALMAN_TRANSFORMBASE_HPP
#define OPENKALMAN_TRANSFORMBASE_HPP

#include "collections/concepts/tuple_like.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief The base for all transforms.
   * \tparam Derived The derived class.
   */
  template<typename Derived>
  struct TransformBase;


  template<typename Derived>
  struct TransformBase
  {

   /**
    * \brief Perform one or more consecutive transforms.
    * \tparam InputDist The prior distribution.
    * \tparam T A tuple-like structure containing zero or more arguments (beyond the input distribution) to the
    * first transform (e.g., a tests and zero or more noise distributions).
    * \tparam Ts A list of tuple-like structures, each containing arguments to the second, third, etc. transform.
    * \return The posterior distribution.
    **/
#ifdef __cpp_concepts
    template<distribution InputDist, tuple_like T, tuple_like...Ts>
#else
    template<typename InputDist, typename T, typename...Ts, std::enable_if_t<
      distribution<InputDist> and (tuple_like<T> and ... and tuple_like<Ts>), int> = 0>
#endif
    auto operator()(const InputDist& x, const T& t, const Ts&...ts) const
    {
      auto y = std::apply([&](const auto&...args) { return static_cast<const Derived&>(*this)(x, args...); }, t);

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
     * \brief Perform one or more consecutive transforms, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam T A tuple-like structure containing zero or more arguments (beyond the input distribution) to the
     * first transform (e.g., a tests and zero or more noise distributions).
     * \tparam Ts A list of tuple-like structures, each containing arguments to the second, third, etc. transform.
     * \return A tuple containing the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<distribution InputDist, tuple_like T, tuple_like...Ts>
#else
    template<typename InputDist, typename T, typename...Ts, std::enable_if_t<
      distribution<InputDist> and (tuple_like<T> and ... and tuple_like<Ts>), int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const T& t, const Ts&...ts) const
    {
      if constexpr (sizeof...(Ts) > 0)
      {
        auto y = std::apply([&](const auto&...args) { return static_cast<const Derived&>(*this)(x, args...); }, t);
        return static_cast<const Derived&>(*this).transform_with_cross_covariance(y, ts...);
      }
      else
      {
        return std::apply([&](const auto&...args) {
          return static_cast<const Derived&>(*this).transform_with_cross_covariance(x, args...);
        }, t);
      }
    }

  };

}

#endif //OPENKALMAN_TRANSFORMBASE_HPP
