/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TRANSFORMBASE_HPP
#define OPENKALMAN_TRANSFORMBASE_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief The base for all transforms.
   * \tparam Derived The derived class.
   */
  template<typename Derived>
  struct TransformBase;


  /**
   * \internal
   * \brief T is a non-empty tuple, pair, array, or other type that can be an argument to std::apply.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept tuple_like = requires { std::tuple_size<T>() == 0; } or requires (T t) { std::get<0>(t); };
#else
  namespace detail
  {
    template<typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template<typename T>
    struct is_tuple_like<T, std::enable_if_t<(std::tuple_size<T>() == 0)>> : std::true_type {};

    template<typename T>
    struct is_tuple_like<T, std::void_t<decltype(std::get<0>(std::declval<T>()))>> : std::true_type {};
  }

  template<typename T>
  inline constexpr bool tuple_like = detail::is_tuple_like<T>::value;
#endif


  template<typename Derived>
  struct TransformBase
  {

   /**
    * \brief Perform one or more consecutive transforms.
    * \tparam InputDist Input distribution.
    * \tparam T A tuple-like structure containing zero or more arguments (beyond the input distribution) to the
    * first transform (e.g., a transformation and zero or more noise distributions).
    * \tparam Ts A list of tuple-like structures, each containing arguments to the second, third, etc. transform.
    * \return The posterior covariance.
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
     * \tparam InputDist Input distribution.
     * \tparam T A tuple-like structure containing zero or more arguments (beyond the input distribution) to the
     * first transform (e.g., a transformation and zero or more noise distributions).
     * \tparam Ts A list of tuple-like structures, each containing arguments to the second, third, etc. transform.
     * \return A tuple containing the posterior covariance and the cross-covariance.
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
