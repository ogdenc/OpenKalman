/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Distance class.
 */

#ifndef OPENKALMAN_DISTANCE_HPP
#define OPENKALMAN_DISTANCE_HPP

#include <cmath>
#include <type_traits>
#include <array>
#include "linear-algebra/values/values.hpp"
#include "linear-algebra/values/functions/internal/update_real_part.hpp"
#include "linear-algebra/values/functions/abs.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \struct Distance
   * \brief A non-negative real or integral number, [0,&infin;], representing a distance.
   * \details This is similar to Axis, but wrapping occurs to ensure that values are never negative.
   */
  struct Distance {};


  /**
   * \brief T is a \ref vector_space_descriptor object representing a distance.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distance_vector_space_descriptor = std::same_as<T, Distance>;
#else
  static constexpr bool distance_vector_space_descriptor = std::is_same_v<T, Distance>;
#endif

} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Distance.
   */
  template<>
  struct vector_space_traits<descriptor::Distance>
  {
  private:

    using T = descriptor::Distance;

  public:

    static constexpr auto
    size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    euclidean_size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    collection(const T& t) { return std::array {t}; }


    static constexpr auto
    is_euclidean(const T&) { return std::false_type{}; }


    /*
     * \brief Maps an element to positive coordinates in 1D Euclidean space.
     * \param euclidean_local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T&, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires { {g(start)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T&, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      return g(start);
    }


    /*
     * \brief Maps a coordinate in positive 1D Euclidean space to an element.
     * \details The resulting distance should always be positive, so this function takes the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires { {g(euclidean_start)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T&, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      auto x = g(euclidean_start);
      // The distance component may need to be wrapped to the positive half of the real axis:
      return value::internal::update_real_part(x, value::abs(value::real(x)));
    }


    /*
     * \details The wrapping operation is equivalent to taking the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T&, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires { {g(start)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, const S&>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T&, const Getter& g, const L& local_index, const S& start)
#endif
    {
      auto x = g(start);
      return value::internal::update_real_part(x, value::abs(value::real(x)));
    }


    /*
     * \details The operation is equivalent to setting and then changing to the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T&, const auto& s, const auto& g, const value::value auto& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires { s(x, start); s(g(start), start); }
#else
    template<typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
      value::value<X> and value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const X&, const S&>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, const S&>::type, const S&>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T&, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start)
#endif
    {
      s(value::internal::update_real_part(x, value::abs(value::real(x))), start);
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_DISTANCE_HPP
