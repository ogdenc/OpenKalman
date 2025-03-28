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
#include "values/concepts/value.hpp"
#include "values/values.hpp"
#include "values/functions/internal/update_real_part.hpp"
#include "values/math/abs.hpp"
#include "values/functions/internal/constexpr_callable.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \struct Distance
   * \brief A non-negative real or integral number, [0,&infin;], representing a distance.
   * \details This is similar to Axis, but wrapping occurs to ensure that values are never negative.
   */
  struct Distance {};


} // namespace OpenKalman::coordinate


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Distance.
   */
  template<>
  struct coordinate_descriptor_traits<coordinate::Distance>
  {
  private:

    using T = coordinate::Distance;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    euclidean_size(const T&) { return std::integral_constant<std::size_t, 1>{}; };


    static constexpr auto
    is_euclidean(const T&) { return std::false_type{}; }


    static constexpr std::size_t
    hash_code(const T&)
    {
      constexpr auto bits = std::numeric_limits<std::size_t>::digits;
      if constexpr (bits < 32) return 0xBD0A_uz;
      else if constexpr (bits < 64) return 0xBD0A6689_uz;
      else return 0xBD0A668977D34578_uz;
    }


    /*
     * \brief Maps an element to positive coordinates in 1D Euclidean space.
     * \param euclidean_local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      return g(0_uz);
    }


    /*
     * \brief Maps a coordinate in positive 1D Euclidean space to an element.
     * \details The resulting distance should always be positive, so this function takes the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto x = g(0_uz);
      // The distance component may need to be wrapped to the positive half of the real axis:
      return value::internal::update_real_part(x, value::abs(value::real(x)));
    }


    /*
     * \details The wrapping operation is equivalent to taking the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto x = g(0_uz);
      return value::internal::update_real_part(x, value::abs(value::real(x)));
    }


    /*
     * \details The operation is equivalent to setting and then changing to the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const value::value auto& x, const value::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<value::value<X> and value::index<L> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
    {
      s(value::internal::update_real_part(x, value::abs(value::real(x))), 0_uz);
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_DISTANCE_HPP
