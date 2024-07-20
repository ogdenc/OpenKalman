/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <array>
#include <functional>

namespace OpenKalman::vector_space_descriptors
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


  /**
   * \internal
   * \brief traits for Distance.
   */
  template<>
  struct fixed_vector_space_descriptor_traits<Distance>
  {
    static constexpr std::size_t size = 1;
    static constexpr std::size_t euclidean_size = 1;
    static constexpr std::size_t component_count = 1;
    using difference_type = Dimensions<1>;
    static constexpr bool always_euclidean = false;

    /*
     * \brief Maps an element to positive coordinates in 1D Euclidean space.
     * \param euclidean_local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto
    to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
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
    static constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      auto x = g(euclidean_start);
      // The distance component may need to be wrapped to the positive half of the real axis:
      using std::abs;
      return internal::update_real_part(x, abs(internal::constexpr_real(x)));
    }


    /*
     * \details The wrapping operation is equivalent to taking the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto
    get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      auto x = g(start);
      using std::abs;
      return internal::update_real_part(x, abs(internal::constexpr_real(x)));
    }


    /*
     * \details The operation is equivalent to setting and then changing to the absolute value.
     * \param local_index This is assumed to be 0.
     */
#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      using std::abs;
      s(internal::update_real_part(x, abs(internal::constexpr_real(x))), start);
    }

  };


} // namespace OpenKalman::vector_space_descriptors


#endif //OPENKALMAN_DISTANCE_HPP
