/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref pattern_descriptor_traits.
 */

#ifndef OPENKALMAN_COORDINATE_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_COORDINATE_DESCRIPTOR_TRAITS_HPP

#include <typeindex>
#include "collections/collections.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief Traits for \ref patterns::pattern objects.
   * \details This should only be specialized for user-defined objects.
   * If this class is specialized with T, the library will also define the following:
   * # std::common_type<T, >
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct pattern_descriptor_traits
  {
    static constexpr bool
    is_specialized = false;

#ifdef DOXYGEN_SHOULD_SKIP_THIS

    /**
     * \brief A callable object returning the number of dimensions at compile time (as a \ref values::index).
     */
    static constexpr auto
    dimension = [](const T&) -> values::index auto { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning the number of dimensions after transforming to Euclidean space (as a \ref values::index).
     */
    static constexpr auto
    stat_dimension = [](const T&) noexcept -> values::index auto { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning a bool reflecting whether the \ref patterns::pattern object describes Euclidean coordinates.
     * \details In this case, dimension() == stat_dimension().
     */
    static constexpr auto
    is_euclidean = [](const T&) -> std::convertible_to<bool> auto { return std::false_type {}; };


    /**
     * \brief A callable object returning a unique hash code for type T, generally of type std::size_t.
     * \details Two coordinates will be equivalent if they have the same hash code. Generally, this can be obtained
     * through calling <code>typeid(t).hash_code()</code>.
     */
    static constexpr auto
    hash_code = [](const T&) -> std::convertible_to<std::size_t> { return typeid(T).hash_code(); };


    /**
     * \brief A callable object mapping a range reflecting vector-space data to a corresponding range in a vector space for directional statistics.
     * \details This is the inverse of <code>from_stat_space</code>.
     * \note Disregarded if T is a \ref patterns::euclidean_pattern. In this case, this will be treated as an identity function.
     * \param data_view A range within a data object corresponding to the descriptor
     */
    static constexpr auto
    to_stat_space = [](const T& t, collections::collection_view auto&& data_view) -> collections::collection decltype(auto)
    {
      return collections::views::all(std::forward<decltype(data_view)>(data_view));
    };


    /**
     * \brief A callable object mapping a range in a vector space for directional statistics back to a range corresponding to the original vector space.
     * \details This is the inverse of <code>to_stat_space</code>.
     * \note Disregarded if T is a \ref patterns::euclidean_pattern. In this case, this will be treated as an identity function.
     * \param data_view A collection of elements within a data object in directional-statistics space corresponding to the descriptor
     */
    static constexpr auto
    from_stat_space = [](const T& t, collections::collection_view auto&& data_view) -> collections::collection decltype(auto)
    {
      return collections::views::all(std::forward<decltype(data_view)>(data_view));
    };


    /**
     * \brief A callable object that maps a range reflecting vector-space data to a wrapped range.
     * \details The wrapped range is equivalent to <code>from_stat_space(t, to_stat_space(t, data_view))<code>.
     * \note Optional. This will be disregarded if T is a \ref patterns::euclidean_pattern.
     * Otherwise, if not provided, the library will use <code>from_stat_space(t, to_stat_space(t, data_view))<code>.
     * \param data_view A collection of elements within a data object corresponding to the descriptor
     */
    static constexpr auto
    wrap = [](const T& t, collections::collection_view auto&& data_view) -> collections::collection decltype(auto)
    {
      return collections::views::all(std::forward<decltype(data_view)>(data_view));
    };

#endif

  };

}



#endif
