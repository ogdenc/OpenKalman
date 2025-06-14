/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref coordinate_descriptor_traits.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
#define OPENKALMAN_VECTOR_SPACE_TRAITS_HPP

#include <type_traits>
#include <typeindex>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "collections/concepts/collection_view.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief Traits for \ref coordinates::pattern objects.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct coordinate_descriptor_traits
  {
    static constexpr bool
    is_specialized = false;


    /**
     * \brief A callable object returning the number of dimensions at compile time (as a \ref values::index).
     */
    static constexpr auto
    dimension =
      [](const T&) noexcept
#ifdef __cpp_concepts
        -> values::index auto
#endif
      { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning the number of dimensions after transforming to Euclidean space (as a \ref values::index).
     */
    static constexpr auto stat_dimension = [](const T&) noexcept -> values::index auto
      { return std::integral_constant<std::size_t, 0_uz>{}; };


    /**
     * \brief A callable object returning a bool reflecting whether the \ref coordinates::pattern object describes Euclidean coordinates.
     * \details In this case, dimension() == stat_dimension().
     */
    static constexpr auto
    is_euclidean =
      [](const T&) noexcept
#ifdef __cpp_concepts
        -> std::convertible_to<bool> auto
#endif
      { return std::false_type {}; };

    /**
     * \brief A callable object returning a unique hash code for type T, of type std::size_t.
     * \details Two coordinates will be equivalent if they have the same hash code. Generally, this can be obtained
     * through calling <code>typeid(t).hash_code()</code>.
     */
    static constexpr auto
    hash_code = [](const T&) noexcept -> std::size_t { return typeid(T).hash_code(); };


    /**
     * \brief A callable object mapping a range reflecting vector-space data to a corresponding range in a vector space for directional statistics.
     * \details This is the inverse of <code>from_stat_space</code>.
     * \note Disregarded if T is a \ref coordinates::euclidean_pattern. In this case, this will be treated as an identity function.
     * \param data_view A range within a data object corresponding to the descriptor
     */
    static constexpr auto
    to_stat_space =
#ifdef __cpp_concepts
      [](const T& t, collections::collection_view auto&& data_view) noexcept -> collections::collection_view decltype(auto)
#else
      [](const T& t, auto&& data_view) noexcept
#endif
    {
      return std::forward<decltype(data_view)>(data_view);
    };


    /**
     * \brief A callable object mapping a range in a vector space for directional statistics back to a range corresponding to the original vector space.
     * \details This is the inverse of <code>to_stat_space</code>.
     * \note Disregarded if T is a \ref coordinates::euclidean_pattern. In this case, this will be treated as an identity function.
     * \param data_view A collection of elements within a data object in directional-statistics space corresponding to the descriptor
     */
    static constexpr auto
    from_stat_space =
#ifdef __cpp_concepts
      [](const T& t, collections::collection_view auto&& data_view) noexcept -> collections::collection_view decltype(auto)
#else
      [](const T& t, auto&& data_view) noexcept
#endif
    {
      return std::forward<decltype(data_view)>(data_view);
    };


    /**
     * \brief A callable object that maps a range reflecting vector-space data to a wrapped range.
     * \details The wrapped range is equivalent to <code>from_stat_space(t, to_stat_space(t, data_view))<code>.
     * If data_view is an std::ranges::output_range, the update should be performed in place.
     * \note Optional. This will be disregarded if T is a \ref coordinates::euclidean_pattern.
     * Otherwise, if not provided, the library will use <code>from_stat_space(t, to_stat_space(t, data_view))<code>.
     * \param data_view A collection of elements within a data object corresponding to the descriptor
     */
    static constexpr auto
    wrap =
#ifdef __cpp_concepts
      [](const T& t, collections::collection_view auto&& data_view) noexcept -> collections::collection_view decltype(auto)
#else
      [](const T& t, auto&& data_view) noexcept
#endif
    {
      return std::forward<decltype(data_view)>(data_view);
    };

  };


  /**
   * \brief Traits for \ref values::index.
   */
#ifdef __cpp_concepts
  template<values::index T>
  struct coordinate_descriptor_traits<T>
#else
  template<typename T>
  struct coordinate_descriptor_traits<T, std::enable_if_t<values::index<T>>>
#endif
  {
    static constexpr bool is_specialized = true;

    using scalar_type = values::number_type_of_t<T>;

    static constexpr auto
    dimension = [](const T& t) { return t; };

    static constexpr auto
    stat_dimension = [](const T& t) { return t; };

    static constexpr auto
    is_euclidean = [](const T&) { return std::true_type {}; };

    static constexpr auto
    hash_code = [](const T& t) -> std::size_t { return t; };

  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_VECTOR_SPACE_TRAITS_HPP
