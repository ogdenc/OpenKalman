/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref internal::collection.
 */

#ifndef OPENKALMAN_COLLECTION_HPP
#define OPENKALMAN_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include <iterator>
#endif
#include "basics/global-definitions.hpp"

namespace OpenKalman::internal
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_collection_impl_std_begin : std::false_type {};

    template<typename T>
    struct is_collection_impl_std_begin<T, std::void_t<
        decltype(*std::begin(std::declval<T>())),
        decltype(std::begin(std::declval<T>()) + std::declval<std::ptrdiff_t>()),
        decltype(std::begin(std::declval<T>()) - std::declval<std::ptrdiff_t>()),
        decltype(std::begin(std::declval<T>())[std::declval<std::ptrdiff_t>()])>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_collection_impl_begin : std::false_type {};

    template<typename T>
    struct is_collection_impl_begin<T, std::void_t<
        decltype(*begin(std::declval<T>())),
        decltype(begin(std::declval<T>()) + std::declval<std::ptrdiff_t>()),
        decltype(begin(std::declval<T>()) - std::declval<std::ptrdiff_t>()),
        decltype(begin(std::declval<T>())[std::declval<std::ptrdiff_t>()])>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_collection_impl_std_end : std::false_type {};

    template<typename T>
    struct is_collection_impl_std_end<T, std::void_t<decltype(std::end(std::declval<T>()))>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_collection_impl_end : std::false_type {};

    template<typename T>
    struct is_collection_impl_end<T, std::void_t<decltype(end(std::declval<T>()))>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_collection_impl_std_size : std::false_type {};

    template<typename T>
    struct is_collection_impl_std_size<T, std::void_t<decltype(std::size(std::declval<T>()))>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_collection_impl_size : std::false_type {};

    template<typename T>
    struct is_collection_impl_size<T, std::void_t<decltype(size(std::declval<T>()))>>
      : std::true_type {};

  } // namespace detail
#endif 


  /**
   * \brief An object describing a collection of objects.
   * \details This will be a \ref internal::tuple_like "tuple_like" object or a sized std::ranges::random_access_range.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref)
  concept collection = internal::tuple_like<T> or
    (std::ranges::random_access_range<std::remove_cvref_t<T>> and std::ranges::sized_range<std::remove_cvref_t<T>>);
#else
  constexpr bool collection =
    internal::tuple_like<T> or
      ((detail::is_collection_impl_begin<std::decay_t<T>>::value or detail::is_collection_impl_std_begin<std::decay_t<T>>::value) and
        (detail::is_collection_impl_end<std::decay_t<T>>::value or detail::is_collection_impl_std_end<std::decay_t<T>>::value) and
        (detail::is_collection_impl_size<std::decay_t<T>>::value or detail::is_collection_impl_std_size<std::decay_t<T>>::value));
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
