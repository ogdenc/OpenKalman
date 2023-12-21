/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref indexible_impl.
 */

#ifndef OPENKALMAN_WRAPPABLE_HPP
#define OPENKALMAN_WRAPPABLE_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool wrappable_impl(std::index_sequence<I...>) {
      return ((dynamic_dimension<T, I> or has_untyped_index<T, I + 1>) and ...); }

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_wrappable : std::false_type {};

    template<typename T>
    struct is_wrappable<T, std::enable_if_t<indexible<T> and (index_count<T>::value >= 1)>>
      : std::bool_constant<(detail::wrappable_impl<T>(std::make_index_sequence<index_count_v<T> - 1>{}))> {};
#endif
  }


  /**
   * \brief Specifies that every fixed-size index of T (other than potentially index 0) is euclidean.
   * \details This indicates that T is suitable for wrapping along index 0.
   * \sa get_wrappable
   */
  template<typename T>
#ifdef __cpp_concepts
  concept wrappable = indexible<T> and (index_count_v<T> >= 1) and
    (detail::wrappable_impl<T>(std::make_index_sequence<index_count_v<T> - 1>{}));
#else
  constexpr bool wrappable = detail::is_wrappable<T>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_WRAPPABLE_HPP
