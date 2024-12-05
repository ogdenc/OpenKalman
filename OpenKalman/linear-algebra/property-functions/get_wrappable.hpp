/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_wrappable function.
 */

#ifndef OPENKALMAN_GET_WRAPPABLE_HPP
#define OPENKALMAN_GET_WRAPPABLE_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool get_wrappable_impl(const T& t, std::index_sequence<I...>)
    {
      return (get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<I + 1>(t)) and ...);
    }
  }


  /**
   * \brief Determine whether T is wrappable (i.e., all its dimensions other than potentially 0 are euclidean).
   * \tparam T A matrix or array
   * \todo Is this necessary?
   * \sa wrappable
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T>
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
#endif
  constexpr bool get_wrappable(const T& t)
  {
    if constexpr (value::fixed<decltype(count_indices(t))>)
    {
      constexpr std::size_t count = std::decay_t<decltype(count_indices(t))>::value;
      return detail::get_wrappable_impl(t, std::make_index_sequence<count - 1> {});
    }
    else
    {
      for (std::size_t i = 1; i < count_indices(t); ++i)
        if (not get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor(t, i))) return false;
      return true;
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_WRAPPABLE_HPP
