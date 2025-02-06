/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref value::internal::get_collection_element.
 */

#ifndef OPENKALMAN_GET_COLLECTION_ELEMENT_HPP
#define OPENKALMAN_GET_COLLECTION_ELEMENT_HPP

#include <type_traits>
#include "basics/internal/collection.hpp"
#include "basics/internal/sized_random_access_range.hpp"
#include "basics/internal/collection_size_of.hpp"
#include "linear-algebra/values/concepts/index.hpp"
#include "linear-algebra/values/traits/fixed_number_of.hpp"

namespace OpenKalman::value::internal
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename Common, typename Arg, std::size_t...Ix> requires
      (... and std::constructible_from<Common, typename std::tuple_element<Ix, std::decay_t<Arg>>::type>)
#else
    template<typename Common, typename Arg, std::size_t...Ix, std::enable_if_t<
      (... and std::is_constructible<Common, typename std::tuple_element<Ix, std::decay_t<Arg>>::type>::value), int> = 0>
#endif
    static constexpr auto
    tuple_to_array(Arg&& arg, std::index_sequence<Ix...>)
    {
      using std::get;
      return std::array<Common, std::tuple_size_v<std::decay_t<Arg>>> {get<Ix>(arg)...};
    }
  } // namespace detail


  /**
   * \internal
   * \brief Get an element of a \ref internal::collection "collection"
   * \tparam Common A common type to which the result of each tuple element will be converted.
   * (Only required if Arg is tuple-like and I is \ref value::dynamic)
   */
#ifdef __cpp_concepts
  template<typename Common = void, OpenKalman::internal::collection Arg, value::index I> requires
      OpenKalman::internal::sized_random_access_range<Arg> or value::fixed<I> or
      requires(Arg&& arg){ detail::tuple_to_array<Common>(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{}); }
#else
  template<typename Common = void, typename Arg, typename I, std::enable_if_t<OpenKalman::internal::collection<Arg> and value::index<I>, int> = 0>
#endif
  constexpr auto
  get_collection_element(Arg&& arg, const I i)
  {
    if constexpr (OpenKalman::internal::sized_random_access_range<Arg>)
    {
      using std::begin;
      return begin(std::forward<Arg>(arg))[static_cast<std::size_t>(i)];
    }
    else if constexpr (value::fixed<I>)
    {
      using std::get;
      return get<value::fixed_number_of_v<I>>(std::forward<Arg>(arg));
    }
    else // if constexpr (internal::tuple_like<Arg> and value::dynamic<I>)
    {
      return detail::tuple_to_array<Common>(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{})[i];
    }
  };

} // OpenKalman::value::internal


#endif //OPENKALMAN_GET_COLLECTION_ELEMENT_HPP
