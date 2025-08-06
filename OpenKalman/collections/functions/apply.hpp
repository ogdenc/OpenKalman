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
 * \brief Definition for \ref collections::apply.
 */

#ifndef OPENKALMAN_COLLECTIONS_APPLY_HPP
#define OPENKALMAN_COLLECTIONS_APPLY_HPP

#include "values/values.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename F, typename T, std::size_t...i>
    constexpr decltype(auto)
    apply_impl(F&& f, T&& t, std::index_sequence<i...>)
    {
      return stdcompat::invoke(std::forward<F>(f), get(std::forward<T>(t), std::integral_constant<std::size_t, i>{})...);
    }
  }


  /**
   * \brief A generalization of std::apply
   * \details This function takes a fixed-size \ref collection and applies its elements as arguments of a function.
   */
#ifdef __cpp_concepts
  template<typename F, collection T> requires values::fixed_number_compares_with<size_of<T>, dynamic_size, std::not_equal_to<>>
#else
  template<typename F, typename T, std::enable_if_t<
    collection<T> and values::fixed_number_compares_with<size_of<T>, dynamic_size, std::not_equal_to<>>, int> = 0>
#endif
  constexpr decltype(auto)
  apply(F&& f, T&& t)
  {
    return detail::apply_impl(std::forward<F>(f), std::forward<T>(t), std::make_index_sequence<size_of_v<T>>{});
  }


}

#endif
