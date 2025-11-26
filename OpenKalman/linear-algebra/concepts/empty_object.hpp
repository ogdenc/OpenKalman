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
 * \brief Definition for \ref empty_object.
 */

#ifndef OPENKALMAN_EMPTY_OBJECT_HPP
#define OPENKALMAN_EMPTY_OBJECT_HPP

#include "linear-algebra/traits/index_count.hpp"
#include "dimension_size_of_index_is.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...is>
    constexpr auto
    empty_object_fixed_index_count(std::index_sequence<is...>)
    {
      return (... or (dimension_size_of_index_is<T, is, 0>));
    }


#ifndef __cpp_concepts
    template<typename T, std::enable_if_t<indexible<T>, int> = 0>
    constexpr auto
    empty_object_impl()
    {
      return detail::empty_object_fixed_index_count<T>(std::make_index_sequence<index_count_v<T>>{});
    }
#endif
  }


  /**
   * \brief Specifies that an object is empty (i.e., at least one index is zero-dimensional).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept empty_object =
    indexible<T> and
    detail::empty_object_fixed_index_count<T>(std::make_index_sequence<index_count_v<T>>{});
#else
  constexpr inline bool empty_object = detail::empty_object_impl<T>();
#endif


}

#endif
