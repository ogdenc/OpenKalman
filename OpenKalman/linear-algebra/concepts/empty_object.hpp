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


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, applicability b, std::size_t...Ix>
    constexpr bool has_0_dim(std::index_sequence<Ix...>)
    {
      return (dimension_size_of_index_is<T, Ix, 0, b> or ...);
    }
  }


  /**
   * \brief Specifies that an object is empty (i.e., at least one index is zero-dimensional).
   */
  template<typename T, applicability b = applicability::guaranteed>
#ifdef __cpp_concepts
  concept empty_object =
#else
  constexpr bool empty_object =
#endif
    indexible<T> and (index_count_v<T> != dynamic_size or b != applicability::guaranteed) and
    (index_count_v<T> == dynamic_size or detail::has_0_dim<T, b>(std::make_index_sequence<index_count_v<T>>{}));


}

#endif
