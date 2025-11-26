/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref dimension_size_of_index_is.
 */

#ifndef OPENKALMAN_DIMENSION_SIZE_OF_INDEX_IS_HPP
#define OPENKALMAN_DIMENSION_SIZE_OF_INDEX_IS_HPP

#include "values/values.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
  /**
   * \brief Specifies that a given index of T has a specified size.
   * \details If <code>b == applicability::permitted</code>, then the concept will apply if there is a possibility that
   * the specified index of <code>T</code> is <code>value</code>.
   * \tparam comp A consteval-callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
  template<typename T, std::size_t index, std::size_t value, auto comp = &stdex::is_eq, applicability b = applicability::guaranteed>
#ifdef __cpp_concepts
  concept dimension_size_of_index_is =
#else
  constexpr bool dimension_size_of_index_is =
#endif
    indexible<T> and
    values::size_compares_with<index_dimension_of<T, index>, std::integral_constant<std::size_t, value>, comp, b>;


}

#endif
