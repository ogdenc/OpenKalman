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
 * \brief Definition for \ref dimension_size_of_index_is.
 */

#ifndef OPENKALMAN_DIMENSION_SIZE_OF_INDEX_IS_HPP
#define OPENKALMAN_DIMENSION_SIZE_OF_INDEX_IS_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t index, std::size_t value, typename = void>
    struct dimension_size_of_index_is_impl : std::false_type {};

    template<typename T, std::size_t index, std::size_t value>
    struct dimension_size_of_index_is_impl<T, index, value, std::enable_if_t<
      index_dimension_of<T, index>::value == value>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a given index of T has a specified size.
   * \details If <code>b == Applicability::permitted</code>, then the concept will apply if there is a possibility that
   * the specified index of <code>T</code> is <code>value</code>.
   */
  template<typename T, std::size_t index, std::size_t value, Applicability b = Applicability::guaranteed>
#ifdef __cpp_concepts
  concept dimension_size_of_index_is = (index_dimension_of_v<T, index> == value) or
#else
  constexpr bool dimension_size_of_index_is = detail::dimension_size_of_index_is_impl<T, index, value>::value or
#endif
    (b == Applicability::permitted and (value == dynamic_size or dynamic_dimension<T, index>));


} // namespace OpenKalman

#endif //OPENKALMAN_DIMENSION_SIZE_OF_INDEX_IS_HPP
