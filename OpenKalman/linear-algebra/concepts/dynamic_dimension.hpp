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
 * \brief Definition for \ref dynamic_dimension.
 */

#ifndef OPENKALMAN_DYNAMIC_DIMENSION_HPP
#define OPENKALMAN_DYNAMIC_DIMENSION_HPP

#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t N, typename = void>
    struct is_dynamic_dimension : std::false_type {};

    template<typename T, std::size_t N>
    struct is_dynamic_dimension<T, N, std::enable_if_t<indexible<T> and index_dimension_of<T, N>::value == dynamic_size>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T's index N has a dimension defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept dynamic_dimension = indexible<T> and (index_dimension_of_v<T, N> == dynamic_size);
#else
  constexpr bool dynamic_dimension = detail::is_dynamic_dimension<T, N>::value;
#endif


}

#endif
