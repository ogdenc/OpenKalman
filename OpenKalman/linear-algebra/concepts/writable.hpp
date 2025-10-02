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
 * \brief Definition for \ref writable.
 */

#ifndef OPENKALMAN_WRITABLE_HPP
#define OPENKALMAN_WRITABLE_HPP

#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/element_type_of.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct writable_impl : std::false_type {};

    template<typename T>
    struct writable_impl<T, std::enable_if_t<not std::is_const_v<typename element_type_of<T>::type>>>
      : std::true_type {};
  }
#endif


  /**
   * \internal
   * \brief Specifies that T is an \ref indexible object whose elements are writable via the associated mdspan.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept writable = indexible<T> and (not std::is_const_v<element_type_of_t<T>>);
#else
  constexpr bool writable = indexible<T> and detail::writable_impl<T>::value;
#endif


}

#endif
