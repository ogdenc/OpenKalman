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
 * \brief Definition for \ref element_type_of.
 */

#ifndef OPENKALMAN_ELEMENT_TYPE_OF_HPP
#define OPENKALMAN_ELEMENT_TYPE_OF_HPP

#include "linear-algebra/traits/get_mdspan.hpp"

namespace OpenKalman
{
  /**
   * \brief The element type (e.g., std::float, std::double, std::complex<double>) of an \ref indexible object.
   * \tparam T A tensor or other array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct element_type_of {};


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T>
  struct element_type_of<T>
#else
  template<typename T>
  struct element_type_of<T, std::enable_if_t<indexible<T>>>
#endif
  {
    using type = typename std::decay_t<decltype(get_mdspan(std::declval<T&>()))>::element_type;
  };


  /**
   * \brief helper template for \ref element_type_of.
   */
  template<typename T>
  using element_type_of_t = typename element_type_of<T>::type;


}

#endif
