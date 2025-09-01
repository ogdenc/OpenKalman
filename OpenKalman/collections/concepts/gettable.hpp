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
 * \brief Definition for \ref collections::gettable.
 */

#ifndef OPENKALMAN_COLLECTIONS_GETTABLE_HPP
#define OPENKALMAN_COLLECTIONS_GETTABLE_HPP

#include "values/values.hpp"
#include "sized.hpp"
#include "collections/traits/size_of.hpp"

namespace OpenKalman::collections
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<std::size_t i, typename T, typename = void, typename = void>
    struct gettable_impl : std::false_type {};

    template<std::size_t i, typename T>
    struct gettable_impl<i, T,
      std::enable_if_t<not sized<T> or values::fixed_value_compares_with<size_of<T>, i, std::greater<>>>,
      std::void_t<decltype(OpenKalman::internal::generalized_std_get<i>(std::declval<T&>()))>> : std::true_type {};
  }
#endif


  /**
   * \brief T has an element i that is accessible by a get<i>(...) function.
   * \details The get<i>(...) function can be std::get<i>(...), a get<i>(...) member function, or
   * a separately-defined matching get<i>(...) function in T's namespace.
   */
  template<std::size_t i, typename T>
#ifdef __cpp_concepts
  concept gettable =
    (not sized<T> or i < size_of_v<T>) and
    requires { OpenKalman::internal::generalized_std_get<i>(std::declval<T&>()); };
#else
  constexpr bool gettable = detail::gettable_impl<i, T>::value;
#endif


}

#endif
