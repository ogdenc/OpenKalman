/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref nested_object function.
 */

#ifndef OPENKALMAN_NESTED_OBJECT_HPP
#define OPENKALMAN_NESTED_OBJECT_HPP

#include "linear-algebra/concepts/has_nested_object.hpp"

namespace OpenKalman
{
  /**
   * \brief Retrieve a nested object of Arg, if it exists.
   * \tparam i Index of the nested matrix (0 for the 1st, 1 for the 2nd, etc.).
   * \tparam Arg A wrapper that has at least one nested object.
   * \internal \sa interface::indexible_object_traits::nested_object
   */
#ifdef __cpp_concepts
  template<has_nested_object Arg>
#else
  template<typename Arg, std::enable_if_t<has_nested_object<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  nested_object(Arg&& arg)
  {
    return interface::indexible_object_traits<stdcompat::remove_cvref_t<Arg>>::template nested_object(std::forward<Arg>(arg));
  }


}

#endif
