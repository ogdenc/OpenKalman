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
 * \brief Definition of \ref get_mdspan function.
 */

#ifndef OPENKALMAN_GET_MDSPAN_HPP
#define OPENKALMAN_GET_MDSPAN_HPP

#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/mdspan-library.hpp"

namespace OpenKalman
{
  /**
   * \brief Get the \ref coordinates::pattern_collection associated with \ref indexible object T.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto
  get_mdspan(T& t)
  {
    using Traits = interface::object_traits<std::remove_cv_t<T>>;
    return stdex::invoke(Traits::get_mdspan, t);
  }


  /**
   * \overload
   * \brief If argument is already an mdspan, return it unchanged.
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  constexpr auto
  get_mdspan(stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> m)
  {
    return std::move(m);
  }


}

#endif
