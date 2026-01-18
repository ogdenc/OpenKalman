/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref object_traits for std::mdspan.
 */

#ifndef OPENKALMAN_INTERFACES_MDSPAN_LIBRARY_HPP
#define OPENKALMAN_INTERFACES_MDSPAN_LIBRARY_HPP

#include "basics/basics.hpp"
#include "mdspan-object.hpp"
#include "linear-algebra/interfaces/library_interface.hpp"
#include "to_diagonal_mdspan_policies.hpp"
#include "diagonal_of_mdspan_policies.hpp"
#include "transpose_mdspan_policies.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief Library interface to an std::mdspan.
   * \todo Remove this
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct library_interface<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
  {
  };

}


#endif
