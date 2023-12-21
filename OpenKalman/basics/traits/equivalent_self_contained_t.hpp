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
 * \brief Declaration for \equivalent_self_contained_t.
 */

#ifndef OPENKALMAN_EQUIVALENT_SELF_CONTAINED_T_HPP
#define OPENKALMAN_EQUIVALENT_SELF_CONTAINED_T_HPP


namespace OpenKalman
{

  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   * \sa self_contained, make_self_contained
   * \internal \sa interface::indexible_object_traits
   */
  template<typename T>
  using equivalent_self_contained_t = std::remove_reference_t<decltype(make_self_contained(std::declval<T>()))>;


} // namespace OpenKalman


#endif //OPENKALMAN_EQUIVALENT_SELF_CONTAINED_T_HPP
