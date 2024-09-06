/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref index_to_scalar_constant function.
 */

#ifndef OPENKALMAN_INDEX_TO_SCALAR_CONSTANT_HPP
#define OPENKALMAN_INDEX_TO_SCALAR_CONSTANT_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Converts an \ref index_value to a \ref scalar_constant.
   * \tparam Scalar The scalar type to be converted to
   * \tparam Arg A \ref scalar_constant
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, index_value Arg>
#else
  template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar>, index_value<Arg>, int> = 0>
#endif
  constexpr auto
  index_to_scalar_constant(Arg&& arg)
  {
    if constexpr (static_index_value<Arg>)
      return values::ScalarConstant<Scalar, Arg::value>{};
    else
      return static_cast<Scalar>(std::forward<Arg>(arg));
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_INDEX_TO_SCALAR_CONSTANT_HPP
