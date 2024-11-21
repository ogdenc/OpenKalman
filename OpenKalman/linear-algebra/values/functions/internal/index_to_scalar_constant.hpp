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

#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/index.hpp"
#include "linear-algebra/values/internal-classes/StaticScalar.hpp"


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Converts an \ref value::index to a \ref value::scalar.
   * \tparam Scalar The scalar type to be converted to
   * \tparam Arg A \ref value::scalar
   */
#ifdef __cpp_concepts
  template<value::number Scalar, value::index Arg>
#else
  template<typename Scalar, typename Arg, std::enable_if_t<value::number<Scalar> and value::index<Arg>, int> = 0>
#endif
  constexpr auto
  index_to_scalar_constant(Arg&& arg)
  {
    if constexpr (value::static_index<Arg>)
      return value::StaticScalar<Scalar, Arg::value>{};
    else
      return static_cast<Scalar>(std::forward<Arg>(arg));
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_INDEX_TO_SCALAR_CONSTANT_HPP
