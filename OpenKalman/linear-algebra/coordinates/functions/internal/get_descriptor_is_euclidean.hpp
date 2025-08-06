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
 * \internal
 * \brief Definition for \ref get_descriptor_is_euclidean.
 */

#ifndef OPENKALMAN_GET_DESCRIPTOR_IS_EUCLIDEAN_HPP
#define OPENKALMAN_GET_DESCRIPTOR_IS_EUCLIDEAN_HPP

#include "values/values.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \internal
   * \brief Get the size of \ref coordinates::descriptor Arg
   */
#ifdef __cpp_concepts
  template<descriptor Arg>
  constexpr OpenKalman::internal::boolean_testable decltype(auto)
#else
  template<typename Arg, std::enable_if_t<descriptor<Arg>, int> = 0>
  constexpr auto
#endif
  get_descriptor_is_euclidean(const Arg& arg)
  {
    if constexpr (interface::coordinate_descriptor_traits<Arg>::is_specialized)
    {
      return interface::coordinate_descriptor_traits<Arg>::is_euclidean(arg);
    }
    else
    {
      static_assert(values::index<Arg>);
      return std::true_type{};
    }
  }


}


#endif
