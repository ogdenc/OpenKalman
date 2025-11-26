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
 * \brief Definition for \ref internal::slice_specifier.
 */

#ifndef OPENKALMAN_SLICE_SPECIFIER_HPP
#define OPENKALMAN_SLICE_SPECIFIER_HPP

#include "values/values.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T>
    struct is_slice_specifier : std::false_type {};

    template<typename Begin, typename End>
    struct is_slice_specifier<std::tuple<Begin, End>> : std::bool_constant<values::index<Begin> and values::index<End>> {};

    template<>
    struct is_slice_specifier<stdex::full_extent_t> : std::true_type {};

    template<typename OffsetType, typename ExtentType, typename StrideType>
    struct is_slice_specifier<stdex::strided_slice<OffsetType, ExtentType, StrideType>> : std::true_type {};
  }


  /**
   * \brief Specifies that an object is empty (i.e., at least one index is zero-dimensional).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept slice_specifier =
#else
  constexpr inline bool empty_object =
#endif
    values::index<T> or
    detail::is_slice_specifier<std::decay_t<T>>::value;


}

#endif
