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
 * \internal
 * \brief mdspan policies for \ref constant_object
 */

#ifndef OPENKALMAN_CONSTANT_ACCESSOR_HPP
#define OPENKALMAN_CONSTANT_ACCESSOR_HPP

#include "basics/basics.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An accessor that returns a constant value for every index.
   */
  template<typename ElementType>
  struct constant_accessor
  {
    using element_type = ElementType;
    using reference = element_type;
    using data_handle_type = element_type;
    using offset_policy = constant_accessor;

    constexpr constant_accessor() noexcept = default;

#ifdef __cpp_concepts
    template<stdex::convertible_to<element_type> OtherElementType> requires
      (not std::is_same_v<element_type, OtherElementType>)
#else
    template<typename OtherElementType, std::enable_if_t<
      stdex::convertible_to<OtherElementType, element_type> and
      (not std::is_same_v<element_type, OtherElementType>), int> = 0>
#endif
    constexpr constant_accessor(const constant_accessor<OtherElementType>& other) noexcept {}

    constexpr reference
    access(data_handle_type p, std::size_t) const noexcept { return std::move(p); }

    constexpr data_handle_type
    offset(data_handle_type p, std::size_t) const noexcept { return std::move(p); }

  };


}

#endif
