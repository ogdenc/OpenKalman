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
 * \brief Header file for compatibility definitions equivalent to those in header &lt;span&gt;.
 */

#ifndef OPENKALMAN_COMPATIBILITY_SPAN_HPP
#define OPENKALMAN_COMPATIBILITY_SPAN_HPP

#ifdef __cpp_lib_span
#include <span>
#else
#include "std-lib-reference/span-tcbrindle/include/tcb/span.hpp"
#endif

#include "basics/compatibility/ranges/range-access.hpp"

namespace OpenKalman::stdex
{
#ifdef __cpp_lib_span
  using std::span;
  using std::dynamic_extent;
  using std::as_bytes;
  using std::as_writable_bytes;
#else
  using tcb::span;
  using tcb::dynamic_extent;
  using tcb::as_bytes;
  using tcb::as_writable_bytes;
#endif

}


#ifndef __cpp_lib_span
#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdex::ranges
#endif
{
  template<typename T, std::size_t Extent>
  constexpr bool enable_borrowed_range<OpenKalman::stdex::span<T, Extent>> = true;

  template<typename T, std::size_t Extent>
  constexpr bool enable_view<OpenKalman::stdex::span<T, Extent>> = true;

}
#endif




#endif
