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
 * \brief Header file for compatibility definitions equivalent to those in header &lt;mdspan&gt;.
 */

#ifndef OPENKALMAN_COMPATIBILITY_MDSPAN_HPP
#define OPENKALMAN_COMPATIBILITY_MDSPAN_HPP

#include "language-features.hpp"
#include "span.hpp"

#ifdef __cpp_lib_mdspan
#include <mdspan>
#else
#ifndef __cpp_lib_span
namespace std::experimental
{
  using OpenKalman::stdex::span;
  //using OpenKalman::stdex::dynamic_extent; // already defined in mdspan reference implementation
  using OpenKalman::stdex::as_bytes;
  using OpenKalman::stdex::as_writable_bytes;
}
#define OPENKALMAN_COMPATIBILITY_SPAN
#endif

#include "std-lib-reference/mdspan-reference-implementation/include/experimental/mdspan"

#undef OPENKALMAN_COMPATIBILITY_SPAN

#endif

namespace OpenKalman::stdex
{
#ifdef __cpp_lib_mdspan
  using std::mdspan;
  using std::extents;
  using std::dextents;
  using std::layout_left;
  using std::layout_right
  using std::layout_stride;
  using std::default_accessor;
#else
  using std::experimental::mdspan;
  using std::experimental::extents;
  using std::experimental::dextents;
  using std::experimental::layout_left;
  using std::experimental::layout_right;
  using std::experimental::layout_stride;
  using std::experimental::default_accessor;
#endif

#ifdef __cpp_lib_mdspan
  using std::full_extent;
  using std::full_extent_t;
  using std::strided_slice;
  using std::submdspan_mapping;
  using std::submdspan_extents;
  using std::submdspan;
#else
  using std::experimental::full_extent;
  using std::experimental::full_extent_t;
  using std::experimental::strided_slice;
  using std::experimental::submdspan_mapping;
  using std::experimental::submdspan_extents;
  using std::experimental::submdspan;
#endif

}

#endif
