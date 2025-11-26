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
 * \brief Definitions relating to standard c++ library concepts.
 */

#ifndef OPENKALMAN_COMPATIBILITY_CALLABLE_CONCEPTS_HPP
#define OPENKALMAN_COMPATIBILITY_CALLABLE_CONCEPTS_HPP

#include "invoke.hpp"

namespace OpenKalman::stdex
{
#ifdef __cpp_lib_concepts
  using std::invocable;
  using std::regular_invocable;
#else
  template<typename F, typename...Args>
  inline constexpr bool
  invocable = std::is_invocable_v<F, Args...>;


  template<typename F, typename...Args>
  inline constexpr bool
  regular_invocable = invocable<F, Args...>;
#endif

}

#endif