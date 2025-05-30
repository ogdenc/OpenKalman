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
 * \brief Definitions implementing features of the c++ ranges library for compatibility.
 *
 * \dir ranges
 * \brief std::ranges definitions for compatibility with c++17 or other legacy versions of c++.
 */

#ifndef OPENKALMAN_RANGES_HPP
#define OPENKALMAN_RANGES_HPP

#ifndef __cpp_lib_ranges

#include "ranges/range-access.hpp"
#include "ranges/range-concepts.hpp"

#endif


#endif //OPENKALMAN_RANGES_HPP
