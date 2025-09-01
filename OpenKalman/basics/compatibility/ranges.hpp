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

#ifdef __cpp_lib_ranges
#include <ranges>
#endif

namespace OpenKalman::stdcompat::ranges {}

#include "common.hpp"
#include "language-features.hpp"
#include "core-concepts.hpp"
#include "invoke.hpp"
#include "comparison.hpp"
#include "internal/movable_box.hpp"
#include "iterator.hpp"

#include "ranges/range-access.hpp"
#include "ranges/range-concepts.hpp"
#include "ranges/functional.hpp"

#include "views/view_interface.hpp"
#include "views/view-concepts.hpp"
#include "views/range_adaptor_closure.hpp"
#include "views/ref_view.hpp"
#include "views/owning_view.hpp"
#include "views/all.hpp"
#include "views/empty.hpp"
#include "views/single.hpp"
#include "views/iota.hpp"
#include "views/transform.hpp"
#include "views/reverse.hpp"

#include "views/repeat.hpp"
#include "views/concat.hpp"
#include "views/to.hpp"

#endif

